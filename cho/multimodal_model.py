import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel
from timm import create_model
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import easyocr
from PIL import Image
from textblob import TextBlob
from functools import lru_cache
from tqdm import tqdm
import gc
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

gc.collect()
torch.cuda.empty_cache()

# 전역 변수로 EasyOCR 리더 초기화
global_reader = None

def get_easyocr_reader():
    global global_reader
    if global_reader is None:
        try:
            global_reader = easyocr.Reader(['ko', 'en'], gpu=True)  # GPU 사용을 True로 변경
        except Exception as e:
            print(f"Error initializing EasyOCR: {e}")
            global_reader = easyocr.Reader(['ko', 'en'], gpu=False)
    return global_reader

# EasyOCR 모델 파일 경로 설정
os.environ['EASYOCR_MODULE_PATH'] = os.path.expanduser('~/.EasyOCR')

# 이미지 전처리 함수
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    denoised = cv2.fastNlMeansDenoising(enhanced)
    
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return morph

# OCR 함수
def extract_text_from_image(image_path):
    try:
        preprocessed = preprocess_image(image_path)
        reader = get_easyocr_reader()
        with torch.no_grad():  # CUDA 컨텍스트 관리
            results = reader.readtext(preprocessed)
        
        texts = []
        sizes = []
        for (bbox, text, prob) in results:
            if prob > 0.5:
                texts.append(text)
                sizes.append(bbox[2][1] - bbox[0][1])
        
        if not sizes:
            return ""
        
        weights = np.array(sizes) / np.mean(sizes)
        weighted_text = ' '.join([text * int(weight) for text, weight in zip(texts, weights)])
        
        return weighted_text
    except Exception as e:
        print(f"Error in extract_text_from_image: {e}")
        return ""  # 오류 발생 시 빈 문자열 반환

# 텍스트 후처리 함수
@lru_cache(maxsize=1000)
def correct_text(text):
    if not text:
        return ""
    corrected_text = str(TextBlob(text).correct())
    return corrected_text

# 멀티모달 데이터셋 클래스
class MultimodalDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, tokenizer=None, max_len=512):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # 미리 텍스트 추출 및 전처리
        self.processed_texts = self.preprocess_all_texts()

    def preprocess_all_texts(self):
        processed_texts = {}
        for idx, row in self.df.iterrows():
            img_name = f"{self.image_dir}/{row[0]}"
            if os.path.exists(img_name):
                text = extract_text_from_image(img_name)
                text = correct_text(text)
                processed_texts[row[0]] = text
            else:
                processed_texts[row[0]] = ""
        return processed_texts

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = f"{self.image_dir}/{self.df.iloc[idx, 0]}"
        
        try:
            image = preprocess_image(img_name)
            pil_image = Image.fromarray(image).convert('L')
            
            if self.transform:
                pil_image = self.transform(pil_image)
            
            text = self.processed_texts[self.df.iloc[idx, 0]]
            
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            
            return {
                'image': pil_image,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(self.df.iloc[idx, 1], dtype=torch.long) if 'target' in self.df.columns else torch.tensor(0),
                'image_id': self.df.iloc[idx, 0],
                'extracted_text': text
            }
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # 오류 발생 시 더미 데이터 반환
            return {
                'image': torch.zeros((1, 224, 224)),
                'input_ids': torch.zeros(self.max_len, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_len, dtype=torch.long),
                'labels': torch.tensor(0, dtype=torch.long),
                'image_id': self.df.iloc[idx, 0],
                'extracted_text': ""
            }

# 멀티모달 모델 클래스
class MultimodalModel(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalModel, self).__init__()
        self.swin_b = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0, in_chans=1)
        self.bert = AutoModel.from_pretrained('klue/bert-base')
        
        self.image_proj = nn.Linear(self.swin_b.num_features, 512)
        self.text_proj = nn.Linear(self.bert.config.hidden_size, 512)
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.fc = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.1)
                        
    def to(self, device):
        super().to(device)
        self.swin_b = self.swin_b.to(device)
        self.bert = self.bert.to(device)
        return self
        
    def forward(self, image, input_ids, attention_mask):
        image_features = self.swin_b(image)
        image_features = self.image_proj(image_features)

        text_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state[:, 0, :]
        text_features = self.text_proj(text_features)
        
        text_length = attention_mask.sum(dim=1).float() / attention_mask.shape[1]
        text_weight = text_length.unsqueeze(1)
        
        weighted_text_features = text_features * text_weight
        
        attended_features, _ = self.attention(image_features.unsqueeze(0), 
                                              weighted_text_features.unsqueeze(0), 
                                              weighted_text_features.unsqueeze(0))
        attended_features = attended_features.squeeze(0)
        
        combined_features = torch.cat((image_features, attended_features), dim=1)
        combined_features = self.dropout(combined_features)
        
        output = self.fc(combined_features)
        return output

# 학습 함수
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad(set_to_none=True)  # 메모리 사용량 최적화
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

# 평가 함수
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    return avg_loss, f1

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_f1 = -np.Inf

    def __call__(self, val_loss, f1_score, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, f1_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, f1_score, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, f1_score, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). '
                            f'F1 score: {f1_score:.6f}. Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        self.best_f1 = f1_score

def main():
    #torch.multiprocessing.set_start_method('spawn')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 데이터 경로 설정
    data_path = '../data/'

    # 데이터 로드 및 분할
    df = pd.read_csv(f"{data_path}/train_correct_labeling.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    test_df = pd.read_csv(f"{data_path}/sample_submission.csv")

    # 토크나이저 및 변환 준비
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # 그레이스케일 이미지에 맞는 값 사용
    ])

    # 데이터셋 및 데이터로더 준비
    train_dataset = MultimodalDataset(train_df, f"{data_path}/train_preprocessed", transform, tokenizer)
    val_dataset = MultimodalDataset(val_df, f"{data_path}/train_preprocessed", transform, tokenizer)
    test_dataset = MultimodalDataset(test_df, f"{data_path}/test_preprocessed", transform, tokenizer)

    BATCH_SIZE = 32
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 모델, 손실 함수, 옵티마이저 준비
    num_classes = len(df['target'].unique())
    model = MultimodalModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


    # 조기 종료 설정
    early_stopping = EarlyStopping(patience=7, verbose=True, delta=0.001, path='best_model.pth')

    num_epochs = 50
    best_f1 = 0
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)

        scheduler.step()  # 학습률 조정

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1 Score: {val_f1:.4f}")

        # 조기 종료 체크 (validation 에러 기준)
        early_stopping(val_loss, val_f1, model)
        if early_stopping.early_stop:
            print(f"Early stopping. Best validation loss: {early_stopping.val_loss_min:.6f}, "
                f"Best F1 score: {early_stopping.best_f1:.6f}")
            break

        # 최고의 F1 스코어 업데이트 (별도로 추적)
        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"New best F1 score: {best_f1:.4f}")

    # 모델 저장
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    # 텍스트 추출 결과를 저장할 리스트 초기화
    text_output_list = []

    # 텍스트 추출 결과를 리스트에 저장
    for batch in tqdm(train_loader, desc="Extracting texts"):
        for image_id, text in zip(batch['image_id'], batch['extracted_text']):
            text_output_list.append({'ID': image_id, 'TEXTS': text})

    # 리스트를 DataFrame으로 변환
    text_output_df = pd.DataFrame(text_output_list)

    # 추출된 텍스트 결과를 CSV 파일로 저장
    text_output_df.to_csv("extracted_texts.csv", index=False)
    print("Training completed and texts extracted.")

    # 테스트 데이터 추론
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    test_predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting test data"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            
            outputs = model(images, input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.cpu().numpy())

    # 결과 저장
    submission_df = pd.DataFrame({'ID': test_df['ID'], 'target': test_predictions})
    submission_df.to_csv("multimodal_pred.csv", index=False)
    print("Test predictions saved to multimodal_pred.csv")



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()