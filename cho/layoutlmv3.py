import os
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import transformers
from transformers import AutoModelForSequenceClassification, AutoProcessor
from torchsummary import summary
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
transformers.logging.set_verbosity_error()

gc.collect()
torch.cuda.empty_cache()

# LayoutLM 데이터셋 클래스
class LayoutLMDataset(Dataset):
    def __init__(self, csv, img_dir, processor, max_length=512, is_test=False):
        if isinstance(csv, pd.DataFrame):
            self.df = csv
        else:
            self.df = pd.read_csv(csv)
        self.img_dir = img_dir
        self.processor = processor
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx]['ID'])
        
        image = Image.open(img_name).convert("RGB")
        
        encoding = self.processor(image, return_tensors="pt", truncation=True, max_length=self.max_length)
        
        for key in ['input_ids', 'attention_mask', 'bbox']:
            encoding[key] = encoding[key].squeeze(0)
        
        encoding['pixel_values'] = encoding['pixel_values'].squeeze(0)
        
        if not self.is_test:
            label = self.df.iloc[idx]['target']
            encoding['labels'] = torch.tensor(label, dtype=torch.long)
        else:
            encoding['labels'] = torch.tensor(0, dtype=torch.long)  # dummy label for test set
        
        return encoding

# 학습 함수 정의
def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(loader)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds_list.extend(outputs.logits.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(batch['labels'].detach().cpu().numpy())

        pbar.set_description(f"Loss: {loss.item():.4f}")

    train_loss /= len(loader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')

    return train_loss, train_acc, train_f1

def custom_collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    bbox = [item['bbox'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    bbox = pad_sequence(bbox, batch_first=True, padding_value=0)

    # Stack other tensors
    pixel_values = torch.stack(pixel_values)
    labels = torch.stack(labels)

    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'bbox': bbox,
        'labels': labels,
    }

# 검증 함수 정의
def validate(loader, model, loss_fn, device):
    model.eval()
    val_loss = 0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            val_loss += loss.item()
            preds_list.extend(outputs.logits.argmax(dim=1).detach().cpu().numpy())
            targets_list.extend(batch['labels'].detach().cpu().numpy())

    val_loss /= len(loader)
    val_acc = accuracy_score(targets_list, preds_list)
    val_f1 = f1_score(targets_list, preds_list, average='macro')

    return val_loss, val_acc, val_f1

# Early Stopping 클래스
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
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        self.best_f1 = f1_score

# 모델 구조 출력 함수
def print_model_summary(model, input_size):
    summary(model, input_size)

# 메인 실행 코드
if __name__ == "__main__":
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = '../data/'
    model_name = "microsoft/layoutlmv3-base"
    global processor  # processor를 전역 변수로 선언
    LR = 1e-5
    EPOCHS = 2
    BATCH_SIZE = 8
    num_workers = 4

    # 모델과 프로세서 초기화
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=17)
    processor = AutoProcessor.from_pretrained(model_name)
    model = model.to(device)


    # 데이터 로드 및 분할
    df = pd.read_csv(os.path.join(data_path, "augmented_train.csv"))
    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['target'])

    train_dataset = LayoutLMDataset(train_df, os.path.join(data_path, "augmented_train/"), processor)
    val_dataset = LayoutLMDataset(val_df, os.path.join(data_path, "augmented_train/"), processor)
    test_dataset = LayoutLMDataset(os.path.join(data_path, "sample_submission.csv"), 
                                os.path.join(data_path, "test/"), 
                                processor, 
                                is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn)
    # 모델 설정
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    early_stopping = EarlyStopping(patience=7, verbose=True, delta=0.001, path='aug_layoutlmv3_model.pth')

    # 학습 루프
    best_val_f1 = 0
    for epoch in range(EPOCHS):
        train_loss, train_acc, train_f1 = train_one_epoch(train_loader, model, optimizer, loss_fn, device)
        val_loss, val_acc, val_f1 = validate(val_loader, model, loss_fn, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "aug_layoutlmv3_model.pth")
        
        early_stopping(val_loss, val_f1, model)
        if early_stopping.early_stop:
            print(f"Early stopping. Best validation loss: {early_stopping.val_loss_min:.6f}, "
                  f"Best F1 score: {early_stopping.best_f1:.6f}")
            break

    # 테스트 데이터 추론
    model.load_state_dict(torch.load("aug_layoutlmv3_model.pth"))
    model.eval()
    preds_list = []

    for batch in tqdm(test_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        preds_list.extend(outputs.logits.argmax(dim=1).detach().cpu().numpy())

    # 결과 저장
    pred_df = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
    pred_df['target'] = preds_list
    pred_df.to_csv("pred_layoutlmv3.csv", index=False)
    print("Prediction completed and saved to pred_layoutlmv3.csv")