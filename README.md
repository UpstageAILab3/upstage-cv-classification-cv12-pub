[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/FVjNDCrt)
# Document Type Classification | 문서 타입 분류
## Team

| ![김나리](https://avatars.githubusercontent.com/u/137861675?v=4) | ![박범철](https://avatars.githubusercontent.com/u/117797850?v=4) | ![서혜교](https://avatars.githubusercontent.com/u/86095630?v=4) | ![조용중](https://avatars.githubusercontent.com/u/5877567?v=4) | ![최윤설](https://avatars.githubusercontent.com/u/72685362?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김나리](https://github.com/narykkim)             |            [박범철](https://github.com/Bomtori)             |            [서혜교](https://github.com/andWHISKEY)             |            [조용중](https://github.com/paanmego)             |            [최윤설](https://github.com/developzest)             |
|                            팀장, 발표, EDA, Pre-processing, Data Augmentation, Modeling                             |                            EDA, Modeling, OCR                             |                 EDA, Pre-processing, Data Argumentation, Modeling                   |                            EDA, Pre-processing, Data Augmentation, Modeling, OCR                             |                            EDA, Pre-processing, Data Augmentation, Modeling                       |

## 1. Competiton Info

### Overview

- 소개
이번 대회는 computer vision domain에서 가장 중요한 태스크인 이미지 분류 대회입니다.

이미지 분류란 주어진 이미지를 여러 클래스 중 하나로 분류하는 작업입니다. 이러한 이미지 분류는 의료, 패션, 보안 등 여러 현업에서 기초적으로 활용되는 태스크입니다. 딥러닝과 컴퓨터 비전 기술의 발전으로 인한 뛰어난 성능을 통해 현업에서 많은 가치를 창출하고 있습니다.

<p align="center">
  <img src="https://aistages-api-public-prod.s3.amazonaws.com/app/Files/043d4abf-731a-4187-8eb2-a6a9ce2e9857.png" alt="Description of Image">
</p>

그 중, 이번 대회는 문서 타입 분류를 위한 이미지 분류 대회입니다. 문서 데이터는 금융, 의료, 보험, 물류 등 산업 전반에 가장 많은 데이터이며, 많은 대기업에서 디지털 혁신을 위해 문서 유형을 분류하고자 합니다. 이러한 문서 타입 분류는 의료, 금융 등 여러 비즈니스 분야에서 대량의 문서 이미지를 식별하고 자동화 처리를 가능케 할 수 있습니다.

이번 대회에 사용될 데이터는 총 17개 종의 문서로 분류되어 있습니다. 1570장의 학습 이미지를 통해 3140장의 평가 이미지를 예측하게 됩니다. 특히, 현업에서 사용하는 실 데이터를 기반으로 대회를 제작하여 대회와 현업의 갭을 최대한 줄였습니다. 또한 현업에서 생길 수 있는 여러 문서 상태에 대한 이미지를 구축하였습니다.

<p align="center">
  <img src="https://aistages-api-public-prod.s3.amazonaws.com/app/Files/4038efd9-851f-412a-8729-dca0ff518844.png" alt="Description of Image">
</p>

이번 대회를 통해서 문서 타입 데이터셋을 이용해 이미지 분류를 모델을 구축합니다. 주어진 문서 이미지를 입력 받아 17개의 클래스 중 정답을 예측하게 됩니다. computer vision에서 중요한 backbone 모델들을 실제 활용해보고, 좋은 성능을 가지는 모델을 개발할 수 있습니다. 그 밖에 학습했던 여러 테크닉들을 적용해 볼 수 있습니다.

본 대회는 결과물 csv 확장자 파일을 제출하게 됩니다.

- input : 3140개의 이미지

- output : 주어진 이미지의 클래스

### Timeline

- ex) 2024.07.30 - Start Date
- ex) 2024.08.11 - Final submission deadline

## 2. Components

### Directory

- _Insert your directory structure_

e.g.
```
├── BeomCheol/
│   ├── ReadMe.md
│   └── efficientnet_code.ipynb
├── cho (조용중)
│   ├── README.md
│   ├── layoutmv3.py
│   ├── model.ipynb
│   ├── multimodal_model.ipynb
│   ├── multimodel_model.py
│   └── preprocessing.ipynb
├── developzest (최윤설)
│   ├── developzest_EDA.ipynb
│   └── developzest_baseline_code.ipynb
├── nary/
│   ├── <Nary-related files>
│   └── <More files or subdirectories>
└── README.md

```

## 3. Data descrption

### Dataset overview

- 주어진 학습 데이터에 대한 정보는 다음과 같습니다.
- train [폴더]
    - 1570장의 이미지가 저장되어 있습니다.
- train.csv [파일]
<p align="center">
  <img src="https://aistages-api-public-prod.s3.amazonaws.com/app/Files/832b4982-bd93-4480-936f-3c93a1aee98b.png" alt="Description of Image">
</p>

- 1570개의 행으로 이루어져 있습니다. train/ 폴더에 존재하는 1570개의 이미지에 대한 정답 클래스를 제공합니다.

- ID 학습 샘플의 파일명

- target 학습 샘플의 정답 클래스 번호

### EDA

- class 별 학습데이터의 양이 고르지 못함.
![학습데이터 갯수그래프](./nary/class_num_graph.png)
- 이미지들의 사이즈 분포 시각화.
![이미지 사이즈분포](./nary/image_size_graph.png)

<br>

### Data Processing

- 이미지의 크기를 모델의 최적사이즈에 맞추기좋게 이미지를 가운데에 두고 여백을 주면서 정방형으로 만든 다음 리사이즈.
  ![이미지 전처리](./nary/image_prepro.png)
- 이미지의 회전 바로잡기(Denosing)
  ![Denoising](./nary/denoising.png)
  ![Denoising](./nary/denosing2.png)

  <br>

## 4. Modeling

### Model descrition

- EfficientNet_b4
- SWIN(Shifted Window)
  ![대체 텍스트](./cho/teaser.png)
- ConvNeXt
  ![ConvNeXt V2 아키텍처](./cho/fcmae_convnextv2.png)
- OCR

### Modeling Process

1. Augmentation으로 데이터증강하여 3개의 모델 실험
2. 하이퍼파라미터 튜닝
3. 데이터를 오프라인으로 증강시켜 학습. (약 25000개)
4. 평가데이터 Denoising
5. 훈련데이터중 일부도 Denosing
6. Paddle OCR을 이용한 단어 추출 후 단어사전을 만들어 분류 (3, 4, 7, 14 클래스만 적용함) https://api.wandb.ai/links/narykkim/p2l1gyy0

## 5. Result

### Leader Board

- 리더보드 캡처 넣기
- _Write rank and score_

---
<br>

### 아쉬웠던 점
- 김나리
    - 데이터를 이미지와 문서로 분류해서 다시 분류하는 시스템을 만들고 싶었는 데, 초반에 데이터량이 적다보니 좋은 결과가 나오지 않아 중단했다.
    - 이미지를 오프라인으로 증강한 후에 시도했으면 좋았을텐데 그러지 못해 아쉽다.
- 박범철
    - Confusion Matrix에서 단일 분류모델에서 FN, FP를 가져와 OCR을 도전했지만, 출력 크기 오류 때문에 시간을 많이 잡아먹었던 점.
- 서혜교
- 조용중
    - OCR 부분을 시도했지만 성공적인 결과는 못 얻은점.
    - model 부분에서 Pretrained=True/False 에 대해서 충분히 테스트를 하지 못한점.
- 최윤설
    - 하다보니 이것저것 시도해보고 싶은게 많았는데 시간이 부족해서 아쉬웠음
    - 다음 대회때부터는 대회 오픈하자마자 이것저것 해보기

<br>

### 개선하고 싶은 점
- 김나리
    - 수업을 듣고 바로 대회를 하니 적용을 바로 할수 있어 좋다.
    - 발표준비하면서 모델에 대하여 조금 더 알수 있었다.
    - Augraphy나 layoutLM 등 조원님들이 공부해서 공유해주신 소중한 샘플코드
- 박범철
    - 세 가지 모델을 앙상블 하였을 때 배깅 비율을 다르게 했었으면.
    - 데이터 증강쪽을 좀 더 전문적으로 실시했다면.
- 서혜교
- 조용중
    - 초기 Preprocessing 을 좀더 다양하게 시도했었으면.
    - test 데이터를 꼼꼼히 살펴볼것.
- 최윤설
    - 다양한 라이브러리를 통해 이미지에 noise를 추가하여 실 데이터와 같이 변형시킬 수 있음
    - 매 대회를 진행하면서 느끼는 점은 데이터 전처리의 중요성!

<br>

### 시도해 보고 싶은 점
- 김나리
    - Paddle OCR을 시간관계상 대충하고 지나갔는 데, 좀더 체계적으로 만들어보고싶다. OCR 대회를 기대해본다.
- 박범철
    - OCR 관련하여 오류를 고치고 계속 시도해봤다면
- 서혜교
- 조용중
    - 최신 모델을 논문 참조하여 로우 레벨로 구현해 보는것.
    - 오류가 큰 하위 몇개 클래스들에 대해서 계층적으로 가중치를 주고 모델에 적용해 보는것.
- 최윤설
    - Pytorch lightning + hydra 로의 변환
