[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/FVjNDCrt)
# Document Type Classification | 문서 타입 분류
## Team

| ![김나리](https://avatars.githubusercontent.com/u/137861675?v=4) | ![박범철](https://avatars.githubusercontent.com/u/117797850?v=4) | ![서혜교](https://avatars.githubusercontent.com/u/86095630?v=4) | ![조용중](https://avatars.githubusercontent.com/u/5877567?v=4) | ![최윤설](https://avatars.githubusercontent.com/u/72685362?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김나리](https://github.com/narykkim)             |            [박범철](https://github.com/Bomtori)             |            [서혜교](https://github.com/andWHISKEY)             |            [조용중](https://github.com/paanmego)             |            [최윤설](https://github.com/developzest)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

## 0. Overview
### Environment
- _Write Development environment_

### Requirements
- albumentations==1.3.1
- ipykernel==6.27.1
- ipython==8.15.0
- ipywidgets==8.1.1
- jupyter==1.0.0
- matplotlib-inline==0.1.6
- numpy==1.26.0
- pandas==2.1.4
- Pillow==9.4.0
- timm==0.9.12


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
├── cho(조용중)
│   ├── model.ipynb
│   └── preprocessing.ipynb
├── developzest(최윤설)
    ├── developzest_EDA.ipynb
    └── developzest_baseline_code.ipynb
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

- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_
