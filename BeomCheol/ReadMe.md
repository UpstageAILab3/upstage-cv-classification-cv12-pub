---

# <아쉬웠던 점>
  - Confusion Matrix에서 단일 분류모델에서 FN, FP를 가져와 OCR을 도전했지만, 출력 크기 오류 때문에 시간을 많이 잡아먹었던 점.
<br>

|                | 예측 긍정 (Positive) | 예측 부정 (Negative) |
|----------------|----------------------|----------------------|
| **실제 긍정 (Positive)** | True Positive (TP)      | False Negative (FN)     |
| **실제 부정 (Negative)** | False Positive (FP)     | True Negative (TN)      |

<br>

---

# <개선하고 싶은 점>
  - 세 가지 모델을 앙상블 하였을 때 배깅 비율을 다르게 했었으면.
  - 데이터 증강쪽을 좀 더 전문적으로 실시했다면.

<br>

---

# <시도해 보고 싶은 점>
  - OCR 관련하여 오류를 고치고 계속 시도해봤다면

<br>
<br>
<br>

# <EfficientNet 모델>
EfficientNet은 이미지 분류와 같은 컴퓨터 비전 태스크를 위한 딥러닝 모델입니다. 이 모델은 기존의 ConvNet 모델들을 효율적으로 확장할 수 있는 방법을 제시하여, 정확도와 연산 효율성의 균형을 최적화합니다.

![이미지 설명](https://velog.velcdn.com/images/9e0na/post/6349a432-9fc1-455d-b881-7b1ca2b4a829/image.png)

## 목차

1. [EfficientNet 소개](#1-efficientnet-소개)
2. [확장 전략: Compound Scaling](#2-확장-전략-compound-scaling)
3. [모델 아키텍처](#3-모델-아키텍처)
4. [효율적인 연산](#4-효율적인-연산)
5. [비교 및 성능](#5-비교-및-성능)
6. [활용 사례](#6-활용-사례)
7. [수학적 표현](#7-수학적-표현)
8. [모델 복잡도 및 파라미터 수](#8-모델-복잡도-및-파라미터-수)

## 1. EfficientNet 소개

EfficientNet은 **성능**과 **효율성**을 동시에 고려하여 개발된 모델로, MobileNet과 ResNet을 포함한 기존의 모델들을 대체할 수 있는 경쟁력 있는 선택지입니다. EfficientNet은 모델의 크기를 확장하면서도, 성능의 감소 없이 계산 복잡도를 줄이는 데 중점을 둡니다.

## 2. 확장 전략: Compound Scaling

EfficientNet의 핵심 개념은 **Compound Scaling**입니다. 이는 모델의 너비(width), 깊이(depth), 해상도(resolution)를 균형 있게 확장하여 최적의 성능을 달성하는 방법입니다.

- **너비(width)**: 네트워크의 채널 수를 증가시켜 더 많은 특징을 학습
- **깊이(depth)**: 네트워크의 레이어 수를 증가시켜 더 깊은 특징을 학습
- **해상도(resolution)**: 입력 이미지의 해상도를 높여 더 세밀한 정보를 학습

이러한 확장 전략은 단순히 하나의 요소만을 확장하는 기존 방법과 달리, 세 가지 요소를 동시에 조정하여 더 효율적인 모델을 만듭니다.

## 3. 모델 아키텍처

EfficientNet은 **MBConv**라는 모바일 친화적인 블록을 사용합니다. 이는 기존의 Convolutional Layer보다 적은 연산으로 동일한 성능을 달성하는 핵심 요소입니다.

| Stage | 해상도     | 채널 수 |
|-------|------------|---------|
| 1     | H/2 × W/2  | 16C     |
| 2     | H/4 × W/4  | 24C     |
| 3     | H/8 × W/8  | 40C     |
| 4     | H/16 × W/16| 80C     |
| 5     | H/32 × W/32| 112C    |
| 6     | H/64 × W/64| 192C    |
| 7     | H/128 × W/128| 320C  |

## 4. 효율적인 연산

EfficientNet은 다음과 같은 요소를 통해 연산 효율성을 극대화합니다:

- **MBConv Block**: 깊이별 분리 합성곱(Depthwise Separable Convolution)과 Squeeze-and-Excitation 기법을 결합하여 성능과 효율성을 높임
- **스케일링 방법**: 너비, 깊이, 해상도의 조합을 통해 연산량 대비 성능을 최적화

## 5. 비교 및 성능

EfficientNet은 기존의 모델들과 비교하여 더 적은 파라미터와 FLOPs(Floating Point Operations)로 유사하거나 더 나은 성능을 제공합니다. 예를 들어, EfficientNet-B7 모델은 ImageNet 데이터셋에서 매우 높은 정확도를 기록하면서도, ResNet-50과 같은 기존 모델들보다 훨씬 적은 연산량을 요구합니다.

## 6. 활용 사례

EfficientNet은 다양한 컴퓨터 비전 태스크에서 사용될 수 있습니다. 예를 들어:

- 이미지 분류
- 객체 탐지(Object Detection)
- 세그멘테이션(Segmentation)

EfficientNet의 효율적인 연산 구조 덕분에, 모바일 기기와 같이 자원이 제한된 환경에서도 탁월한 성능을 발휘할 수 있습니다.

## 7. 수학적 표현

EfficientNet에서 사용되는 주요 연산은 다음과 같습니다:

$$
Y = \text{Conv}(X, W) + B
$$

여기서:
- $X$는 입력 데이터
- $W$는 가중치
- $B$는 편향(bias)
- $\text{Conv}$는 합성곱 연산을 의미

## 8. 모델 복잡도 및 파라미터 수

EfficientNet은 다양한 버전(B0 ~ B7)으로 제공되며, 각 버전은 모델의 복잡도와 성능이 다릅니다. 모델의 파라미터 수와 FLOPs는 이미지 해상도와 네트워크 깊이에 비례하여 증가합니다.

- **EfficientNet-B0**: 기본 모델, 약 5.3M 파라미터
- **EfficientNet-B7**: 확장된 모델, 약 66M 파라미터

EfficientNet은 파라미터 수와 계산량이 증가하면서도 선형적으로 성능이 향상되는 특성을 보입니다.

---

EfficientNet은 성능과 효율성의 균형을 최적화한 모델로, 컴퓨터 비전 분야에서 다양한 응용이 가능합니다.
