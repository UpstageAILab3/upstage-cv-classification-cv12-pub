# Document Type Classification | 문서 타입 분류 | 경진대회
## 팀원
|![최윤설](https://avatars.githubusercontent.com/u/72685362?v=4) |
:--------------------------------------------------------------: 
|[최윤설](https://github.com/developzest)|
|EDA, Pre-processing, Data Augmentation, Modeling|

### EDA

- 학습 데이터에 오분류된 데이터 확인하여 label 수정
  ![alt text](./train_dataset_incorrect_label_image.png)
- 학습 데이터는 대체로 clean한 반면 평가 데이터는 상/하/좌/우 반전 및 회전등이 적용된 noise 데이터
- 직사각형 이미지 99%

### Data Augmentation

- 평가 데이터 셋에 대한 분석을 통해 'augraphy' 의 다양한 기능 적용
  1. 윤곽선 감지를 사용하여 텍스트 선을 감지하고 부드러운 텍스트 취소선, 강조 또는 밑줄 효과 추가
  2. 이미지에 낙서 적용
  3. 입력 용지의 색상 변경
  4. 잉크 번짐 효과 (두 이미지 혼합하여 블리드스루 효과)
  5. 접기 효과
  6. 조명 또는 밝기 그래디언트
  7. 종이 표면에 그림자 효과
  8. 크기 조정(resizing), 뒤집기(flips), 회전(rotation) 등 기본적인 기하학적 변환 적용
- torchvision.transforms v1과 호환되는 v2 사용
- 적용 전
  ![alt text](./before_apply_augraphy_image.png)
- 적용 후
  ![alt text](./after_apply_augraphy_image.png)

### 아쉬웠던 점
- 하다보니 이것저것 시도해보고 싶은게 많았는데 시간이 부족해서 아쉬웠음
- 다음 대회때부터는 대회 오픈하자마자 이것저것 해보기

### 얻은 것
- 다양한 라이브러리를 통해 이미지에 noise를 추가하여 실 데이터와 같이 변형시킬 수 있음
- 매 대회를 진행하면서 느끼는 점은 데이터 전처리의 중요성!

### 시도해보고 싶은 것
- Pytorch lightning + hydra 로의 변환