# MobileViT: Light-weight, General-purpose Vision Transformer  
NeurIPS 2021 — Mehta & Rastegari

---

## 1. Introduction: Why MobileViT?

모바일 비전 모델은 크게 두 계열로 나뉜다.

| 모델 계열 | 장점 | 단점 |
|----------|------|-------|
| CNN (MobileNet, EfficientNet 등) | 가볍고 빠름, 모바일 친화적 | 국소적인 패턴에는 강하지만, 전역 문맥(global context) 파악이 약함 |
| Transformer (ViT, DeiT 등) | Self-attention으로 전역 의존성 학습에 강함 | 연산량과 메모리 비용이 커서 모바일/엣지 환경에 부적합 |

**MobileViT의 목표**

> MobileNet처럼 가볍게 유지하면서도, ViT처럼 전역 문맥을 보는 모델을 만들 수 없을까?

이를 위해 MobileViT는 **CNN의 local inductive bias**와  
**Transformer의 global dependency modeling**을 하나의 블록(MobileViT block) 안에 결합한 **mobile-friendly hybrid architecture**를 제안한다.

핵심 포인트는 다음과 같다.

- CNN이 잘하는 일:  
  - 작은 커널로 지역적 패턴(local pattern) 추출  
  - 매개변수 효율적, 하드웨어 친화적
- Transformer가 잘하는 일:  
  - Self-attention으로 이미지 전체의 관계(global relations)를 한 번에 모델링  
  - 객체 간 관계, 장면 수준 문맥 이해에 강함
- MobileViT:  
  - “Local → Global → Local” 구조의 MobileViT block을 통해  
    둘의 장점을 동시에 활용

---

## 2. MobileViT Architecture Overview

MobileViT는 전체적으로 보면 **MobileNetV2 스타일의 백본** 위에  
몇몇 stage에서 **MobileViT block**을 끼워 넣은 구조이다.

<img width="779" height="498" alt="image" src="https://github.com/user-attachments/assets/9cc5ffa0-703b-489f-b195-59192a24fb45" />


대략적인 흐름:

- MobileNet stem
- MV2 blocks (MobileNetV2 구조)
- MobileViT block 삽입 (여러 stage)
- Classification head

모델 스케일에 따라 다음과 같은 버전이 있다.

- MobileViT-XXS  
- MobileViT-XS  
- MobileViT-S  

각 버전은 주로 채널 수와 MobileViT block 반복 횟수가 다르며,  
파라미터 수는 대략 1.3M ~ 5.6M 수준으로 매우 가볍다.

---

## 3. MobileViT Block: 구조와 역할

MobileViT block은 이 논문의 핵심 설계 요소이다.  
이 블록의 목적은 다음과 같다.

1. 더 적은 파라미터로 local + global 정보를 동시에 모델링  
2. CNN과 Transformer 둘 다에 잘 맞는 연산 방식 유지 (unfold → processing → fold 구조)  
3. 모바일 환경에서도 사용할 수 있을 정도의 계산량 유지  

### 3.1 전체 아이디어

일반적인 convolution은 다음과 같이 볼 수 있다.

- Unfold (입력 feature를 local patch 단위로 펼치고)
- Local processing (커널과 곱셈/합 연산)
- Fold (다시 공간 구조로 재배치)

MobileViT block은 이 중에서 **local processing** 부분을  
단순한 커널 곱셈이 아니라 **Transformer 기반의 global processing**으로 바꾼 구조라고 볼 수 있다.

즉, “Transformer as Convolution”이라는 관점을 취한다.

---

### 3.2 Step 1: Local Spatial Representation – 왜 먼저 CNN을 쓰는가?

input tensor : (H, W, C)

MobileViT block은 먼저 두 종류의 convolution을 적용한다:

1. nxn convolution
   - 목적: local spatial pattern 인코딩  
   - 주변 픽셀 기반의 지역적 특징 추출

2. 1×1 pointwise convolution  
   - 목적: 채널 차원 확장 (C → d)  
   - 더 풍부한 표현을 위해 high-dimensional space로 projection

--> Local Representation <img width="168" height="32" alt="image" src="https://github.com/user-attachments/assets/9711e7e5-80fe-4672-b0ca-c0f887445c32" />
 을 얻는다.

의미:

- Transformer에 바로 raw feature를 넣는 것보다  
  stabilizing & data efficiency 측면에서 유리  
- CNN의 강한 inductive bias를 먼저 적용  
  → 학습이 안정적으로 진행됨

---

### 3.3 Step 2: Global Representation – Unfold + Transformer

이제 local feature <img width="168" height="32" alt="image" src="https://github.com/user-attachments/assets/0da60a53-0328-4e6a-b663-3cbd3e47c1af" />
 에서 전역 정보를 추출하기 위해  
**unfold → transformer → fold** 과정을 거친다.

#### 3.3.1 Unfold: patch 기반 시퀀스 만들기

- <img width="168" height="32" alt="image" src="https://github.com/user-attachments/assets/cd55cc0c-127a-42d3-90a6-8fe48131bd94" /> 를 겹치지 않는 P×P patch 들로 나눔
- 각 patch를 flatten하여 N개의 non-overlapping flattened patches <img width="194" height="34" alt="image" src="https://github.com/user-attachments/assets/ec37a6ef-9a85-4f0d-a781-5de329f4066a" /> 으로 변환
  - P = wh, N = HW/p, h<=n, w<=n
- <img width="253" height="40" alt="image" src="https://github.com/user-attachments/assets/575c93bf-bfdb-463a-978d-9406f16ec553" />

Transformer에 입력되는 시퀀스가 된다.

#### 3.3.2 Transformer : global dependency 학습

<img width="486" height="42" alt="image" src="https://github.com/user-attachments/assets/ea3dda54-e3ce-4c2f-b402-86f32097311d" /> <br>
- <img width="192" height="33" alt="image" src="https://github.com/user-attachments/assets/de790e67-ea64-4895-8f31-152ee65ce97a" /> <br>
- inter-patch relationship 학습

---

### 3.4 Step 3: Fold + Local-Global Fusion

Transformer 출력 X_G 는 patch 시퀀스 형태이므로  
다시 이미지 구조로 복원해야 한다.

1. Fold  
   - patch를 원래 P×P spatial block으로 reshape  
   - <img width="187" height="34" alt="image" src="https://github.com/user-attachments/assets/820ff911-7dc4-4c57-9cb2-f553ade87d22" /> -> <img width="202" height="35" alt="image" src="https://github.com/user-attachments/assets/c0e790f7-064e-4ac9-9062-7269e6564565" />

2. 1×1 projection conv  
   - d -> C projection

3. Local + Global Fusion  
   - X와 concatenate  
   - nxn conv 로 concatenated feature 를 fuse

<img width="1117" height="394" alt="image" src="https://github.com/user-attachments/assets/0283b5d0-e171-40fc-b484-d3bc6ea84f6f" />

MobileViT의 effective receptive field : HxW

---

### 3.5 MobileViT Block 총정리

MobileViT block은 다음과 같이 요약할 수 있다:

- Local CNN → Global Transformer → Local CNN  
- CNN의 지역적 패턴 학습 + Transformer의 전역 문맥 이해를  
  조합한 경량 구조

역할:

- CNN: edge, texture, local pattern  
- Transformer: long-range dependency, global context  
- Fusion: 둘을 결합한 강력한 특징 제공

---

## 4. Training & Implementation Details (요약)

- Optimizer: AdamW  
- Multi-scale image augmentation  
- Dropout, regularization 적극 활용  
- 모델 크기  
  - MobileViT-XXS: ~1.3M  
  - MobileViT-XS: ~2.3M  
  - MobileViT-S: ~5.6M  
- 설계 목표:  
  - MobileNet 수준의 FLOPs  
  - 더 높은 accuracy, 더 강한 generalization

---

## 5. Experimental Results

### 5.1 ImageNet Classification

| Model | Params(M) | FLOPs(G) | Top-1 (%) |
|-------|-----------|----------|-----------|
| MobileNetV2 | 3.4 | 0.30 | 72.0 |
| MobileNetV3 | 5.4 | 0.22 | 75.2 |
| MobileViT-XXS | 1.3 | 0.36 | 74.8 |
| MobileViT-XS | 2.3 | 0.99 | 78.4 |
| MobileViT-S | 5.6 | 2.0 | 78.8 |

---

### 5.2 COCO Object Detection

| Backbone | Params(M) | FLOPs(G) | AP (%) |
|----------|-----------|----------|--------|
| MobileNetV2 | 4.3 | 0.90 | 22.1 |
| MobileNetV3 | 5.0 | 0.85 | 22.0 |
| MobileViT-XXS | 2.7 | 0.90 | 25.0 |
| MobileViT-XS | 4.5 | 1.35 | 26.9 |
| MobileViT-S | 7.8 | 2.35 | 30.2 |

---

### 5.3 Cityscapes Semantic Segmentation

| Backbone | Params(M) | mIoU (%) |
|----------|-----------|----------|
| MobileNetV2 | 2.1 | 66.1 |
| MobileNetV3 | 2.9 | 60.7 |
| MobileViT-XXS | 1.2 | 67.8 |
| MobileViT-XS | 2.0 | 69.0 |
| MobileViT-S | 4.6 | 73.2 |

---

## 6. Ablation Study Summary

### MobileViT Block 유무 비교
- 동일 FLOPs 대비 MobileViT block을 삽입하면  
  classification / detection / segmentation 모두 향상

### Patch Size P
- 너무 크면 local detail 손실  
- 너무 작으면 연산량 증가  
- P=2 또는 4가 최적

### CNN-only vs Transformer-only vs Hybrid
- CNN-only: local strong, global weak  
- Transformer-only: global strong, mobile-unfriendly  
- MobileViT: 정확히 그 중간점을 잡은 효율적 hybrid

---

## 7. Key Takeaways

- MobileViT는 **모바일 환경에서도 전역 문맥 정보를 학습할 수 있는 lightweight Transformer 기반 비전 모델**  
- MobileNet 대비 훨씬 더 나은 accuracy–efficiency trade-off  
- Classification, detection, segmentation에서 일관되게 우수  

핵심은 MobileViT block:  
“Local CNN → Global Transformer → Local CNN” 구조로  
가벼우면서도 전역 reasoning이 가능한 feature representation 생성

---

## 8. Conclusion

MobileViT는 다음을 증명한다:

> “모바일 환경에서도 적절한 구조 설계를 통해 Transformer 기반 전역 문맥 이해가 충분히 가능하다.”

DETR가 Vision Transformer의 강점을 object detection 분야로 끌고 왔다면,  
MobileViT는 “mobile-scale Vision Transformer”라는 새로운 연구 방향을 열었다.

