# 🧠 DETR : End-to-End Object Detection with Transformers
> Nicolas Carion et al., ECCV 2020

---

## 🎯 1. Motivation — Object Detection의 한계

- 기존 detector들은 **object detection을 set prediction 문제**로 보지 못함.
- 대신 **간접적 방식(surrogate regression/classification)** 으로 접근.
  - **Regression** : anchor나 proposal을 기준으로 박스 좌표 예측
  - **Classification** : 해당 위치에 어떤 객체가 있는지 분류
- → 결국 사람 손으로 설계한 요소들에 의존
  - **Anchor box**, **proposal**, **NMS(Non-Max Suppression)** 등

📉 **문제점**
- 많은 heuristic(경험적 규칙)과 hyperparameter에 의존
- 객체 간 **중복 예측**, **상호 관계 미반영**

---

## 🚀 2. Key Idea — Direct Set Prediction

> “Object detection을 **직접적인 집합 예측 문제(set prediction problem)** 로 본다.”

- **DETR**은 모든 객체를 **한 번에 동시에** 예측함 (End-to-End)
- 핵심 구성요소  
  1. **Bipartite matching loss (Hungarian matching)**  
     → 예측과 정답을 1:1로 대응시켜 nms 등의 후처리 없이 중복 제거를 학습 과정에서 해결 <br>
     → permutation-invariant 보장 (prediction 순서가 바뀌어도 결과가 동일)  
  3. **Transformer encoder-decoder 구조**  
     → self-attention은 시퀀스 내 모든 요소들이 서로를 바라보며(attend) 학습하기 때문에 객체 간의 관계(겹침, 구분, 상호배타성)을 스스로 학습 가능 <br>
     → 객체 간 관계를 전역적으로 모델링
  5. **Parallel decoding**  
     → set 의 크기가 가변적이기 때문에 기존에는 autoregressive를 사용했지만, 시간이 오래 걸리고, 순서에 따라 오류가 누적될 수 있음 (집합에는 순서도 없다)
     → 모든 객체를 병렬로 예측 (non-autoregressive)

---

## ⚙️ 3. DETR Architecture Overview

<img width="800" height="230" alt="image" src="https://github.com/user-attachments/assets/62676cba-f860-4735-9583-6db1e5408031" />


### 🔹 Backbone
<img width="448" height="154" alt="image" src="https://github.com/user-attachments/assets/8526a585-d969-4be2-9787-d2f5d10dd36b" /> <br>
- <img width="150" height="24" alt="image" src="https://github.com/user-attachments/assets/162ab421-f7cd-48b5-8356-6d383012f472" /> → <img width="121" height="28" alt="image" src="https://github.com/user-attachments/assets/b0d5b00e-4f8d-4144-ae4a-9e79a31a4267" />
- C=2048, H, W = H0/32, W0/32
- 이후 1×1 conv로 차원 축소 (C → d)

### 🔹 Transformer Encoder
- <img width="121" height="28" alt="image" src="https://github.com/user-attachments/assets/b0d5b00e-4f8d-4144-ae4a-9e79a31a4267" /> → <img width="124" height="26" alt="image" src="https://github.com/user-attachments/assets/7ad54e83-3650-400e-93c8-002772ed8adc" />
- 입력 feature를 flatten → d x WH sequence  
- 각 레이어: **Multi-Head Self-Attention (MHSA) + FFN**
- **Fixed positional encoding** 추가 → 위치 정보 보존
- 역할: **instance를 분리(separate)**, 이미지 전역 정보 통합

<img width="362" height="377" alt="image" src="https://github.com/user-attachments/assets/d13baae6-1464-49c1-b1c9-110b94542fd9" />


### 🔹 Transformer Decoder
- 순서가 있는 sequence 가 아닌 **순서가 없는 객체 집합(set) 예측**
- 때문에 autoregressive 대신 병렬(parallel)로 N개의 객체를 한번에 예측
- 이때 decoder의 입력 : **object queries** <br>
  → 학습 가능한 벡터(learnable)로, 각각이 하나의 객체를 예측하는 placeholder <br>
  → decoder는 permutation-invariant 해야 하므로 N개의 embedding은 unique <br>
  → **1 query = 1 detection slot**
  → object queries = object query features(학습 시작시 0으로 초기화, learnable) + object query positional encoding(learnable)
- **Self-attention + Encoder-Decoder attention**
  - Self-attention: 객체 간 관계 학습  
  - Encoder-Decoder attention: 이미지와의 관계 학습

### 🔹 Feed-Forward Network (FFN)
- 각 디코더 출력 → 3-layer MLP로 변환  
- 출력: `[bbox (cx, cy, h, w), class probability]`
- bbox는 이미지 대비 normalized
- **“no object” class** 포함 (빈 슬롯 처리)

### 🔹 Auxiliary Decoding Losses
- 각 decoder layer 후에도 FFN + Hungarian loss 적용  
- → 학습 안정화, “적절한 개수의 객체 예측”에 도움  
- 모든 FFN은 파라미터 공유

---

## 🧩 4. Loss Function — Set Prediction Loss

### 사전 지식
#### ① Hungarian matching algorithm 
- 두 집합 사이의 일대일 대응 시 가장 비용이 적게 드는 bipartite matchint(이분 매칭)을 찾는 알고리즘
- 어떤 집합 I와 matching 대상인 집합 J가 있으며 i∈I 를 j∈J에 매칭하는데 드는 비용을 c(i,j)라고 할 때, **σ:I→J로의 일대일 대응 중에서 가장 적은 cost가 드는 matching에 대한 permutation σ을 찾는 것**
<img width="728" height="285" alt="image" src="https://github.com/user-attachments/assets/b76759d0-193a-442e-99b8-c8baa67dd4f7" />
<img width="730" height="279" alt="image" src="https://github.com/user-attachments/assets/8cdabd12-c8f8-4d35-ab81-6cc89daa6562" />

#### ② Bounding box loss
- 기존의 detector들은 anchor 등을 기반으로 예측하기 때문에 예측 bbox 범위가 크게 벗어나지 않는다
- DETR은 initial guess 없이 예측하기 때문에 예측 값의 범주가 크다
- **scale-invariant** 한 GIoU 를 L1 loss와 함께 사용하여 보완
<img width="500" height="375" alt="image" src="https://github.com/user-attachments/assets/6f51738a-a708-4042-8c96-061084f7a61a" />
<img width="323" height="88" alt="image" src="https://github.com/user-attachments/assets/adf9fc48-98ea-43b6-a0c6-1ebbe094350f" />



### ① Matching cost (Hungarian matching)
- 예측과 정답 간 **pairwise cost** 계산<br>
  cost = class_cost + giou_cost + bbox_l1_cost
- <img width="392" height="42" alt="image" src="https://github.com/user-attachments/assets/3b32352f-1b38-47ea-9580-f0790c6fe14a" />
- “no-object(∅)” 매칭은 상수 cost  

### ② Loss 구성
- 앞의 hungarian matching으로 구한 pair를 기반으로 계산
- <img width="488" height="42" alt="image" src="https://github.com/user-attachments/assets/8681c8eb-cec0-4819-be8c-1a0be6fdd288" />
- **Class term** : negative log-likelihood  
- **Box term** : L1 + GIoU (scale invariance 보완)
- **no-object 클래스 가중치 1/10** (class imbalance 완화)

---

## 🧱 5. Training Details

- Optimizer: **AdamW**
  - Transformer lr = 1e-4  
  - Backbone lr = 1e-5  
  - weight decay = 1e-4
- Data Augmentation  
  - Image resize (shortest side 480 ~ 800)  
  - Random crop p=0.5  
  - Dropout 0.1  
- 300 epochs (200 + lr drop × 0.1)
- Dataset: **COCO**

---

## 🔍 6. Results & Analysis

### ✅ 성능
- **Faster R-CNN** 수준의 성능 (COCO AP ≈ 42)  
- **큰 객체(large object)** 에서 특히 우수  
  - 이유: **Transformer의 non-local computation**  
    → 이미지 전역의 feature를 한 번에 통합

### ⚠️ 한계
- 학습 속도가 매우 느림 (300 epochs 필요)  
- 작은 객체(small object)는 상대적으로 성능 하락

---

## 🔬 7. Ablation Study

| 구성요소 | 제거 시 영향 | 의미 |
|-----------|---------------|-------|
| **Encoder 제거** | instance 구분 실패, AP↓ | 전역 self-attention이 객체 분리에 중요 |
| **Decoder 제거** | local patch만 고려, AP↓ | 전역 reasoning 손실 |
| **Positional Encoding 제거** | 약 7–8 AP 하락 | 공간 정보 손실 |
| **FFN 제거** | 약 2–3 AP 하락 | class + box 예측 품질 저하 |
| **Auxiliary Loss 제거** | 학습 불안정, 수렴 느림 | 각 디코더 단계의 보조 supervision 필요 |

📈 **결론:**  
각 구성요소가 모두 성능에 필수적이며,  
특히 **Transformer의 전역 연산 + object query 구조** 가 핵심.

---

## 🧠 8. Key Takeaways

- DETR은 **anchor, NMS, proposal 등 수작업 규칙을 모두 제거**  
- **Transformer의 self-attention** 으로 객체 간 관계를 학습  
- **Hungarian matching loss** 로 1:1 매칭 + permutation invariance 확보  
- **병렬 디코딩 (parallel decoding)** 으로 모든 객체를 동시에 예측  
- 완전한 **End-to-End object detection pipeline** 달성  

---

## 🧭 9. Discussion

| 장점 | 단점 |
|------|------|
| ✔ 단순한 구조 (anchor/NMS X) | ❌ 긴 학습 시간 |
| ✔ 객체 간 관계 학습 | ❌ 작은 객체 탐지 약함 |
| ✔ 전역 문맥 활용 | ❌ 고해상도 이미지 처리 시 계산량 많음 |

---

## 🔮 10. Conclusion

> DETR은 트랜스포머를 이용해 객체 검출을  
> **‘순서 없는 집합 예측(set prediction)’** 문제로 재정의하였다.  
>  
> self-attention으로 객체 간 관계를 학습하고,  
> 헝가리안 매칭을 통해 중복 없는 예측을 달성하여  
> 객체 검출을 **진정한 End-to-End 학습 문제**로 바꿔놓았다.

---

### 📚 Reference
- Carion et al., “End-to-End Object Detection with Transformers,” ECCV 2020.
