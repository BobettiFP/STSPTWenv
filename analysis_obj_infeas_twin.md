# obj_infeas_*_twin Figure 및 Raw Data 종합 분석

세 figure (obj_infeas_easy_twin.png, obj_infeas_medium_twin.png, obj_infeas_hard_twin.png)와 raw data (test_stsptw_matched.csv, test_tsptw_on_stsptw_dw_sweep.csv)로부터 도출할 수 있는 결론을 정리한다.

**평가 기준: Infeas(ins_infeasible_pct)와 aug_score 둘 다 낮을수록 좋음.**

---

## 1. 실험 구조 요약

| 구분 | 내용 |
|------|------|
| 모델 | POMO, POMO*, POMO*+PIP |
| 문제 유형 | STSPTW(본래 문제), TSPTW(STSPTW 인스턴스에 적용) |
| 데이터 | test_stsptw_matched.csv (delay_weight 0.1~1.0), test_tsptw_on_stsptw_dw_sweep.csv (delay_scale 0.01~1.0) |
| 난이도 | easy, medium, hard |

---

## 2. Hardness별 주요 결과

### 2.1 Easy

| 모델 | STSPTW Infeas | TSPTW Infeas | STSPTW aug_score | TSPTW aug_score |
|------|---------------|--------------|------------------|-----------------|
| POMO | 96.7~97.6% | 95.8~97.3% | **2.27~2.31** | **2.49~2.51** |
| POMO* | 37.4~40.7% | 40.2~65.1% | 2.92~2.98 | 2.98~2.99 |
| POMO*+PIP | 35.2~37.3% | 38.7~51.0% | 2.93~2.98 | 3.00~3.02 |

**요약:**
- POMO: aug_score는 가장 낮지만 Infeas가 95% 이상으로 매우 높음.
- POMO*+PIP: Infeas가 가장 낮음 (STSPTW 35~37%, TSPTW 39~51%).
- POMO*: Infeas는 POMO*+PIP보다 약간 높고, aug_score는 비슷.
- delay_weight 증가 시 TSPTW에서 Infeas가 40% → 65%로 증가.

---

### 2.2 Medium

| 모델 | STSPTW Infeas | TSPTW Infeas | STSPTW aug_score | TSPTW aug_score |
|------|---------------|--------------|------------------|-----------------|
| POMO | parse_failed | **100%** | - | nan |
| POMO* | 83.7~92.5% | 83.0~91.6% | **3.62~3.83** | 3.70~3.85 |
| POMO*+PIP | 83.7~91.4% | 82.8~92.4% | 3.70~3.90 | 3.72~3.90 |

**요약:**
- POMO: STSPTW는 parse_failed, TSPTW는 100% Infeas로 실질적 사용 불가.
- POMO*: aug_score가 가장 낮음 (3.62~3.83), Infeas는 83~92%.
- POMO*+PIP: aug_score는 POMO*보다 약간 높고, Infeas는 비슷.
- delay_weight 증가 시 aug_score는 감소(개선), Infeas는 증가하는 경향.

---

### 2.3 Hard

| 모델 | STSPTW Infeas | TSPTW Infeas | STSPTW aug_score | TSPTW aug_score |
|------|---------------|--------------|------------------|-----------------|
| POMO | parse_failed | **100%** | - | nan |
| POMO* | 79.7~99.5% | 73.5~94.7% | **4.60~4.91** | 4.95~5.11 |
| POMO*+PIP | 78.0~96.5% | 74.5~95.4% | 4.97~5.14 | 4.96~5.09 |

**요약:**
- POMO: medium과 동일하게 사용 불가.
- POMO* (STSPTW): aug_score가 가장 낮음 (4.60~4.91), 특히 delay_weight 0.3~0.4에서 약 4.60.
- POMO*+PIP: Infeas는 POMO*보다 약간 낮지만 aug_score는 더 높음.
- POMO* (TSPTW): Infeas가 가장 낮음 (73.5~94.7%), aug_score는 4.95~5.11로 상대적으로 높음.

---

## 3. 도출 가능한 결론

### 3.1 모델 비교

1. **POMO**: easy에서만 aug_score가 가장 낮지만, Infeas가 95% 이상이라 실용성이 낮음. medium/hard에서는 parse_failed 또는 100% Infeas로 사용 불가.

2. **POMO***: aug_score 측면에서 가장 우수. medium/hard에서 POMO*+PIP보다 낮은 aug_score를 보임. 대신 Infeas는 POMO*+PIP보다 높은 경우가 많음.

3. **POMO*+PIP**: Infeas 측면에서 우수. easy에서 가장 낮은 Infeas, hard에서도 POMO*보다 낮은 Infeas. 대신 aug_score는 POMO*보다 높음.

### 3.2 STSPTW vs TSPTW

- **STSPTW**: 본래 문제에 맞게 학습된 모델이므로 aug_score가 더 낮은 경향.
- **TSPTW**: STSPTW 인스턴스에 TSPTW 모델을 적용한 경우로, Infeas가 더 높고 aug_score도 더 높은 경향.
- delay_weight가 커질수록 TSPTW의 Infeas가 크게 증가 (easy: 40%→65%, medium/hard에서도 유사한 패턴).

### 3.3 delay_weight 영향

- **Infeas**: delay_weight 증가 시 대체로 Infeas 증가.
- **aug_score**: STSPTW에서는 delay_weight 증가 시 감소(개선)하는 경향 (medium, hard). easy에서는 큰 변화 없음.

### 3.4 Trade-off

- **Infeas vs aug_score**: 두 지표 간 trade-off가 존재.
  - POMO*: aug_score 우선 → Infeas 상대적으로 높음.
  - POMO*+PIP: Infeas 우선 → aug_score 상대적으로 높음.
- **난이도 증가**: easy → medium → hard로 갈수록 Infeas가 전반적으로 증가하고, aug_score도 증가.

### 3.5 권장 설정 (목적별)

| 목적 | 권장 모델 | 권장 설정 |
|------|-----------|-----------|
| aug_score 최소화 | POMO* (STSPTW) | medium/hard: delay_weight 0.8~1.0, hard: 0.3~0.4 |
| Infeas 최소화 | POMO*+PIP (STSPTW) | easy: delay_weight 0.2~0.5 |
| 균형 | POMO*+PIP | medium/hard에서 Infeas·aug_score 절충 |

---

## 4. aug_score가 delay_weight 증가 시 감소하는 이유 (STSPTW)

STSPTW에서 delay_weight(delay_scale)가 커질 때 aug_score가 줄어드는 현상은 주로 다음 두 가지 요인으로 설명할 수 있다.

### 4.1 생존자 편향 (Survivorship Bias)

aug_score는 **적어도 하나의 feasible solution이 있는 인스턴스**에 대해서만 평균을 낸다.

- **delay_weight가 낮을 때**: 대부분의 인스턴스가 feasible. aug_score는 전체 인스턴스(짧은/긴 투어 모두 포함)에 대한 평균.
- **delay_weight가 높을 때**: stochastic delay가 커져서 time window 위반이 많아짐. **긴 투어일수록** 여러 구간에서 delay가 누적되어 feasible하지 않을 가능성이 커짐.
- 결과적으로, delay_weight가 높을 때 feasible로 남는 인스턴스는 **상대적으로 짧은 투어**를 가진 경우가 많음.
- aug_score = feasible 인스턴스들의 평균 tour length → 이 subset이 짧은 투어 위주로 구성되므로 aug_score가 감소.

### 4.2 학습 효과 (Training Effect)

`test_stsptw_matched.csv`는 **같은 delay_weight로 학습된 모델**을 같은 delay_weight로 평가한다.

- delay_weight가 큰 환경에서 학습한 모델은, time window 위반을 줄이기 위해 **짧은 투어**를 선호하도록 학습됨.
- delay가 큰 환경에서는 긴 투어가 위험하므로, 모델이 더 보수적인(짧은) 경로를 선택하는 쪽으로 수렴.
- 따라서 delay_weight가 클수록, 그 환경에 맞게 학습된 모델이 더 짧은 투어를 생성 → aug_score 감소.

### 4.3 요약

| 요인 | 설명 |
|------|------|
| 생존자 편향 | aug_score가 feasible 인스턴스만 평균하므로, delay_weight↑ → 긴 투어 인스턴스가 infeasible로 빠짐 → 남은 인스턴스는 짧은 투어 위주 → aug_score↓ |
| 학습 효과 | delay_weight↑ 환경에서 학습한 모델은 짧은 투어를 선호 → aug_score↓ |

두 요인이 함께 작용해, STSPTW에서 delay_weight가 증가할 때 aug_score가 감소하는 경향이 나타난다.

---

## 5. 데이터 출처

- `test_stsptw_matched.csv`: STSPTW matched 실험 (delay_weight 0.1~1.0, 10구간) — 각 모델을 학습 시 사용한 delay_weight와 동일한 값으로 평가
- `test_tsptw_on_stsptw_dw_sweep.csv`: TSPTW on STSPTW sweep (delay_scale 0.01~1.0, 100구간)
