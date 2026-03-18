## Data Generation: Methodology vs. This Repo

이 문서는 **Benchmarking a DRL Extension of PIP for TSPTW with Stochastic Travel Times**(이하 Methodology 문서)에 나온 데이터 생성 방식과, 이 repo(STSPTWenv)의 실제 구현을 짧게 비교합니다.  
의도적으로 **수식은 모두 텍스트/코드 스타일**로만 표기해서, 렌더링 이슈가 생기지 않도록 했습니다.

---

## 1. 큰 그림 요약

- **Layer 1 (deterministic TSPTW 인스턴스 생성)**  
  - Methodology: PIP(Bi et al., NeurIPS 2024)의 데이터 생성기를 backbone 으로 사용.  
  - 이 repo: `POMO+PIP/envs/TSPTWEnv.py` 가 **공식 PIP-constraint repo와 동일한 코드**를 사용하므로, Layer 1 은 사실상 PIP와 동일.

- **Layer 2 (stochastic overlay)**  
  - Methodology: Gamma 분포 및 Two-point 분포로 mean-preserving 노이즈를 추가하고, CV 를 명시적으로 스윕.  
  - 이 repo: 시간·거리 의존적인 lognormal-like delay 를 추가하고, `delay_scale` 로 강도만 조절.

- **Feasibility certification**  
  - Methodology: Dumas DP + Monte Carlo 로 Regime A/B/C 를 분리하고, 평가는 Regime A 에서만 함.  
  - 이 repo: Regime 구분이나 사전 인증 절차 없이, 생성된 인스턴스를 그대로 학습/평가에 사용.

---

## 2. Layer 1: Instance Backbone (TSPTW)

### 2.1 노드 좌표

- Methodology:  
  - n ∈ {50, 100} 개의 노드를 `[0, 1]^2` 에서 균등 분포로 샘플.  
  - 좌표:  
    - x_i, y_i ~ Uniform(0, 1)

- 이 repo (`TSPTWEnv.get_random_problems` / `generate_tsptw_data`):  
  - 내부적으로는 `[0, coord_factor]^2` (기본 100) 에서 균등 샘플 후, 정규화 시 `[0, 1]^2` 범위를 사용.  
  - 좌표 생성 예시:  
    - node_xy = torch.rand(batch, problem_size, 2) * coord_factor

→ 좌표 스케일링 방식이 다를 뿐, **정규화 이후 분포는 사실상 동일**합니다.

### 2.2 평균 이동 시간

- Methodology:  
  - 평균 이동 시간  
    - mu_ij = || x_i - x_j ||_2  (유클리드 거리)

- 이 repo:  
  - travel_time = torch.cdist(node_xy, node_xy, p=2) / speed  
  - 기본 speed = 1.0 이므로, travel_time 이 곧 mu_ij 와 동일.

→ 평균 이동 시간 정의는 **완전히 동일**합니다.

### 2.3 Time Window 생성 (hard / easy / medium)

#### Methodology 문서(개념)

- “tightness parameter” 를 이용해 time window 의 폭과 위치를 조절하고,  
  hardness ∈ {easy, medium, hard} 에 따라 제약 강도를 다르게 설정한다고만 설명.  
- 구체적인 수식/코드는 Methodology 문서에 자세히 적혀 있지는 않음(대신 PIP generator 를 쓰라고 명시).

#### 이 repo (PIP-constraint와 동일한 코드)

`POMO+PIP/envs/TSPTWEnv.py` 의 `get_random_problems` 기준:

- **hard**  
  - Da Silva & Urrutia(2010) / Cappart hybrid-cp-rl-solver 의 방식 사용.  
  - 랜덤 feasible tour 를 하나 만들고, 그 위를 따라가며 누적 거리에 기반해 TW 하한/상한을 샘플:  
    - rand_tw_lb ~ Uniform(total_dist - max_tw_size/2, total_dist)  
    - rand_tw_ub ~ Uniform(total_dist, total_dist + max_tw_size/2)

- **easy / medium**  
  - JAMPR(Falkner) 스타일의 time window 생성 (`generate_tsptw_data`, `gen_tw`).  
  - tw_start 는 전체 horizon 의 앞 절반에서 랜덤,  
  - duration 은 dura_region 에 따라 샘플:  
    - easy: dura_region = [0.5, 0.75] (넓은 TW)  
    - medium: dura_region = [0.1, 0.2] (더 좁은 TW)  
  - tw_end = tw_start + duration, horizon 을 넘지 않도록 자름.

#### 정리

- 이 repo 의 TSPTW 데이터 생성 코드는 **공식 PIP-constraint 의 TSPTWEnv 와 동일**합니다.  
- 따라서 Methodology 문서가 “PIP 의 data generator 를 backbone 으로 사용하라”고 한 Layer 1 요구사항은, 이 repo 에서 이미 만족되고 있습니다.

---

## 3. Layer 2: Stochastic Overlay

여기부터가 Methodology 문서와 이 repo 의 **본질적인 차이**입니다.

### 3.1 Methodology 문서의 설계

Methodology 문서는 다음 두 가지 분포를 사용해,  
“평균은 유지하면서 분산(CV)을 조절하는” stochastic overlay 를 정의합니다.

- **Primary: Gamma 분포 (Taş et al., 2014)**  
  - 정의:  
    - tilde{t}_{ij} ~ Gamma(k, mu_{ij} / k)  
    - E[tilde{t}_{ij}] = mu_{ij}  
    - CV = 1 / sqrt(k)
  - k ∈ {1, 4, 16, ∞} 를 스윕해서 CV ∈ {1.0, 0.5, 0.25, 0} 를 얻음.  
  - k → ∞ 일 때 deterministic 케이스가 되어, PIP 와 동일한 실험이 됨.

- **Secondary: Two-point(혼합) 분포 (Zhang et al.)**  
  - 정의(개념):  
    - tilde{t}_{ij} = mu_{ij} (1 - delta)  with prob. p  
    - tilde{t}_{ij} = mu_{ij} (1 + epsilon) with prob. 1 - p  
  - delta, epsilon 을 적절히 잡아서 E[tilde{t}_{ij}] = mu_{ij} 를 유지(평균 보존).

핵심 포인트:

- underlying deterministic 구조(mu_ij)는 그대로 두고,  
  Gamma / Two-point 로 분산만 조절해 **CV 를 실험 변수**로 사용한다는 점.

### 3.2 이 repo(STSPTWEnv)의 설계

`POMO+PIP/envs/STSPTWEnv.py` 기준:

- deterministic travel time:  
  - mu_ij = || x_i - x_j ||_2 / speed (speed = 1)

- stochastic overlay:  
  - tilde{t}_{ij} = mu_{ij} + delay  
  - delay = delay_scale * base_delay(current_time_norm, distance) * random_factor(current_time_norm)

여기서:

- base_delay 는  
  - 시간대(아침/저녁 rush hour)를 반영하는 time_factor(current_time_norm)  
  - 거리에 따른 포화형 factor (1 - exp(-distance / 0.5))  
  를 곱해서 만듭니다.

- random_factor 는  
  - eps ~ Normal(0, 1) 에 대해 exp(mu + sigma * eps) 형태의 lognormal-like 노이즈.

정리하면:

- Methodology:  
  - tilde{t}_{ij} 를 Gamma / Two-point 로 직접 샘플  
  - CV 를 k 로 명확하게 제어  
  - 시간/거리 의존성은 별도로 넣지 않음(arc 별 독립).

- 이 repo:  
  - tilde{t}_{ij} = mu_{ij} + (시간/거리 의존 delay)  
  - 분산의 세기는 delay_scale 로만 조절  
  - 아침/저녁 피크, 거리 증가에 따른 혼잡 증가 등 **시계열적/공간적 구조**를 반영.

즉, 이 repo 의 STSPTWEnv 는 “정확히 같은 분포”는 아니고,  
**보다 현실적인 traffic-like noise**를 흉내 내는 쪽으로 설계되어 있습니다.

---

## 4. Feasibility Certification과 Regime

### 4.1 Methodology 문서

Methodology 문서는 data 를 세 가지 regime 으로 나눕니다.

- Regime A  
  - deterministic 문제(TSPTW)는 feasible  
  - stochastic 문제(TSPTW-S)에서도, 어떤 정책 pi 가 chance constraint 를 만족하는 feasible 해를 낼 수 있음.

- Regime B  
  - deterministic 문제는 feasible  
  - stochastic noise 때문에 사실상 chance constraint 를 만족하는 해가 없음.

- Regime C  
  - deterministic 문제부터 infeasible.

이걸 위해:

- deterministic feasibility:  
  - Dumas et al.(1995) DP 로 mean-instance 를 풀어서 feasible 여부 판정.

- stochastic feasibility:  
  - “가장 좋은 deterministic tour(pi*)” 를 Monte Carlo 로 여러 번 시뮬레이션 해서,  
    - hat{F}(pi*) = P(모든 노드에서 time window 안에 도착)  
    - hat{F}(pi*) >= alpha 이면 Regime A, 그렇지 않으면 Regime B.

**평가 시에는 Regime A 인스턴스만 사용**해서,  
“instance 가 애초에 너무 빡세서 어쩔 수 없이 infeasible인 경우(Regime B/C)”를 메트릭에서 분리합니다.

### 4.2 이 repo

이 repo 에는:

- Dumas DP 구현 없음  
- Regime A/B/C 분류 없음  
- stochastic feasibility 를 사전 인증하는 Monte Carlo 절차 없음

즉, “feasible instance 만 골라서 평가한다”는 Methodology 의 엄밀한 설정은 만족하지 않고,  
