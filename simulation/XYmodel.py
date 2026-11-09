"""
2D XY Model - Numba 가속 Monte Carlo Metropolis (L=64 스케일업 버전)
=====================================================================

이전 프로토타입(xy_model_mc.py, L=10, 순수 Python)과 물리/알고리즘은
완전히 동일합니다. 차이는 성능뿐입니다:

    - metropolis_sweep의 핵심 루프를 @njit으로 JIT 컴파일
    - vortex(winding number) 계산도 @njit으로 가속
    - L=10 -> L=64 (스핀 개수 100개 -> 4096개)로 확장해도 실용적인 속도 유지

주의: Numba의 @njit 함수 안에서는 np.random.default_rng()가 아니라
      np.random.random(), np.random.randint() 같은 numba 자체 RNG를 씁니다.
      (numpy의 Generator 객체는 nopython 모드에서 지원 안 됨)
"""

import time
import numpy as np
from numba import njit


# ---------------------------------------------------------------
# Numba 가속 핵심 함수들
# ---------------------------------------------------------------

@njit(cache=True)
def _local_energy_delta(theta, L, i, j, old_theta, new_theta, J):
    up = theta[(i - 1) % L, j]
    down = theta[(i + 1) % L, j]
    left = theta[i, (j - 1) % L]
    right = theta[i, (j + 1) % L]

    sum_cos = np.cos(up) + np.cos(down) + np.cos(left) + np.cos(right)
    sum_sin = np.sin(up) + np.sin(down) + np.sin(left) + np.sin(right)

    E_old = -J * (np.cos(old_theta) * sum_cos + np.sin(old_theta) * sum_sin)
    E_new = -J * (np.cos(new_theta) * sum_cos + np.sin(new_theta) * sum_sin)
    return E_new - E_old


@njit(cache=True)
def _metropolis_sweep_numba(theta, T, J, delta_max):
    """L*L번의 단일 스핀 업데이트 시도 (1 sweep). theta 배열을 in-place로 수정."""
    L = theta.shape[0]
    N = L * L
    accepted = 0

    for _ in range(N):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        old_theta = theta[i, j]
        new_theta = (old_theta + (np.random.random() * 2.0 - 1.0) * delta_max) % (2.0 * np.pi)

        dE = _local_energy_delta(theta, L, i, j, old_theta, new_theta, J)

        if dE <= 0.0 or np.random.random() < np.exp(-dE / T):
            theta[i, j] = new_theta
            accepted += 1

    return accepted / N


@njit(cache=True)
def _total_energy_numba(theta, J):
    L = theta.shape[0]
    E = 0.0
    for i in range(L):
        for j in range(L):
            th = theta[i, j]
            right = theta[i, (j + 1) % L]
            down = theta[(i + 1) % L, j]
            E += -J * (np.cos(th - right) + np.cos(th - down))
    return E


@njit(cache=True)
def _wrap_angle(a):
    while a > np.pi:
        a -= 2.0 * np.pi
    while a < -np.pi:
        a += 2.0 * np.pi
    return a


@njit(cache=True)
def _count_vortices_numba(theta):
    """plaquette winding number 기반 vortex/antivortex 개수 세기."""
    L = theta.shape[0]
    n_vortex = 0
    n_antivortex = 0
    for i in range(L):
        for j in range(L):
            t00 = theta[i, j]
            t01 = theta[i, (j + 1) % L]
            t11 = theta[(i + 1) % L, (j + 1) % L]
            t10 = theta[(i + 1) % L, j]

            d1 = _wrap_angle(t01 - t00)
            d2 = _wrap_angle(t11 - t01)
            d3 = _wrap_angle(t10 - t11)
            d4 = _wrap_angle(t00 - t10)

            w = round((d1 + d2 + d3 + d4) / (2.0 * np.pi))
            if w > 0:
                n_vortex += 1
            elif w < 0:
                n_antivortex += 1
    return n_vortex, n_antivortex


# ---------------------------------------------------------------
# 사용자 인터페이스 클래스 (이전 XYModel2D와 동일한 API 유지)
# ---------------------------------------------------------------

class XYModel2DFast:
    def __init__(self, L=64, J=1.0, seed=None):
        self.L = L
        self.N = L * L
        self.J = J
        if seed is not None:
            np.random.seed(seed)  # numba의 njit 함수들이 참조하는 전역 RNG 시드
        self.theta = np.random.uniform(0, 2 * np.pi, size=(L, L))

    def metropolis_sweep(self, T, delta_max=None):
        if delta_max is None:
            delta_max = max(0.4, min(2.5, T * 1.8))  # 온도에 비례한 자동 조절
        return _metropolis_sweep_numba(self.theta, T, self.J, delta_max)

    def run(self, T, n_sweeps, delta_max=None, verbose=False):
        rates = np.empty(n_sweeps)
        for s in range(n_sweeps):
            rates[s] = self.metropolis_sweep(T, delta_max=delta_max)
            if verbose and (s + 1) % max(1, n_sweeps // 10) == 0:
                print(f"  sweep {s+1}/{n_sweeps}  accept_rate={rates[s]:.3f}")
        return rates

    def total_energy(self):
        return _total_energy_numba(self.theta, self.J)

    def magnetization(self):
        mx = np.mean(np.cos(self.theta))
        my = np.mean(np.sin(self.theta))
        return np.sqrt(mx**2 + my**2)

    def count_vortices(self):
        """(vortex 개수, antivortex 개수) 반환. 정상적인 경우 항상 거의 같음(쌍 생성)."""
        return _count_vortices_numba(self.theta)

    def spin_config_cos_sin(self):
        """CNN 입력용 (2, L, L) 형태의 (cos, sin) 듀얼 채널 배열."""
        return np.stack([np.cos(self.theta), np.sin(self.theta)], axis=0)


# ---------------------------------------------------------------
# 데이터셋 생성 (개선 1: annealing / 개선 2: burn-in + decorrelation)
# ---------------------------------------------------------------

def build_temperature_grid():
    """T_BKT(~0.893) 근방에 촘촘한 비균일 온도 그리드 (고온 -> 저온 순)."""
    low = np.linspace(0.30, 0.70, 5)
    mid = np.linspace(0.75, 1.05, 13)   # T_BKT 근방 -- 촘촘하게
    high = np.linspace(1.10, 1.80, 8)
    grid = np.concatenate([low, mid, high])
    grid = sorted(set(round(float(t), 3) for t in grid), reverse=True)
    return grid


def generate_dataset(L=64, J=1.0, temperatures=None,
                      n_burnin=1000, n_samples_per_T=20, sweeps_between=50,
                      seed=None, verbose=True):
    """
    여러 온도에서 CNN 학습용 스핀 배치 데이터셋을 생성.

    [개선 1] Simulated annealing 방식
    ----------------------------------
    온도마다 XYModel2DFast를 새로 만들지 않고, 모델 하나를 계속 재사용합니다.
    즉 이전 온도에서 평형에 도달한 스핀 배치를 다음 온도의 "초기 상태"로
    그대로 이어받습니다. T_BKT 근방은 critical slowing down 때문에 무작위
    상태에서 평형까지 도달하는 데 훨씬 오래 걸리는데, 이전 온도의 평형 상태에서
    출발하면 그보다 훨씬 적은 sweep으로도 평형에 도달합니다.

    temperatures는 높은 온도 -> 낮은 온도 순으로 정렬해서 넘기는 걸 권장합니다
    (뜨겁게 무질서화된 상태에서 시작해서 서서히 식히는 물리적으로 자연스러운 방향).

    [개선 2] Burn-in과 decorrelation 분리
    ----------------------------------------
    각 온도에서:
      1) n_burnin sweep 동안은 그냥 진행만 하고 저장하지 않음
         (이 온도의 진짜 평형 상태에 도달할 시간을 줌)
      2) 그 다음부터 n_samples_per_T개를 뽑되, 매번 sweeps_between sweep씩
         추가로 진행한 뒤에 저장 (연속 sweep 간의 강한 상관관계를 줄여서
         서로 통계적으로 "다른" 샘플이 되도록 함)

    Parameters
    ----------
    temperatures : list[float]
        샘플링할 온도 목록. 높은 온도 -> 낮은 온도 순 권장.
    n_burnin : int
        온도가 바뀐 뒤 평형 도달을 위해 버리는 sweep 수.
    n_samples_per_T : int
        온도 하나당 뽑을 샘플 개수.
    sweeps_between : int
        같은 온도에서 샘플 사이에 진행할 sweep 수 (decorrelation).

    Returns
    -------
    list[dict]
        각 원소는 {"T", "spin_config", "energy", "magnetization",
                   "n_vortex", "n_antivortex"} 를 담은 딕셔너리.
        spin_config는 CNN 입력용 (2, L, L) 배열 ((cos, sin) 채널).
    """
    if temperatures is None:
        raise ValueError("temperatures 리스트를 지정해야 합니다")

    model = XYModel2DFast(L=L, J=J, seed=seed)
    dataset = []

    for T in temperatures:
        # --- burn-in: 이 온도의 평형 상태에 도달할 때까지 버림 ---
        model.run(T, n_burnin)

        # --- decorrelated 샘플 n_samples_per_T개 수집 ---
        for _ in range(n_samples_per_T):
            model.run(T, sweeps_between)

            n_vortex, n_antivortex = model.count_vortices()
            dataset.append({
                "T": T,
                "spin_config": model.spin_config_cos_sin(),
                "energy": model.total_energy(),
                "magnetization": model.magnetization(),
                "n_vortex": n_vortex,
                "n_antivortex": n_antivortex,
            })

        if verbose:
            last = dataset[-1]
            print(f"  T={T:.3f}  |M|={last['magnetization']:.3f}  "
                  f"vortex={last['n_vortex']}  (샘플 {n_samples_per_T}개 수집)")

    return dataset


if __name__ == "__main__":
    L = 64
    T = 1.0
    J = 1.0

    print(f"=== L={L} ({L*L}개 스핀), Numba 가속 버전 ===\n")

    # 1) JIT 컴파일 워밍업 (첫 호출은 컴파일 시간 포함되므로 별도로 시간 측정)
    warmup = XYModel2DFast(L=8, J=J, seed=0)
    warmup.metropolis_sweep(T)
    warmup.count_vortices()
    warmup.total_energy()

    # 2) 본 실행: L=64에서 컴파일된 함수로 순수 계산 시간만 측정
    model = XYModel2DFast(L=L, J=J, seed=42)
    print(f"초기 에너지: {model.total_energy():.2f}, 초기 |M|: {model.magnetization():.3f}")

    n_sweeps = 2000
    t0 = time.perf_counter()
    model.run(T=T, n_sweeps=n_sweeps, verbose=True)
    elapsed = time.perf_counter() - t0

    n_vortex, n_antivortex = model.count_vortices()
    print(f"\n{n_sweeps} sweep 완료 (소요시간 {elapsed:.2f}초, "
          f"sweep당 {elapsed/n_sweeps*1000:.2f}ms)")
    print(f"평형화 후 에너지: {model.total_energy():.2f}, |M|: {model.magnetization():.3f}")
    print(f"vortex: {n_vortex}, antivortex: {n_antivortex}")
    print(f"CNN 입력 형태: {model.spin_config_cos_sin().shape}")

    # ------------------------------------------------------------
    # 데이터셋 생성 데모 (실제 15,000개 규모가 아니라 동작 확인용 소규모 예시)
    # 고온 -> 저온 순으로 annealing, 각 온도마다 burn-in 후 decorrelated 샘플링
    # ------------------------------------------------------------
    print("\n=== 데이터셋 생성 데모 (annealing + burn-in/decorrelation) ===")
    demo_temperatures = [1.60, 1.20, 1.00, 0.893, 0.70, 0.40]  # 고온 -> 저온, T_BKT 포함

    t0 = time.perf_counter()
    dataset = generate_dataset(
        L=L, J=J,
        temperatures=demo_temperatures,
        n_burnin=300,        # 데모라 실제 생성보다 적게 잡음 (실전에서는 더 크게)
        n_samples_per_T=5,
        sweeps_between=30,
        seed=123,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n총 {len(dataset)}개 샘플 생성, 소요시간 {elapsed:.2f}초")
    print(f"샘플 하나 예시: T={dataset[0]['T']}, "
          f"spin_config shape={dataset[0]['spin_config'].shape}, "
          f"|M|={dataset[0]['magnetization']:.3f}")
