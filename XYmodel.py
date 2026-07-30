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
