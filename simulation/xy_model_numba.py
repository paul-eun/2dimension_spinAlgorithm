"""
2D XY Model - Numba 가속 Wolff Cluster Algorithm (L=64 스케일업 버전)
=====================================================================

기존 Metropolis 버전(스핀을 1개씩 흔들어보는 방식)을 Wolff 클러스터
알고리즘으로 교체한 버전입니다.

    - 반사축(n_hat)을 무작위로 정한 뒤, 씨앗 스핀에서부터 확률적으로
      클러스터를 키우고, 클러스터 전체를 한 번에 거울 반사시킵니다.
    - 이 결합 확률(p_add)식이 클러스터를 뒤집어도 볼츠만 분포를
      정확히 따른다는 것을 수학적으로 보장합니다.
    - Metropolis 대비 상관관계가 낮은 샘플을 훨씬 적은 sweep으로 얻을
      수 있어 burn-in/decorrelation에 필요한 계산량이 크게 줄어듭니다.

주의: Numba의 @njit 함수 안에서는 np.random.default_rng()가 아니라
      np.random.random(), np.random.randint() 같은 numba 자체 RNG를 씁니다.
      (numpy의 Generator 객체는 nopython 모드에서 지원 안 됨)

(AI의 도움을 받았다)
"""

import time
import numpy as np
from numba import njit


# ---------------------------------------------------------------
# Numba 가속 Wolff 알고리즘 핵심 로직
# ---------------------------------------------------------------

@njit(cache=True)
def _wolff_step_numba(theta, T, J):
    """2D XY 모델 1회 Wolff Cluster Update (in-place modification)"""
    L = theta.shape[0]

    # 1. 무작위 반사 축(Random Projection Vector) n_hat 선택
    phi_n = np.random.random() * 2.0 * np.pi
    nx = np.cos(phi_n)
    ny = np.sin(phi_n)

    # 2. 씨앗(Seed) 스핀 무작위 선택
    i0 = np.random.randint(0, L)
    j0 = np.random.randint(0, L)

    visited = np.zeros((L, L), dtype=np.bool_)
    stack_i = np.empty(L * L, dtype=np.int32)
    stack_j = np.empty(L * L, dtype=np.int32)

    stack_i[0] = i0
    stack_j[0] = j0
    visited[i0, j0] = True
    stack_ptr = 1
    cluster_size = 0

    di = np.array([-1, 1, 0, 0], dtype=np.int32)
    dj = np.array([0, 0, -1, 1], dtype=np.int32)

    # 3. 클러스터 성장 (BFS/DFS Stack)
    while stack_ptr > 0:
        stack_ptr -= 1
        curr_i = stack_i[stack_ptr]
        curr_j = stack_j[stack_ptr]
        cluster_size += 1

        curr_angle = theta[curr_i, curr_j]
        S_dot_n = np.cos(curr_angle) * nx + np.sin(curr_angle) * ny

        for k in range(4):
            ni = (curr_i + di[k]) % L
            nj = (curr_j + dj[k]) % L

            if not visited[ni, nj]:
                neighbor_angle = theta[ni, nj]
                S_nbr_dot_n = np.cos(neighbor_angle) * nx + np.sin(neighbor_angle) * ny

                # XY 모델 Wolff Bond 결합 확률
                dot_product = S_dot_n * S_nbr_dot_n
                if dot_product > 0:
                    p_add = 1.0 - np.exp(-2.0 * J * dot_product / T)
                    if np.random.random() < p_add:
                        visited[ni, nj] = True
                        stack_i[stack_ptr] = ni
                        stack_j[stack_ptr] = nj
                        stack_ptr += 1

    # 4. 클러스터에 속한 모든 스핀 n_hat 축 기준 반사
    for i in range(L):
        for j in range(L):
            if visited[i, j]:
                theta[i, j] = (2.0 * phi_n - theta[i, j] + np.pi) % (2.0 * np.pi)

    return cluster_size


@njit(cache=True)
def _wolff_sweep_numba(theta, T, J):
    """누적 뒤집힌 스핀 수 >= L*L 이 될 때까지 Wolff step을 수행하는 1 Sweep 정의"""
    L = theta.shape[0]
    N = L * L
    flipped_spins = 0
    steps = 0

    while flipped_spins < N:
        c_size = _wolff_step_numba(theta, T, J)
        flipped_spins += c_size
        steps += 1

    return steps  # 1 sweep에 소요된 cluster step 수 반환


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
# 사용자 인터페이스 클래스 (Wolff 적용 버전)
# ---------------------------------------------------------------

@njit(cache=True)
def _seed_numba(seed):
    # numba RNG는 NumPy RNG와 별개라서, 반드시 njit 함수 안에서 시드를 줘야 함
    np.random.seed(seed)


class XYModel2DFast:
    def __init__(self, L=64, J=1.0, seed=None):
        self.L = L
        self.N = L * L
        self.J = J
        if seed is not None:
            np.random.seed(seed)   # 초기 배치(아래 uniform)용 NumPy RNG
            _seed_numba(seed)      # Wolff 업데이트용 numba RNG
        self.theta = np.random.uniform(0, 2 * np.pi, size=(L, L))

    def wolff_sweep(self, T):
        """Metropolis sweep 대신 Wolff sweep 수행"""
        return _wolff_sweep_numba(self.theta, T, self.J)

    def run(self, T, n_sweeps, verbose=False):
        steps_list = np.empty(n_sweeps)
        for s in range(n_sweeps):
            steps_list[s] = self.wolff_sweep(T)
            if verbose and (s + 1) % max(1, n_sweeps // 10) == 0:
                print(f"  sweep {s+1}/{n_sweeps} 완료 (스텝 수: {steps_list[s]:.0f})")
        return steps_list

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
# 데이터셋 생성 (Wolff 알고리즘에 맞게 간소화된 burn-in/decorrelation)
# ---------------------------------------------------------------

def build_temperature_grid():
    """0.30~1.80 구간을 0.025 간격으로 촘촘한 균일 온도 그리드 (고온 -> 저온 순)."""
    n_points = round((1.80 - 0.30) / 0.025) + 1  # 61개
    grid = np.linspace(0.30, 1.80, n_points)
    grid = sorted(set(round(float(t), 3) for t in grid), reverse=True)
    return grid


def generate_dataset(L=64, J=1.0, temperatures=None,
                      n_burnin=50, n_samples_per_T=20, sweeps_between=5,
                      seed=None, verbose=True):
    """
    Wolff 알고리즘을 사용한 고품질 데이터셋 생성.

    Wolff 특성상 평형 도달과 decorrelation이 압도적으로 빨라, Metropolis
    버전에서 쓰던 n_burnin(1000)/sweeps_between(40~50) 값을 대폭 낮추어도
    (기본값 50/5) 충분히 독립적인 샘플을 얻을 수 있습니다.

    temperatures는 높은 온도 -> 낮은 온도 순으로 정렬해서 넘기는 걸 권장합니다
    (뜨겁게 무질서화된 상태에서 시작해서 서서히 식히는 물리적으로 자연스러운 방향).

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
        # --- burn-in: Wolff 알고리즘은 평형 도달이 압도적으로 빨라 50 sweep이면 충분함 ---
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
                  f"vortex={last['n_vortex']}  (샘플 {n_samples_per_T}개 수집 완료)")

    return dataset


if __name__ == "__main__":
    L = 64
    T = 1.0
    J = 1.0

    print(f"=== L={L} ({L*L}개 스핀), Numba Wolff 가속 버전 ===\n")

    # 1) JIT 컴파일 워밍업 (첫 호출은 컴파일 시간 포함되므로 별도로 시간 측정)
    warmup = XYModel2DFast(L=8, J=J, seed=0)
    warmup.wolff_sweep(T)
    warmup.count_vortices()
    warmup.total_energy()

    # 2) 본 실행: L=64에서 컴파일된 함수로 순수 계산 시간만 측정
    model = XYModel2DFast(L=L, J=J, seed=42)
    print(f"초기 에너지: {model.total_energy():.2f}, 초기 |M|: {model.magnetization():.3f}")

    n_sweeps = 500  # Wolff 알고리즘은 500 sweep으로도 평형화에 충분함
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
    # 데이터셋 생성 데모 (실제 규모가 아니라 동작 확인용 소규모 예시)
    # 고온 -> 저온 순으로 annealing, 각 온도마다 burn-in 후 decorrelated 샘플링
    # ------------------------------------------------------------
    print("\n=== Wolff 기반 데이터셋 생성 데모 ===")
    demo_temperatures = [1.60, 1.20, 1.00, 0.893, 0.70, 0.40]  # 고온 -> 저온, T_BKT 포함

    t0 = time.perf_counter()
    dataset = generate_dataset(
        L=L, J=J,
        temperatures=demo_temperatures,
        n_burnin=50,         # Wolff 알고리즘이므로 50으로도 대폭 감소 가능
        n_samples_per_T=5,
        sweeps_between=5,    # 단 5 sweep만으로도 충분히 독립적인 샘플 생성
        seed=123,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n총 {len(dataset)}개 샘플 생성, 소요시간 {elapsed:.2f}초")
    print(f"샘플 하나 예시: T={dataset[0]['T']}, "
          f"spin_config shape={dataset[0]['spin_config'].shape}, "
          f"|M|={dataset[0]['magnetization']:.3f}")
