import numpy as np
from nn_magnetics.data.create_data import simulate_demag


def simulate_task(index):
    chi_perp = np.random.uniform(0.0, 1.0)
    chi_long = np.random.uniform(0.0, 1.0)
    chi = (chi_perp, chi_perp, chi_long)

    a = np.random.uniform(low=0.3, high=3.0)
    b = np.random.uniform(low=0.3, high=3.0)
    # chi = np.random.uniform(0, 1)

    if a > b:
        a, b = b, a

    print(f"Starting simuation: {index}")
    data = simulate_demag(a, b, chi, 10)
    path = f"data/anisotropic_chi/eval/data_coarse_{index}.npz"
    np.savez(path, **data)


if __name__ == "__main__":
    for idx in range(2, 3):
        simulate_task(idx)
