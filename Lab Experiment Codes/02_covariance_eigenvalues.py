# 2. Covariance, Eigenvalues and Eigenvectors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    n = int(input("Enter number of data points (minimum 3): ") or "100")
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, n)
    y = 0.7 * x + rng.normal(0, 0.6, n)
    data = pd.DataFrame({"X": x, "Y": y})
    cov = data.cov().to_numpy()
    values, vectors = np.linalg.eig(cov)
    print("First five rows:\n", data.head())
    print("Covariance matrix:\n", cov)
    print("Eigenvalues:", values)
    print("Eigenvectors (columns):\n", vectors)

    plt.scatter(x, y, alpha=.65, label="Data")
    center = data.mean().to_numpy()
    for i in range(2):
        direction = vectors[:, i] * np.sqrt(values[i]) * 2
        plt.arrow(*center, *direction, color=["red", "green"][i], width=.01, length_includes_head=True)
    plt.title("Data and eigenvector directions"); plt.xlabel("X"); plt.ylabel("Y"); plt.legend(); plt.grid(); plt.show()

if __name__ == "__main__":
    main()
