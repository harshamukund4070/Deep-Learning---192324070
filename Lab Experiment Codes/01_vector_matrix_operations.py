import numpy as np
import matplotlib.pyplot as plt
def main():
    a = np.array([1, 2, 3], dtype=float)
    b = np.array([4, 5, 6], dtype=float)
    A = np.array([[1, 2], [3, 4]], dtype=float)
    B = np.array([[5, 6], [7, 8]], dtype=float)
    print("Vector addition:", a + b)
    print("Dot product:", np.dot(a, b))
    print("Matrix addition:\n", A + B)
    print("Matrix multiplication:\n", A @ B)
    print("Transpose of A:\n", A.T)
    print("Determinant of A:", np.linalg.det(A))
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1); plt.imshow(A, cmap="Blues"); plt.title("Matrix A"); plt.colorbar()
    plt.subplot(1, 2, 2); plt.imshow(A @ B, cmap="Oranges"); plt.title("A @ B"); plt.colorbar()
    plt.tight_layout(); plt.show()
if __name__ == "__main__":
    main()