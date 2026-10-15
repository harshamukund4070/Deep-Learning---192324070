# 7. Perceptron and Forward Propagation
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z): return 1/(1+np.exp(-np.clip(z, -500, 500)))
def main():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    w = np.array([1.0, 1.0]); b = -1.5
    z = X @ w + b; y = (z >= 0).astype(int)
    print("Inputs:\n", X); print("Linear outputs:", z); print("Perceptron predictions:", y)
    print("Sigmoid probabilities:", sigmoid(z))
    xx = np.linspace(-.2, 1.2, 100)
    plt.scatter(X[:,0], X[:,1], c=y, s=100, cmap="coolwarm")
    plt.plot(xx, 1.5-xx, "k--", label="Decision boundary")
    plt.xlabel("x1"); plt.ylabel("x2"); plt.title("Perceptron decision boundary")
    plt.legend(); plt.grid(); plt.show()

if __name__ == "__main__":
    main()
