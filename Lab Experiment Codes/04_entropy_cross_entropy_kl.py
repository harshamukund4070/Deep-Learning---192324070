# 4. Entropy, Cross-Entropy and KL Divergence
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr

def main():
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.5, 0.3, 0.2])
    entropy = -np.sum(p * np.log2(p))
    cross_entropy = -np.sum(p * np.log2(q))
    kl = np.sum(rel_entr(p, q))  # natural-log KL divergence
    print("P:", p, "\nQ:", q)
    print(f"Entropy H(P): {entropy:.4f} bits")
    print(f"Cross-entropy H(P,Q): {cross_entropy:.4f} bits")
    print(f"KL divergence D_KL(P||Q): {kl:.4f} nats")
    labels = ["Class 1", "Class 2", "Class 3"]
    x = np.arange(3); width = .35
    plt.bar(x-width/2, p, width, label="P (true)")
    plt.bar(x+width/2, q, width, label="Q (predicted)")
    plt.xticks(x, labels); plt.ylabel("Probability"); plt.title("True vs predicted distribution")
    plt.legend(); plt.show()

if __name__ == "__main__":
    main()
