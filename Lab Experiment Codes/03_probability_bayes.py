# 3. Probability Distributions and Bayes' Theorem
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom

def main():
    p_disease = float(input("Prior probability of condition (0-1) [0.01]: ") or "0.01")
    sensitivity = float(input("Test sensitivity (0-1) [0.95]: ") or "0.95")
    specificity = float(input("Test specificity (0-1) [0.90]: ") or "0.90")
    if not all(0 <= v <= 1 for v in (p_disease, sensitivity, specificity)):
        raise ValueError("All probabilities must be between 0 and 1.")
    denominator = sensitivity*p_disease + (1-specificity)*(1-p_disease)
    posterior = sensitivity*p_disease/denominator if denominator else float("nan")
    print(f"P(condition | positive test) = {posterior:.4f} ({posterior*100:.2f}%)")

    x = np.linspace(-4, 4, 300)
    k = np.arange(0, 21)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(x, norm.pdf(x), label="Normal(0,1)"); ax[0].set_title("Normal distribution"); ax[0].legend()
    ax[1].bar(k, binom.pmf(k, 20, .4)); ax[1].set_title("Binomial(n=20, p=0.4)"); ax[1].set_xlabel("Successes")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
