# 6. Gradient Descent, SGD, Momentum and Adam comparison
import numpy as np
import matplotlib.pyplot as plt

def grad(x): return 2*(x-3)
def main():
    steps = int(input("Number of optimization steps [50]: ") or "50")
    lr = 0.1
    methods = ["Gradient Descent", "SGD-like noisy", "Momentum", "Adam"]
    histories = {}
    for method in methods:
        x, velocity, m, v = 0.0, 0.0, 0.0, 0.0
        hist = []
        for t in range(1, steps+1):
            g = grad(x)
            if method == "SGD-like noisy": g += np.random.default_rng(t).normal(0, .8)
            if method == "Momentum":
                velocity = .9*velocity + g
                x -= lr*velocity
            elif method == "Adam":
                m = .9*m + .1*g; v = .999*v + .001*g*g
                mh = m/(1-.9**t); vh = v/(1-.999**t)
                x -= .2*mh/(np.sqrt(vh)+1e-8)
            else: x -= lr*g
            hist.append((x-3)**2)
        histories[method] = hist
        print(f"{method}: final x={x:.4f}, loss={(x-3)**2:.6f}")
    for name, values in histories.items(): plt.plot(values, label=name)
    plt.yscale("log"); plt.xlabel("Step"); plt.ylabel("Squared error"); plt.title("Optimizer comparison")
    plt.legend(); plt.grid(); plt.show()

if __name__ == "__main__":
    main()
