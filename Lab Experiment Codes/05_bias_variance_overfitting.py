# 5. Bias-Variance, Overfitting and Underfitting
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

def main():
    rng = np.random.default_rng(7)
    x = np.linspace(0, 1, 35)
    y = np.sin(2*np.pi*x) + rng.normal(0, .18, len(x))
    xt = np.linspace(0, 1, 300)
    plt.scatter(x, y, color="black", label="Training data")
    for degree in [1, 4, 14]:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(x[:, None], y)
        pred = model.predict(xt[:, None])
        train_mse = mean_squared_error(y, model.predict(x[:, None]))
        print(f"Degree {degree}: training MSE = {train_mse:.4f}")
        plt.plot(xt, pred, label=f"Degree {degree}")
    plt.plot(xt, np.sin(2*np.pi*xt), "--", label="True pattern")
    plt.ylim(-2, 2); plt.title("Underfitting vs reasonable fit vs overfitting")
    plt.legend(); plt.grid(); plt.show()

if __name__ == "__main__":
    main()
