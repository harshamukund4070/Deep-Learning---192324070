# 8. Two-Layer Neural Network with Backpropagation (NumPy)
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))
def main():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], float)
    y = np.array([[0],[1],[1],[0]], float)  # XOR
    rng = np.random.default_rng(4)
    W1 = rng.normal(0, .8, (2, 4)); b1 = np.zeros((1,4))
    W2 = rng.normal(0, .8, (4, 1)); b2 = np.zeros((1,1))
    losses=[]
    for epoch in range(10000):
        h = sigmoid(X@W1+b1); out = sigmoid(h@W2+b2)
        loss = -np.mean(y*np.log(out+1e-9)+(1-y)*np.log(1-out+1e-9))
        losses.append(loss)
        d2=(out-y)/len(X); dW2=h.T@d2; db2=d2.sum(axis=0, keepdims=True)
        d1=(d2@W2.T)*h*(1-h); dW1=X.T@d1; db1=d1.sum(axis=0, keepdims=True)
        W2-=1.0*dW2; b2-=1.0*db2; W1-=1.0*dW1; b1-=1.0*db1
    print("XOR predictions:\n", np.round(out,3))
    print("Predicted classes:", (out>=.5).astype(int).ravel())
    plt.plot(losses); plt.xlabel("Epoch"); plt.ylabel("Binary cross-entropy"); plt.title("Backpropagation training loss")
    plt.grid(); plt.show()

if __name__ == "__main__":
    main()
