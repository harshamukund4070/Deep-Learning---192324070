# 9. Multilayer Perceptron using TensorFlow/Keras
# Install: pip install tensorflow scikit-learn matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def main():
    X,y=make_moons(n_samples=1000, noise=.2, random_state=42)
    X=StandardScaler().fit_transform(X)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    model=tf.keras.Sequential([tf.keras.layers.Input((2,)),tf.keras.layers.Dense(16,activation="relu"),
        tf.keras.layers.Dense(8,activation="relu"),tf.keras.layers.Dense(1,activation="sigmoid")])
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    h=model.fit(Xtr,ytr,epochs=30,batch_size=32,validation_split=.2,verbose=0)
    print("Test loss, accuracy:",model.evaluate(Xte,yte,verbose=0))
    plt.plot(h.history["accuracy"],label="train");plt.plot(h.history["val_accuracy"],label="validation")
    plt.xlabel("Epoch");plt.ylabel("Accuracy");plt.title("MLP learning curves");plt.legend();plt.show()

if __name__=="__main__": main()
