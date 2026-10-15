# 10. L2, Dropout and Early Stopping Comparison
# Install: pip install tensorflow scikit-learn matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    X,y=make_moons(n_samples=1200,noise=.3,random_state=5)
    X=StandardScaler().fit_transform(X)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    configs={
        "L2": [tf.keras.layers.Dense(64,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(.01))],
        "Dropout": [tf.keras.layers.Dense(64,activation="relu"),tf.keras.layers.Dropout(.4)],
        "Baseline": [tf.keras.layers.Dense(64,activation="relu")]
    }
    for name,layers in configs.items():
        model=tf.keras.Sequential([tf.keras.layers.Input((2,))]+layers+[tf.keras.layers.Dense(32,activation="relu"),tf.keras.layers.Dense(1,activation="sigmoid")])
        model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)] if name!="Baseline" else []
        h=model.fit(Xtr,ytr,validation_split=.2,epochs=50,batch_size=32,verbose=0,callbacks=callbacks)
        print(name,"test metrics:",model.evaluate(Xte,yte,verbose=0))
        plt.plot(h.history["val_loss"],label=name)
    plt.xlabel("Epoch");plt.ylabel("Validation loss");plt.title("Regularization comparison");plt.legend();plt.grid();plt.show()

if __name__=="__main__": main()
