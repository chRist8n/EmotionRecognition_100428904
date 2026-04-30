import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import classification.MLP as mlp

X_train = np.load("data/processed/train/X.npy")
y_train = np.load("data/processed/train/y.npy")

X_test = np.load("data/processed/test/X.npy")
y_test = np.load("data/processed/test/y.npy")


def main():
    #Test MLP
    model = mlp.MLP(input_size=19, hidden_size_1=64, hidden_size_2=32, output_size=7)
    model.train(X_train, y_train, epochs=500)

    preds = model.predict(X_test)

    print("\n\nDone: \n")

    print("Pred shape:", preds.shape)
    print("Test shape:", y_test.shape)

    accuracy = np.mean(preds == y_test)
    print("Accuracy:", accuracy)

    print("Pred counts:", np.bincount(preds))
    print("True counts:", np.bincount(y_test))
    print(confusion_matrix(y_test, preds))

    print(classification_report(y_test, preds))


    # X = np.random.rand(1, 23)
    # #X = np.random.rand(5, 23)

    # output = model.forward(X)
    # print("output:", output)
    # print("shape:", output.shape)

    # print(np.sum(output, axis=1))


if __name__ == "__main__":
   main()