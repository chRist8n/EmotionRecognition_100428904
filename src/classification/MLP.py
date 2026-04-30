import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, lr=0.1):
        self.lr = lr

        # weights
        self.W1 = np.random.randn(input_size, hidden_size_1) * 0.05
        self.b1 = np.zeros((1, hidden_size_1))

        self.W2 = np.random.randn(hidden_size_1, hidden_size_2) * 0.05
        self.b2 = np.zeros((1, hidden_size_2))

        self.W3 = np.random.randn(hidden_size_2, output_size) * 0.05
        self.b3 = np.zeros((1, output_size))

    ## Forward Pass
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.relu(self.Z2)

        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = self.softmax(self.Z3)

        return self.A3
    
    ## Loss Function
    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]

        log_likelihood = -np.log(y_pred[range(m), y_true] + 1e-9)
        weighted = log_likelihood * self.class_weights[y_true]

        return np.mean(weighted)
    
    ## Backpropagation
    def backward(self, X, y_true):
        m = X.shape[0]

        dZ3 = self.A3.copy()
        dZ3[range(m), y_true] -= 1
        dZ3 *= self.class_weights[y_true][:, None]
        dZ3 /= m

        # Output layer
        dW3 = self.A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        # Hidden layer 2
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * (self.Z2 > 0)

        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer 1
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0)

        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

    ## Training Loop
    def train(self, X, y, epochs=100):
        class_counts = np.bincount(y)
        class_weights = np.median(class_counts) / (class_counts + 1e-6)
        self.class_weights = class_weights / np.mean(class_weights)

        for epoch in range(epochs):
            perm = np.random.permutation(len(X))
            X = X[perm]
            y = y[perm]

            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)

            self.backward(X, y)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")


    def predict_proba(self, X):
        return self.forward(X)

    ## Prediction
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
