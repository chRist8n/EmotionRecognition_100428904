import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    max_iter=300,
    random_state=42,
    early_stopping=True
)

model.fit(X_train, y_train, sample_weight=sample_weights)

y_pred = model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))