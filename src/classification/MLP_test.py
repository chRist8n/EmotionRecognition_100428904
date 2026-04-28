import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    max_iter=500,
    alpha=0.001,
    random_state=42,
    early_stopping=True
)

model.fit(X_resampled, y_resampled)
#model.fit(X_train, y_train, sample_weight=sample_weights)

y_pred = model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))



#===============================================================#

#test importances
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model,
    X_val,
    y_val,
    n_repeats=10,
    random_state=42,
    scoring="accuracy"
)

importances = result.importances_mean
indices = np.argsort(importances)[::-1]

feature_names = [
    "left_eye_open",
    "right_eye_open",
    "eye_squint",
    #"cheek_raise",
    "eye_mouth_dist",
    "mouth_open",
    "mouth_width",
    "mouth_area_proxy",
    "lip_compression",
    "mouth_eye_ratio",
    "mouth_curve",
    "mouth_stretch",
    "mouth_corner_diff",
    "left_brow_height",
    "right_brow_height",
    "left_brow_tilt",
    "right_brow_tilt",
    "brow_eye_ratio",
    "lip_ratio",
    "eye_diff",
    "eye_y_diff",
    "eye_inner_y_diff",
    "neutral_score",
    "face_asymmetry"
    #"eye_x_diff",
    #"mouth_asym",
    #"brow_diff",
]

for i in indices:
    name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
    print(name, importances[i])


