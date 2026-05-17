from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

################################################################
# Class Distribution Chart
labels = ["neutral", "happy", "sad", "surprise", "anger", "fear", "disgust"]
counts = [566, 1020, 410, 122, 296, 63, 136]

plt.figure(figsize=(8,5))
plt.bar(labels, counts)

plt.title("Class Distribution in Training Dataset")
plt.xlabel("Emotion Class")
plt.ylabel("Number of Samples")

plt.xticks(rotation=30)
plt.tight_layout()

plt.bar(labels, counts)

for i, v in enumerate(counts):
    plt.text(i, v + 10, str(v), ha='center')
    
plt.savefig("backend/figure_generation/class_distribution.png", dpi=300)
plt.show()

################################################################
# Loss per epoch plot
epochs = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275]
loss = [0.6851, 0.3900, 0.3612, 0.6396, 0.6682, 0.4351,
        0.3145, 0.3603, 0.2981, 0.4736, 0.6885, 0.2904]

smoothed = np.convolve(loss, np.ones(3)/3, mode='valid')

plt.figure(figsize=(7,4))
plt.plot(epochs, loss, marker='o')

plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.tight_layout()
plt.savefig("backend/figure_generation/training_loss.png", dpi=300)
plt.show()

#################################################################
# confusion matrix

y_pred = np.load("backend/figure_generation/y_pred.npy")
y_true = np.load("backend/figure_generation/y_true.npy")

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7,6))

sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt=".2f"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.savefig("backend/figure_generation/confusion_matrix.png", dpi=300)
plt.show()

#################################################################
# normalised confusion matrix

cm_normalised = cm.astype('float') / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(7,6))

sns.heatmap(
    cm_normalised,
    annot=True,
    cmap="Blues",
    fmt=".2f"
)

plt.title("Normalised Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.savefig("backend/figure_generation/normalised_confusion_matrix.png", dpi=300)
plt.show()

