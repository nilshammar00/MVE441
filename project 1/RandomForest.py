import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def read_mnist_txt(file_path):
    images = []
    labels = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if not line.strip():
                continue
            parts = line.strip().split()
            label = int(parts[1])
            pixels = list(map(float, parts[2:]))
            pixel_array = np.array(pixels).reshape(16, 16)
            normalized_array = (pixel_array + 1) / 2
            images.append(normalized_array)
            labels.append(label)
    
    return images, labels

def randomForestClassifier(trainingSet, trainingLabels, testSet, n_estimators=100):
    # Flatten 16x16 images to 256-length vectors for classification
    X_train = [img.flatten() for img in trainingSet]
    X_test = [img.flatten() for img in testSet]
    
    # Initialize Random Forest
    # n_estimators: number of trees in the forest
    # random_state: random speed for reproducibility
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the Classifier
    rf.fit(X_train, trainingLabels)
    
    # Predict using trained classifier on testSet
    predictedLabels = rf.predict(X_test)

    return predictedLabels

# Load and prepare data
print(os.getcwd())
images, labels = read_mnist_txt(r"C:\Users\rebec\OneDrive\Dokument\GitHub\MVE441\project 1\Numbers.txt")

randomIndices = np.random.permutation(len(images))
images = np.array(images)[randomIndices]
labels = np.array(labels)[randomIndices]

trainingSet = images[:len(images)-200]
trainingLabels = labels[:len(images)-200]
testSet = images[-200:]
testSetLabels = labels[-200:]

# Call the classifier
predictedLabels = randomForestClassifier(trainingSet, trainingLabels, testSet, n_estimators=100)

# Accuracy
accuracy = accuracy_score(testSetLabels, predictedLabels) * 100
print(f'Accuracy: {accuracy:.2f}%')

# Visualize results
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(testSet[i], cmap='gray')
    plt.title(f'Predicted: {predictedLabels[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()
