import numpy as np
import matplotlib.pyplot as plt

    
def read_mnist_txt(file_path):
    images = []
    labels = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Split the line into components
            parts = line.strip().split()
            
            # The first part is the image number in quotes (e.g., "1"), 
            # the second is the label, and the rest are pixel values
            image_num = parts[0].strip('"')
            label = int(parts[1])
            pixels = list(map(float, parts[2:]))
            
            # Convert pixel values to a 16x16 numpy array
            # The values are between -1 and 1, so we normalize to 0-1 for display
            pixel_array = np.array(pixels).reshape(16, 16)
            normalized_array = (pixel_array + 1) / 2  # Scale from [-1, 1] to [0, 1]
            
            images.append(normalized_array)
            labels.append(label)
    
    return images, labels


def kNearestNeighboors(trainingSet, trainingLabels, testSet, k=5, norm=2):
    ## find distance of each entry in testSet to each entry in trainingSet
    ## take the k nearest neighbors, and assign the label of the majority
    ## of those k neighbors to the testSet entry
    ## return the testSet with the labels assigned

    #! add Cross Validation error for different k values
    testSetLabels = []
    for index,image in enumerate(testSet):
        distances = []
        for i in range(len(trainingSet)):
            distance = np.linalg.norm(trainingSet[i]-image,ord=norm)
            distances.append((distance,trainingLabels[i]))
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]
        labels = [neighbor[1] for neighbor in neighbors]
        label = max(labels, key=labels.count)
        testSetLabels.append(label)
    return testSetLabels

import os
print(os.getcwd())


images, labels = read_mnist_txt('Numbers.txt')

# Split the data into training and test sets
randomIndices = np.random.permutation(len(images))
images = np.array(images)[randomIndices]
labels = np.array(labels)[randomIndices]

trainingSet = images[:len(images)-200]
trainingLabels = labels[:len(images)-200]
testSet = images[-200:]
testSetLabels = labels[-200:]
k = 5
norm = 2
predictedLabels = kNearestNeighboors(trainingSet, trainingLabels, testSet, k, norm)
# Calculate accuracy
correct = 0
for i in range(len(testSet)):
    if predictedLabels[i] == testSetLabels[i]:
        correct += 1
accuracy = correct / len(testSet) * 100
print(f'Accuracy: {accuracy:.2f}%')
# Visualize the first 10 test images and their predicted labels
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(testSet[i], cmap='gray')
    plt.title(f'Predicted: {predictedLabels[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()
