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

def Perceptron_Single_Layer(trainingSet, trainingLabels, testSet, iterations=1000, learningRate=0.1):
    # Create a mapping for labels
    unique_labels = np.unique(trainingLabels)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    # Map training and test labels to indices
    mapped_trainingLabels = np.array([label_to_index[label] for label in trainingLabels])
    mapped_testLabels = np.array([label_to_index[label] for label in testSetLabels])

    num_classes = len(unique_labels)  # Determine the number of classes

    ## initialize weights and bias
    weights = np.zeros((trainingSet.shape[1] * trainingSet.shape[2], num_classes))  # 256 (flattened) x num_classes
    bias = np.zeros(num_classes)
    trainingSet = trainingSet.reshape(trainingSet.shape[0], -1)  # Flatten the images
    testSet = testSet.reshape(testSet.shape[0], -1)  # Flatten the images
    
    for _ in range(iterations):
        for i in range(trainingSet.shape[0]):
            # Compute the linear combination of inputs and weights
            linear_output = np.dot(trainingSet[i], weights) + bias
            
            # Apply the step function to determine the predicted class
            predicted_label = np.argmax(linear_output)
            
            # Check if the prediction is correct
            true_label = mapped_trainingLabels[i]
            if predicted_label != true_label:
                # Update weights and bias for the true label
                weights[:, true_label] += learningRate * trainingSet[i]
                bias[true_label] += learningRate
                
                # Update weights and bias for the predicted label
                weights[:, predicted_label] -= learningRate * trainingSet[i]
                bias[predicted_label] -= learningRate

    # Classify the training set
    classified_labels = []
    for i in range(trainingSet.shape[0]):
        linear_output = np.dot(trainingSet[i], weights) + bias
        classified_label = np.argmax(linear_output)
        classified_labels.append(index_to_label[classified_label])  # Map back to original labels

    return classified_labels

# Split the data into training and test sets

images, labels = read_mnist_txt('Numbers.txt')
randomIndices = np.random.permutation(len(images))
images = np.array(images)[randomIndices]
labels = np.array(labels)[randomIndices]

trainingSet = images[:len(images)-200]
trainingLabels = labels[:len(images)-200]
testSet = images[-200:]
testSetLabels = labels[-200:]

predictedLabels = Perceptron_Single_Layer(trainingSet, trainingLabels, testSet)
# Calculate accuracy
correct = 0
for i in range(len(testSet)):
    if predictedLabels[i] == testSetLabels[i]:
        correct += 1
accuracy = correct / len(testSet) * 100
print(f'Accuracy: {accuracy:.2f}%')