from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import os

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


def dataPrep(images, labels):
    randomIndices = np.random.permutation(len(images))
    images = np.array(images)[randomIndices]
    labels = np.array(labels)[randomIndices]

    trainingSet = images[:len(images)-200]
    trainingLabels = labels[:len(images)-200]
    testSet = images[-200:]
    testSetLabels = labels[-200:]
    print('Shape of training set:', trainingSet.shape)
    print('Shape of test set:', testSet.shape)
    


    return trainingSet, trainingLabels, testSet, testSetLabels

def LDA(filePath = 'Numbers.txt'):
    images, labels = read_mnist_txt(filePath)

    trainingSet, trainingLabels, testSet, testSetLabels = dataPrep(images, labels)
    xTrain = trainingSet.reshape(1800, 256)
    xTest = testSet.reshape(200, 256)
    lda = LinearDiscriminantAnalysis()
    lda.fit(xTrain, trainingLabels)

    predictedLabels = lda.predict(xTest)
    print(f"traindata score", lda.score(xTrain, trainingLabels))
    print(f"Testdata score", lda.score(xTest, testSetLabels))
    return predictedLabels

if __name__ == '__main__':
    LDA()
           