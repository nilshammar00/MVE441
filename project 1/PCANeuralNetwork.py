import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
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


def dimRedPCA(trainingSet, testSet, trainVariance = .95):
    # Reshape for PCA-algorithm
    xTrain = trainingSet.reshape(1800, 256)
    xTest = testSet.reshape(200, 256)
    print('Trainset shape post transform:', xTrain.shape)
    print('Testset shape posst transform:', xTest.shape)
     # How much of the original variance is retained through the PCA
    pca = PCA(n_components=trainVariance)
    pca.fit(xTrain) # fit the PCA to the training data
    print('Number of components before fit', xTrain.shape)
    print('Number of Components post fit:', pca.n_components_)
    
    trainPCA = pca.transform(xTrain)
    testPCA = pca.transform(xTest)

    return trainPCA, testPCA
    

def dataNNClassifier(trainingLabels, trainingData, testLabels, testData):
    clf = MLPClassifier(solver='sgd', activation='logistic', max_iter=5000) # sgd = stochastic gradient descent, logistic function for the hidden layer, max_iter = maximum number of itteration, but will conclude after 10 epocs without improvement
    clf.fit(trainingData, trainingLabels)
    print('Training score:', clf.score(trainingData, trainingLabels))
    print('Testing score', clf.score(testData, testLabels))
    predictedLabels = clf.predict(testData)
    print('Shape prediction list', predictedLabels.shape)
    return predictedLabels
    
def logisticNNPCA(filePath ='Numbers.txt', trainVariance = .95):
    # first arg defines the data set
    # Second arg sets the variance for the PCA reduction
    images, labels = read_mnist_txt(filePath)

    trainingSet, trainingLabels, testSet, testSetLabels = dataPrep(images, labels)

    trainPCA, testPCA = dimRedPCA(trainingSet, testSet, trainVariance)

    predictedLabels = dataNNClassifier(trainingLabels, trainPCA, testSetLabels, testPCA)
    print(predictedLabels)
    return predictedLabels


if __name__ == '__main__':
    logisticNNPCA()
        
