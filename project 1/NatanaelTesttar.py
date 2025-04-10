import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
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
    #print('Shape of training set:', trainingSet.shape)
    #print('Shape of test set:', testSet.shape)

    return trainingSet, trainingLabels, testSet, testSetLabels


def dataReshaper(trainingSet, testSet):
    xTrain = trainingSet.reshape(300, 256)
    xTest = testSet.reshape(300, 256)
    print('Trainset shape post transform:', xTrain.shape)
    print('Testset shape posst transform:', xTest.shape)
    return xTrain, xTest


"""

NeuralNetwork + PCA

"""

def dimRedPCA(trainingSet, testSet, trainVariance = .95):

    xTrain, xTest = dataReshaper(trainingSet, testSet)

    # How much of the original variance is retained through the PCA
    pca = PCA(n_components=trainVariance)
    pca.fit(xTrain) # fit the PCA to the training data
    #print('Number of components before fit', xTrain.shape)
    #print('Number of Components post fit:', pca.n_components_)
    
    trainPCA = pca.transform(xTrain)
    testPCA = pca.transform(xTest)

    return trainPCA, testPCA

def dataNNClassifier(trainingLabels, trainingData, testLabels, testData, maxIter = 5000):
    clf = MLPClassifier(solver='sgd', activation='logistic', max_iter=maxIter) # sgd = stochastic gradient descent, logistic function for the hidden layer, max_iter = maximum number of itteration, but will conclude after 10 epocs without improvement
    clf.fit(trainingData, trainingLabels)
    #print('Training error score:', 1-clf.score(trainingData, trainingLabels))
    #print('Testing error score', 1-clf.score(testData, testLabels))
    predictedLabels = clf.predict(testData)
    #print('Shape prediction list', predictedLabels.shape)
    return 1-clf.score(testData, testLabels)

"""
Random Forest
"""

def randomForestClassifier(trainingSet, trainingLabels, testSet, testLabels, n_estimators = 100):
    # Flatten 16x16 images to 256-length vectors for classification
    X_train, X_test = dataReshaper(trainingSet, testSet)
    
    # Initialize Random Forest
    # n_estimators: number of trees in the forest
    # random_state: random speed for reproducibility
    rf = RandomForestClassifier(n_estimators)
    
    # Train the Classifier
    rf.fit(X_train, trainingLabels)
    print(X_test.shape) 
    print(testSetLabels.shape)
    #print('Training error score:', 1-rf.score(X_train, trainingLabels))
    #print('Testing error score', 1-rf.score(X_test, testLabels))
    # Predict using trained classifier on testSet
    #predictedLabels = rf.predict(X_test)
    
    return 1-rf.score(X_test, testLabels)


"""

Call on models

"""
    
def logisticNNPCA(trainingSet, trainingLabels, testSet, testSetLabels, maxIter = 5000, trainVariance = .95):
    # first arg defines the data set
    # Second arg sets the variance for the PCA reduction


    trainPCA, testPCA = dimRedPCA(trainingSet, testSet, trainVariance)
    #print('NN+PCA')
    predictedLabels = dataNNClassifier(trainingLabels, trainPCA, testSetLabels, testPCA, maxIter)
    #print(f"NN+PCA", predictedLabels)
    return predictedLabels

def randomForest(trainingSet, trainingLabels, testSet, testSetLabels, numberEstimators=100):

    #print ('Random Forest')
    predictedLabels = randomForestClassifier(trainingSet, trainingLabels, testSet, testSetLabels, n_estimators=numberEstimators)
    #print(f"RF", predictedLabels)
    return predictedLabels


def kNearestNeighboors(trainingSet, trainingLabels, testSet, testSetLabels, k=5, norm=2):
    ## find distance of each entry in testSet to each entry in trainingSet
    ## take the k nearest neighbors, and assign the label of the majority
    ## of those k neighbors to the testSet entry
    ## return the testSet with the labels assigned

    #! add Cross Validation error for different k values
    #print('KNN')    
    predictedLabels = []
    for index,image in enumerate(testSet):
        distances = []
        for i in range(len(trainingSet)):
            distance = np.linalg.norm(trainingSet[i]-image,ord=norm)
            distances.append((distance,trainingLabels[i]))
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]
        labels = [neighbor[1] for neighbor in neighbors]
        label = max(labels, key=labels.count)
        predictedLabels.append(label)

    correct = 0
    for i in range(len(testSet)):
        if predictedLabels[i] == testSetLabels[i]:
            correct += 1
        accuracy = correct / len(testSet)
    #print(f'Accuracyerror:', 1-accuracy)
    #print(f'KNN', predictedLabels)
    return 1-accuracy

""""
init

"""
def loadMINIST(filePath = 'Numbers.txt'):
    images, labels = read_mnist_txt(filePath)
    trainingSet, trainingLabels, testSet, testSetLabels = dataPrep(images, labels)
    
    return trainingSet, trainingLabels, testSet, testSetLabels


def Misslabeling(labels,percentage):
    ## randomly select percentage of labels to be mislabelled
    ## return the mislabelled labels

    listOfLabels = []
    for i in range(len(labels)):
        if labels[i] not in listOfLabels:
            listOfLabels.append(labels[i])

    # Calculate the number of labels to mislabel
    numToMisslabel = int(len(labels) * percentage)
    indicesToMisslabel = np.random.choice(len(labels), numToMisslabel, replace=False)
    misslabelledLabels = labels.copy()
    
    for index in indicesToMisslabel:
        # Randomly select a new label that is different from the original label
        originalLabel = labels[index]
        newLabel = originalLabel
        while newLabel == originalLabel:
            newLabel = listOfLabels[np.random.randint(0, len(listOfLabels))]  # Assuming labels are from 0 to 9
        misslabelledLabels[index] = newLabel
    
    return misslabelledLabels 

if __name__ == '__main__':
    
    
    trainingSet, BasetrainingLabels, testSet, testSetLabels = loadMINIST()
    PCATuning = 5000
    RFTuning = 400
    KNNTuning = 4
    evalLength = 12
    NN1 = []
    RF1 = []
    KNN1 = []
    NN2 = []
    RF2 = []
    KNN2 = []
    NN3 = []
    RF3 = []
    KNN3 = []

    Train1, Train2, Train3, Train4, Train5, Test6 = np.array_split(trainingSet, 6)
    LTrain1, LTrain2, LTrain3, LTrain4, LTrain5, LTest6 = np.array_split(BasetrainingLabels, 6)
    """
    for i in range(evalLength):
        #trainingLabels = Misslabeling(BasetrainingLabels, .1)
        trainingLabels = BasetrainingLabels
        nn = logisticNNPCA(trainingSet, trainingLabels, testSet, testSetLabels, PCATuning)
        rf = randomForest(trainingSet, trainingLabels, testSet, testSetLabels, RFTuning)
        knn = kNearestNeighboors(trainingSet, trainingLabels, testSet, testSetLabels,KNNTuning)
        NN1.append(nn)
        RF1.append(rf)
        KNN1.append(knn)

    for i in range(evalLength):
        #trainingLabels = Misslabeling(BasetrainingLabels, .15)
        trainingLabels = BasetrainingLabels
        nn = logisticNNPCA(trainingSet, trainingLabels, testSet, testSetLabels, PCATuning)
        rf = randomForest(trainingSet, trainingLabels, testSet, testSetLabels, RFTuning)
        knn = kNearestNeighboors(trainingSet, trainingLabels, testSet, testSetLabels,KNNTuning)
        NN2.append(nn)
        RF2.append(rf)
        KNN2.append(knn)

    for i in range(evalLength):
        #trainingLabels = Misslabeling(BasetrainingLabels, .3)
        trainingLabels = BasetrainingLabels
        nn = logisticNNPCA(trainingSet, trainingLabels, testSet, testSetLabels, PCATuning)
        rf = randomForest(trainingSet, trainingLabels, testSet, testSetLabels, RFTuning)
        knn = kNearestNeighboors(trainingSet, trainingLabels, testSet, testSetLabels,KNNTuning)
        NN3.append(nn)
        RF3.append(rf)
        KNN3.append(knn)

    missError = [NN1,NN2,NN3, RF1,RF2,RF3, KNN1,KNN2,KNN3]

    plt.figure()
    plt.title('Classifier Performance with misslabeled Data')
    plt.xlabel('Model with % of misslabeled data')
    plt.ylabel('Error score of model')
    plt.boxplot(missError, positions=[1,2,3,5,6,7,9,10,11], labels=['10%','NN+PCA 15%','30%', '10%','RF 15%', '30%', '10%', 'KNN 15%', '30%'])
    plt.show()
    """


    print(1, Train1.shape, LTrain1.shape, Test6.shape, LTest6.shape)
    nn = logisticNNPCA(Train1, LTrain1, Test6, LTest6, PCATuning)
    rf = randomForest(Train1, LTrain1, Test6, LTest6, RFTuning)
    knn = kNearestNeighboors(Train1, LTrain1, Test6, LTest6 ,KNNTuning)
    NN1.append(nn)
    RF1.append(rf)
    KNN1.append(knn)
    print(2)
    nn = logisticNNPCA(Train2, LTrain2, Test6, LTest6, PCATuning)
    rf = randomForest(Train2, LTrain2, Test6, LTest6, RFTuning)
    knn = kNearestNeighboors(Train2, LTrain2, Test6, LTest6 ,KNNTuning)
    NN1.append(nn)
    RF1.append(rf)
    KNN1.append(knn)
    print(3)
    nn = logisticNNPCA(Train3, LTrain3, Test6, LTest6, PCATuning)
    rf = randomForest(Train3, LTrain3, Test6, LTest6, RFTuning)
    knn = kNearestNeighboors(Train3, LTrain3, Test6, LTest6 ,KNNTuning)
    NN1.append(nn)
    RF1.append(rf)
    KNN1.append(knn)
    print(4)
    nn = logisticNNPCA(Train4, LTrain4, Test6, LTest6, PCATuning)
    rf = randomForest(Train4, LTrain4, Test6, LTest6, RFTuning)
    knn = kNearestNeighboors(Train4, LTrain4, Test6, LTest6 ,KNNTuning)
    NN1.append(nn)
    RF1.append(rf)
    KNN1.append(knn)
    print(5)
    nn = logisticNNPCA(Train5, LTrain5, Test6, LTest6, PCATuning)
    rf = randomForest(Train5, LTrain5, Test6, LTest6, RFTuning)
    knn = kNearestNeighboors(Train5, LTrain5, Test6, LTest6 ,KNNTuning)
    NN1.append(nn)
    RF1.append(rf)
    KNN1.append(knn)


    classError = [NN1, RF1, KNN1]

    plt.figure()
    plt.title('Tuned Classifier Performance')
    plt.xlabel('Tuned models')
    plt.ylabel('Error score of model')
    plt.boxplot(classError, positions=[1,3,5], labels=['NN+PCA','RF','KNN'])
    plt.show()


