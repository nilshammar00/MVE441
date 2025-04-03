import numpy as np
import matplotlib.pyplot as plt


    

def kNearestNeighboors(trainingSet,trainingLabels,testSet, k,norm):
    ## find distance of each entry in testSet to each entry in trainingSet
    ## take the k nearest neighbors, and assign the label of the majority
    ## of those k neighbors to the testSet entry
    ## return the testSet with the labels assigned

    for index,image in enumerate(testSet):
        distances = []
        for i in range(len(trainingSet)):
            distance = np.linalg.norm(trainingSet[i]-image,ord=norm)
            distances.append((distance,trainingLabels[i]))
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]
        labels = [neighbor[1] for neighbor in neighbors]
        label = max(labels, key=labels.count)





