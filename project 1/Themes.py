import numpy as np

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