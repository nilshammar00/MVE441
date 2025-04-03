from sklearn.ensemble import RandomForestClassifier

def RF(trainingSet,trainingLabels,testSet):
    # Initialize the Classifier
    # n_estimators: number of trees in the forest
    # random_state: random speed for reproducibility
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the Classifier
    clf.fit(trainingSet, trainingLabels)

    # Prediction using trained classifier on testSet
    predictions = clf.predict(testSet)

    return predictions