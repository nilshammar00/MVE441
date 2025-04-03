# Random forrest

import numpy as np
import matplotlib.pyplot as plt

def RF(test_set, train_set):
    # Initialize the Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the Classifier
    clf.fit(train_set)

    # Prediction

    # Evaluation of classifier