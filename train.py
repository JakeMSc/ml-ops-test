from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
import os
import joblib
import numpy as np

# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")


# Fit a model
clf = GaussianNB()
clf.fit(X_train,y_train)

# Calculate accuracy
acc = clf.score(X_test, y_test)
print(acc)

# Write metrics to file
with open('metrics.json', 'w') as outfile:
    outfile.write(json.dumps({"Accuracy": acc}))

# Plot confusion matrix
disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, cmap=plt.cm.Blues)
plt.savefig('confusion_matrix.png')

# Save the model
joblib.dump(clf, "model.joblib")