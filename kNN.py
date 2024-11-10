import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

path = 'wdbc.data.mb.csv'
data = pd.read_csv(path, header=None)

# Separate features and labels
X = data.iloc[:, :-1].values  # Selecting all rows/columns except the last one (the actual data)
y = data.iloc[:, -1].values  # Selecting all rows for only the last column (the labels)

# Apply min-max normalization to the dataset
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Splitting the data set into 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=234)

# Visualizing the data
'''
plt.figure()
cmap = ListedColormap(['#FF0000', '#0000FF'])
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolors='k', s=20)
plt.show()
'''



# Method for calculating Euclidean distance
def euclidean_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))


# kNN Classifier
class kNN:
    def __init__(self, k):
        self.k = k
        self.point = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):

        predictions = [self._predict(x) for x in X]
        return predictions

    # Helper function for predict()
    def _predict(self, x):

        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get indices and labels of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]

        # Get the most common label of the nearest neighbors and return that value
        most_common = Counter(k_labels).most_common()
        return int(most_common[0][0])


# Run each test case and print out the accuracy and confusion matrix
for k in [1, 3, 5, 7, 9]:
    print('------------ k=' + str(k) + ' ------------')
    clf = kNN(k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    accuracy = np.sum(predictions == y_test) / len(y_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
