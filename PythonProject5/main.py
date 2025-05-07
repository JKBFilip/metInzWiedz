import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))

def cosine_distance(x, y):
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def knn_predict(X_train, y_train, X_test, k=3, distance_func=euclidean_distance):
    predictions = []
    for x_test in X_test:
        distances = [distance_func(x_test, x_train) for x_train in X_train]
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = [y_train[i] for i in nearest_indices]
        predicted_label = max(set(nearest_labels), key=nearest_labels.count)
        predictions.append(predicted_label)
    return predictions

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

X_train_norm = normalize(X_train)
X_test_norm = normalize(X_test)

for name, func, xtest, xtrain in [
    ("Euklidesowa", euclidean_distance, X_test, X_train),
    ("Manhattan", manhattan_distance, X_test, X_train),
    ("Cosinusowa", cosine_distance, X_test_norm, X_train_norm)
]:
    y_pred = knn_predict(xtrain, y_train, xtest, k=3, distance_func=func)
    acc = accuracy_score(y_test, y_pred)
    print(f"Dokładność dla metryki {name}: {acc:.2%}")
