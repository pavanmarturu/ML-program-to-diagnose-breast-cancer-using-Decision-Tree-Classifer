import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import datasets

# importing breast_cancer_dataset
data = datasets.load_breast_cancer()

# Spliting the given data into training sets and test sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

#decision tree classifier and the hyperparameter space
clf = DecisionTreeClassifier()
param_dist = {
    "max_depth": [3, None],
    "max_features": np.arange(1, 11),
    "min_samples_leaf": np.arange(1, 11),
    "criterion": ["gini", "entropy"]
}

# Performing a random search for hyperparameter tuning
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=10, cv=5, random_state=42)
random_search.fit(X_train, y_train)
best_clf = random_search.best_estimator_

# Evaluating the performance of the classifier using k-fold cross-validation
scores = cross_val_score(best_clf, X_train, y_train, cv=5)
print("Cross-validation scores:", scores)
print("Mean cross-validation score:", np.mean(scores))

#predictions based on the test data and evaluate the performance of the classifier
y_pred = best_clf.predict(X_test)
print("Accuracy of test data:", accuracy_score(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("Specificity of test data:", specificity)
print("Sensitivity (precision) of test data:", sensitivity)
print(classification_report(y_test, y_pred))

