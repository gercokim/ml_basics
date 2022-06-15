import sklearn
from sklearn import datasets, svm

# sklearn dataset
cancer = datasets.load_breast_cancer()

# features and targets
print(cancer.feature_names)
print(cancer.target_names)
print("---------------------------")

X = cancer.data
Y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

print(x_train, y_train)

# for making results more readable, i.e. 0, 1 to malignant, benign
classes = ['malignant', 'benign']