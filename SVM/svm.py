import sklearn
from sklearn import datasets, svm, metrics
from sklearn.neighbors import KNeighborsClassifier


# sklearn dataset
cancer = datasets.load_breast_cancer()

# features and targets
print(cancer.feature_names)
print(cancer.target_names)

X = cancer.data
Y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

print(x_train, y_train)

# for making results more readable, i.e. 0, 1 to malignant, benign
classes = ['malignant', 'benign']

# creates classifier (support vector classifier) with a linear kernel and an increased soft margin
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)

# calculates predictions from test data
y_pred = clf.predict(x_test)

# calculates accuracy of model
acc = metrics.accuracy_score(y_test, y_pred)

print(acc)