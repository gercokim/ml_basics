import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Reads data from student-mat
data = pd.read_csv("LinReg\student-mat.csv", sep=";")

# Gets rid of the data that we don't want / features
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# What we want to find / label
predict = "G3"

# returns a new dataframe without the label
X = np.array(data.drop([predict], 1))

# returns a new dataframe that is just the label
Y = np.array(data[predict])

best = 0
# finds best model within 30 tries
for _ in range(30):
    # splits data into 4 arrays 
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    # creating linear model
    linear = linear_model.LinearRegression()

    # finds best fit line using training data
    linear.fit(x_train, y_train)

    # calculates accuracy of linear model on test data
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        # saves the model 
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")

# loads pickle file
linear = pickle.load(pickle_in)

# weights of input features
print("Co:", linear.coef_)

# value of bias 
print("Intercept:", linear.intercept_)

# calculates predictions of label given test data
predictions = linear.predict(x_test)

# loops through each index of predictions array
for x in range(len(predictions)):
    # prints out prediction, the data that was used to predict, actual result
    print(predictions[x], x_test[x], y_test[x])

# plots data
p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data['G3'])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()