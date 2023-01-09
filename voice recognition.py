# Importing Important files :-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing file:-
df = pd.read_csv('voice.csv')
print(df.head())

# showing Size of data:-
print("\n")
print("Shape of the data is:")
print(df.shape)
print(" ")

#checking Null Values:-
print(df.isnull().sum())

#checking Dublicate Valuews :-
df.drop_duplicates(inplace=True)

# As label is in 'object' class, we change it to int:-
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
df['label'] = lb.fit_transform(df['label'])

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(" ")
print(type(x))
print(type(y))
print(" ")

# pie chart of the input outcomes:-

labels = ["male", "female"]
z = np.array([1548, 1548])
plt.pie(z, labels=labels)
plt.show()

#dividing our data into Training data and Test data:-
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
print(f"Training data X: {x_train.shape}")
print(f"Testing data X: {x_test.shape}")
print(f"Training data Y: {y_train.shape}")
print(f"Testing data Y: {y_test.shape}")

# A function to tell model's accuracy and Classification report
from sklearn.metrics import confusion_matrix, classification_report


def apply_model(model, x_train, x_test, y_train, y_test, Name):
    model.fit(x_train, y_train)
    ypred = model.predict(x_test)
    print(" ")
    print("\n\n------------------------------------------------------------")
    print(f"{Name} Model Results:")
    print("\n------------------------------------------------------------")
    print(" ")
    print(f'Training Score {model.score(x_train, y_train)}')
    print(f'Testing Score {model.score(x_test, y_test)}')
    cm = confusion_matrix(y_test, ypred)
    print(f'Confusion_matrix {cm}\n')
    print(f'Classification_report {classification_report(y_test, ypred)}\n', )
    return (model.score(x_test, y_test))


"""
# we need to apply all these models:
#   a. Decision Tree Classifier
#   b. Random Forest Classifier
#   c. KNN Classifier
#   d. Logistic Regression
#   e. SVM Classifier

#   We will be using Hyperparameter tunning (RandomizedSearchCv) to get the best parameters for each model to get the best Individual results.
"""

from sklearn.model_selection import RandomizedSearchCV

test_values = []

# A: Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
params_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 12, 14, 16, 18, 25, 30],
    'min_samples_split': [10, 12, 14, 16, 18, 20, 22, 24]
}

rscv1 = RandomizedSearchCV(dt, param_distributions=params_dt)
r1 = apply_model(rscv1, x_train, x_test, y_train, y_test,
                 "DecisionTreeClassifier")
test_values.append(r1)

# B: Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
params_rf = {
    'n_estimators': [30, 40, 50, 60, 70, 80, 90, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 13, 15, 17, 18],
    'min_samples_split': [12, 14, 16, 28, 20, 22, 24]
}
rscv2 = RandomizedSearchCV(rf, param_distributions=params_rf)
r2 = apply_model(rscv2, x_train, x_test, y_train, y_test,
                 "RandomForestClassifier")
test_values.append(r2)

# C: KNNClassifier

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
params_knn = {'n_neighbors': list(range(10, 80, 5))}
rscv3 = RandomizedSearchCV(knn, param_distributions=params_knn)
r3 = apply_model(rscv3, x_train, x_test, y_train, y_test,
                 "KNeighborsClassifier")
test_values.append(r3)

# D: Logistic Regression

from sklearn.linear_model import LogisticRegression

rscv4 = LogisticRegression(solver='liblinear')
r4 = apply_model(rscv4, x_train, x_test, y_train, y_test, "LogisticRegression")
test_values.append(r4)

# E: SVM

from sklearn.svm import SVC

svm = SVC(kernel="linear", C=15)
r5 = apply_model(svm, x_train, x_test, y_train, y_test, "SVM Classifier")
test_values.append(r5)

# checking the best model with best accuracy:-
test_keys = [
    "Decision Tree Classifier", " Random Forest Classifier", "KNN Classifier",
    "Logistic Regression", "SVM Classifier"
]

res = dict(zip(test_keys, test_values))

Keymax = max(res, key=lambda x: res[x])
print("\n\n------------------------------------------------------------")
print(f"\nbest accuracy found is {max(test_values)} of model {Keymax}")
print("\n------------------------------------------------------------")