import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv('abalone.csv')
myframe = pd.DataFrame(dataset)
myframe['Sex'] = myframe['Sex'].map({'M':0, 'F':1, 'I':2})

y , X = myframe['Rings'] , myframe.drop(['Rings'], axis=1)

def age_group(rings):
    if rings <= 8:
        return "young"
    elif rings <= 11:
        return "adult"
    else:
        return "old"
y = y.apply(age_group)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

class modeling:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.report = None
    def training(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    def result(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        self.report = classification_report(y_test, y_pred)
        return self.report

class RandomForestModel(modeling):
    def __init__(self):
        super().__init__("Random Forest")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
class SVMModel(modeling):
    def __init__(self):
        super().__init__("SVM")
        self.model = SVC(kernel="rbf")
class KNNModel(modeling):
    def __init__(self):
        super().__init__("KNN")
        self.model = KNeighborsClassifier(n_neighbors=5)

def run_all_models(X_train, X_test, y_train, y_test):
    models = [RandomForestModel(), SVMModel(), KNNModel()]
    results = {}
    for m in models:
        m.training(X_train, y_train)
        report = m.result(X_test, y_test)
        results[m.model_name] = report
    return results

final_results = run_all_models(X_train, X_test, y_train, y_test)
for name, report in final_results.items():
    print(f"\n Classification Report for {name}:\n")
    print(report)
    print("***************************************************************")