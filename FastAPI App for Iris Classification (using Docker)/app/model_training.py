from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd

import pickle

def preprocessing_fn(data):
    
    X = data.drop(["Id", "Species"], axis=1).values
    y = data["Species"].values

    lbl_clf = LabelEncoder()
    Y_encoded = lbl_clf.fit_transform(y)

    seed = 2

    x_train, x_test, y_train, y_test = train_test_split(X, Y_encoded, test_size=0.25, random_state=seed, shuffle=True)

    return x_train, x_test, y_train, y_test


def create_model(x_train, y_train):
    
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    return clf

def save_model(model):
    pickle.dump(model, open('model.pkl', 'wb'))

    # with open("model.pkl", "rb") as file:
    #     aa = pickle.load(file)
        
    # pred = aa.predict([[1,2,1,2]])
    # print(pred)

    

if __name__ == "__main__":
    
    data = pd.read_csv("iris.csv")

    x_train, x_test, y_train, y_test = preprocessing_fn(data = data)

    model = create_model(x_train = x_train, y_train = y_train)

    save_model(model = model)
