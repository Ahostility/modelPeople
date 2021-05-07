import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from IPython.display import Image
from graphviz import Source
import numpy as np
import joblib
import sys


def checkDataCSV(file_csv) -> tuple:
    top_layer = pd.read_excel(io=file_csv,
                              sheet_name='Пропуски_занятий',
                              usecols=[0, 1, 3, 5, 6, 7, 10, 11, 12, 13, 17]
                              )
    table_personal_X = top_layer.loc[:, "Причина пропуска":"Домашние животные"]
    answer_personal_Y = top_layer['Пропущено академических часов']
    print(table_personal_X.describe())
    mean_y = answer_personal_Y.mean(axis=0)
    top_layer['criterion'] = answer_personal_Y.apply(lambda i: 1 if i >= mean_y else 0)
    standart_answer_Z = top_layer['criterion']
    mean_x = table_personal_X.mean(axis=0)
    std_x = table_personal_X.std(axis=0)
    table_personal_X -= mean_x
    table_personal_X /= std_x
    return table_personal_X, standart_answer_Z


def learnModel(x, y):
    print(len(x))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=10)
    clf = clf.fit(x_train, y_train)  # Класификатор дерева решений из полученной выборки

    print("Accuracy on training set: {:.3f}".format(clf.score(x_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(clf.score(x_test, y_test)))
    save_model(clf)

    return x_test


def prediction(csv_test,csv) -> DecisionTreeClassifier:
    choice = input("Enter choice for Data: ")
    if choice == '1':
        # for test use
        print(f"Model is complete, we choice 1")
        model = load_model()
        x_test, y_test = csv_test[0],csv_test[1]
        print("Accuracy on test set: {:.3f}".format(model.score(x_test,y_test)))
    else:
        #for learn model
        print(f"To Learn model, we choice 2")
        x_test = learnModel(csv[0], csv[1])
        model = load_model()

    y_pred = model.predict(x_test)
    print(f"Probability: {model.predict_proba(x_test)}")
    index_test_X = list(x_test.loc[:, 'Причина пропуска'].index)
    print(f"Answer: {y_pred}")
    predictionAnswer(y_pred, index_test_X)
    treeGraph(model,x_test)
    return model


def treeGraph(classificator, features) -> None:
    print(len(features))
    graph = Source(export_graphviz(classificator, out_file=None,
                                   filled=True, rounded=True,
                                   special_characters=True, feature_names=features.columns,
                                   class_names=['0', '1']
                                   ))
    png_bytes = graph.pipe(format='png')
    with open('classificationGraphModel.png', 'wb') as f:
        f.write(png_bytes)
    Image(png_bytes)
    print("Complete png_create")


def predictionAnswer(pred, test: list) -> None:
    answer_pil = pd.Series(list(pred), index=test)
    answer = pd.DataFrame({'Answer': answer_pil})
    print(answer)


def save_model(classificator) -> None:
    try:
        joblib.dump(classificator,"model.pkl")
        print("Complete save model!")
    except:
        print("error")


def load_model() -> DecisionTreeClassifier:
    try:
        clf = joblib.load("model.pkl")
        print("Open model!")
        return clf
    except:
        print("Ooops, not Model!")
        sys.exit()


if __name__ == '__main__':
    test = checkDataCSV('ML_TEST.xlsx')
    data = checkDataCSV('ML_NIR_DATE_ASK.xlsx')
    prediction(test,data)
