import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus as dotp
from graphviz import Source
import numpy as np


def experement():
    data = np.random.randint(low=-2,high=3,size=10)
    df = pd.DataFrame(data=data,columns=['value'])
    df['value'] = df.value.apply(lambda i: f"{i} < 0: {i}" if i < 0 else f"{i}>=0: 1")
    print(df)


def get_write() -> tuple:
    top_layer = pd.read_excel(io='ML_NIR_DATE_ASK.xlsx',
                              # engine='openpyxl',
                              sheet_name='Пропуски_занятий',
                              usecols=[0,1,3,5,6,7,10,11,12,13,17]
                              )
    table_personal_X = top_layer.loc[:,"Причина пропуска":"Домашние животные"]
    answer_personal_Y = top_layer['Пропущено академических часов']
    print(table_personal_X.describe())
    mean_y = answer_personal_Y.mean(axis=0)
    top_layer['criterion'] = answer_personal_Y.apply(lambda i : 1 if i >= mean_y else 0)
    standart_answer_Z = top_layer['criterion']
    mean_x = table_personal_X.mean(axis=0)
    std_x= table_personal_X.std(axis=0)
    table_personal_X -= mean_x
    table_personal_X /= std_x
    return table_personal_X,standart_answer_Z


def mystery(x,y) -> DecisionTreeClassifier:
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
    print(f"x_train: {x_train}")
    clf = DecisionTreeClassifier(criterion="entropy",max_depth=10)
    clf = clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy on training set: {:.3f}".format(clf.score(x_train,y_train)))
    print("Accuracy on test set: {:0.3f}".format(clf.score(x_test,y_test)))
    print(f"type(clf): {clf}")
    print(y_pred)
    return clf


def createGraph(classificator,features) -> None:
    print(len(features))
    graph = Source(export_graphviz(classificator, out_file=None,
                            filled=True, rounded=True,
                            special_characters=True,feature_names=features.columns,
                            class_names=['0','1']
                                   ))
    png_bytes = graph.pipe(format='png')
    with open('classificationGraphModel.png', 'wb') as f:
        f.write(png_bytes)
    Image(png_bytes)
    print("Complete png_create")



if __name__ == '__main__':
    print(get_write())
    model = get_write()
    classification = mystery(model[0],model[1])
    print(createGraph(classification,model[0]))
