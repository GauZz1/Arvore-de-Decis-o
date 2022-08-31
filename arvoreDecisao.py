import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def readData(arqName, delimiterUse):
    global data
    data = pd.read_csv(arqName, delimiter=delimiterUse) # colect data form database
    
def createTree(parameter, columsToDrop) :
    X = data.drop(columsToDrop, axis=1); # remove 'Chuva' and 'Exemplo' from 'data' and atributes to X 
    Y = data[parameter]; # create a new table with 'Chuva' atributte where Sim = 1 e Não = 0

    xtr, Xval, ytr, yval = train_test_split(X, Y, test_size=0.5, random_state=0);

    ohe = preprocessing.OneHotEncoder(sparse=False) # define OnHotEncoder



    encoded_x = ohe.fit_transform(X) # transform table X to binary
    encoded_y = ohe.fit_transform(Y) # transform table X to binary

    arvore = DecisionTreeClassifier()
    arvore = arvore.fit(encoded_x, encoded_y) # create tree

    p = arvore.predict(encoded_x)

    np.sqrt(confusion_matrix(p, encoded_y))

    return arvore