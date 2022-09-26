import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#x,y,h,s,v
def svr_learning(version,xys):
    # getting the data
    dataset = pd.read_csv('ml_data/imgv' + str(version) + '_data.csv')
    # XH = dataset.iloc[:, :-3].values
    # yH = dataset.iloc[:, -3].values
    # dataset.drop('H', axis=1, inplace=True)
    # XS = dataset.iloc[:, :-2].values
    # yS = dataset.iloc[:, -2].values
    # dataset.drop('S', axis=1, inplace=True)
    XV = dataset.iloc[:, :-1].values
    yV = dataset.iloc[:, -1].values

    # yH = yH.reshape(len(yH),1)
    # yS = yS.reshape(len(yS),1)
    yV = yV.reshape(len(yV),1)

    # sc_XH = StandardScaler()
    # sc_yH = StandardScaler()
    # XH = sc_XH.fit_transform(XH)
    # yH = sc_yH.fit_transform(yH)
    # sc_XS = StandardScaler()
    # sc_yS = StandardScaler()
    # XS = sc_XS.fit_transform(XS)
    # yS = sc_yS.fit_transform(yS)
    sc_XV = StandardScaler()
    sc_yV = StandardScaler()
    XV = sc_XV.fit_transform(XV)
    yV = sc_yV.fit_transform(yV)

    X_train, X_test, y_train, y_test = train_test_split(XV, yV, test_size=0.2, shuffle= True)
    regressorX = SVR(kernel = 'rbf')
    regressorX.fit(X_train, y_train)
    print('Score: ', regressorX.score(X_test, y_test))

    # regressorH = SVR(kernel = 'rbf')
    # regressorH.fit(XH, yH)
    # regressorS = SVR(kernel = 'rbf')
    # regressorS.fit(XS, yS)
    regressorV = SVR(kernel = 'rbf')
    regressorV.fit(XV, yV)
    
    # predict_theseH = []
    # predict_theseS = []
    predict_theseV = []
    # for i in range(xys):
        # predict_theseH.append(i)
        # predict_theseS.append(i)
            # predict_theseV.append(i)
    
    # predictionsH = []
    # predictionsS = []
    predictionsV = []

    # predictionsH = regressorH.predict(predict_theseH)
    # predictionsS = regressorS.predict(predict_theseS)
    predictionsV = regressorV.predict(xys)

    # predictionsH = predictionsH.reshape(len(predictionsH),1)
    # predictionsS = predictionsS.reshape(len(predictionsS),1)
    predictionsV = predictionsV.reshape(len(predictionsV),1)

    # predictions_inverseH = sc_yH.inverse_transform(predictionsH)
    # predictions_inverseS = sc_yS.inverse_transform(predictionsS)
    predictions_inverseV = sc_yV.inverse_transform(predictionsV)


    return predictions_inverseV #predictions_inverseH, predictions_inverseS, predictions_inverseV
    