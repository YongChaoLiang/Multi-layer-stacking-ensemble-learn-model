import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
def read_data():
    df=pd.read_excel(r"D:\桌面文件\研一上\学习笔记\机器学习\jupyter notebook save path\data\Dmax1531.xlsx")
    x=df.iloc[:,:3]
    y=df.iloc[:,-1:].values.ravel()
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=2024)
    return x_train, x_test, y_train, y_test
def single_models():
    #Linear model
    LR=LinearRegression()
    model_train(LR,'LinearRegression')
    ridge=Ridge(random_state=2024)
    model_train(ridge,'ridge')
    lasso=Lasso()
    model_train(lasso,'lasso')
    elasticNet = ElasticNet(random_state=2024)
    model_train(elasticNet,'elasticNet')
    bayesianRidge=BayesianRidge()
    model_train(bayesianRidge,'bayesianRidge')
    #other
    svr=SVR()
    model_train(svr,'svr')
    KNN= KNeighborsRegressor()
    model_train(KNN,'KNN')
    LGBM=lgb.LGBMRegressor(random_state=2024)
    model_train(LGBM,'Lightgbm')
    XGB = XGBRegressor()
    model_train(XGB,'XGBoost')
    DT=DecisionTreeRegressor(random_state=2024)
    model_train(DT,'DT')
    ET=ExtraTreesRegressor(random_state=2024)
    model_train(ET,'ET')
    GBDT=GradientBoostingRegressor(random_state=2024)
    model_train(GBDT,'GradientBoosting')
    RF = RandomForestRegressor(random_state=2024)
    model_train(RF,'RandomForest')
#Model evaluation
def correlation_coefficient(y_true, y_pred):
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    return correlation
def model_train(model, str):
    print('--------------' +str+ '---------------------------------------')
    x_train, x_test, y_train, y_test=read_data()
    model.fit(x_train, y_train)
    print('-----------Train result--------------------------------------------')
    train_pred = model.predict(x_train)
    train_R=correlation_coefficient(y_train,train_pred)
    train_MSE = mean_squared_error(train_pred, y_train)
    train_MAE = mean_absolute_error(train_pred, y_train)
    print("train_R:",train_R)
    print("train_MSE:",train_MSE)
    print("train_MAE:",train_MAE)
    print('-----------Test result---------------------------------------------')
    test_pred=model.predict(x_test)
    test_R=correlation_coefficient(y_test,test_pred)
    test_mse = mean_squared_error(y_test,test_pred,)
    test_mae = mean_absolute_error(y_test,test_pred)
    print("test_R:",test_R)
    print("test_MSE:",test_mse)
    print("test_MAE:",test_mae)
    return
single_models()