import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,cross_validate,train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor
from mlxtend.regressor import StackingCVRegressor
def correlation_coefficient(y_true, y_pred):
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    return correlation
df=pd.read_excel(r"D:\桌面文件\研一上\学习笔记\机器学习\jupyter notebook save path\data\smogn.xlsx")
x=df.iloc[:,:3]
y=df.iloc[:,-1:]
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))
x=x_scaler.fit_transform(x)
y=y_scaler.fit_transform(y)
y=y.ravel()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2024)
LR=LinearRegression()
svr=SVR()
KNN=KNeighborsRegressor(weights='distance'
                        ,algorithm='ball_tree'
                        ,leaf_size=23
                        ,n_neighbors=4
                        ,p=7,n_jobs=-1)
LGBM=lgb.LGBMRegressor(learning_rate= 0.3574102384700801
                       ,max_depth=7
                       ,min_child_samples=2
                       ,min_child_weight=0.11805399313801943
                       ,n_estimators=213
                       ,num_leaves=29
                       ,reg_lambda=0.5671639569326817
                       ,subsample=0.2237073465735047
                       ,random_state=2024)
XGB=XGBRegressor(learning_rate=0.5219244284474196
                 ,n_estimators=112
                 ,random_state=2024
                 ,n_jobs=-1 )
ET=ExtraTreesRegressor(max_depth=16
                       ,max_features=0.690224732325302
                       ,n_estimators=573
                       ,random_state=2024)
GBDT=GradientBoostingRegressor(learning_rate=0.20001677228988277
                               ,max_depth=8
                               ,min_samples_leaf=5
                               ,n_estimators=522
                               ,random_state=2024)
RF=RandomForestRegressor(max_depth=18
                         ,max_features=0.8795114223458718
                         ,min_samples_split=2
                         ,n_estimators=31
                         ,random_state=2024)
#1
stack_gen21=StackingCVRegressor(regressors=(KNN,LGBM,RF,KNN,GBDT)
                                                ,meta_regressor=svr
                                                ,shuffle=True
                                                ,use_features_in_secondary=True,n_jobs=-1
                                                ,random_state=2024)
stack_gen22=StackingCVRegressor(regressors=(XGB,ET,ET,LGBM,RF)
                                            ,meta_regressor=KNN
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen23=StackingCVRegressor(regressors=(svr,RF,KNN,KNN,LGBM)
                                            ,meta_regressor=RF
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen3=StackingCVRegressor(regressors=(stack_gen21,stack_gen22,stack_gen23)
                                       ,meta_regressor=LR
                                       ,shuffle=True
                                       ,use_features_in_secondary=True
                                       ,n_jobs=-1,random_state=2024).fit(x_train,y_train)
train_pred = stack_gen3.predict(x_train)
train_R=correlation_coefficient(y_train,train_pred)
train_MSE=mean_squared_error(train_pred,y_train)
train_MAE=mean_absolute_error(train_pred,y_train)
test_pred=stack_gen3.predict(x_test)
test_R=correlation_coefficient(y_test,test_pred)
test_mse=mean_squared_error(y_test,test_pred,)
test_mae=mean_absolute_error(y_test,test_pred)
print('-----------Train result--------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(train_R,train_MSE,train_MAE))
print('-----------Test result---------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(test_R,test_mse,test_mae))
#2
stack_gen21=StackingCVRegressor(regressors=(KNN,LGBM,RF,KNN,GBDT)
                                                ,meta_regressor=svr
                                                ,shuffle=True
                                                ,use_features_in_secondary=True,n_jobs=-1
                                                ,random_state=2024)
stack_gen22=StackingCVRegressor(regressors=(XGB,ET,ET,LGBM,RF)
                                            ,meta_regressor=KNN
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen23=StackingCVRegressor(regressors=(svr,RF,LGBM,svr,RF )
                                            ,meta_regressor=ET
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen3=StackingCVRegressor(regressors=(stack_gen21,stack_gen22,stack_gen23)
                                       ,meta_regressor=LR
                                       ,shuffle=True
                                       ,use_features_in_secondary=True
                                       ,n_jobs=-1,random_state=2024).fit(x_train,y_train)
train_pred = stack_gen3.predict(x_train)
train_R=correlation_coefficient(y_train,train_pred)
train_MSE=mean_squared_error(train_pred,y_train)
train_MAE=mean_absolute_error(train_pred,y_train)
test_pred=stack_gen3.predict(x_test)
test_R=correlation_coefficient(y_test,test_pred)
test_mse=mean_squared_error(y_test,test_pred,)
test_mae=mean_absolute_error(y_test,test_pred)
print('-----------Train result--------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(train_R,train_MSE,train_MAE))
print('-----------Test result---------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(test_R,test_mse,test_mae))
#3
stack_gen21=StackingCVRegressor(regressors=(KNN,LGBM,RF,KNN,GBDT)
                                                ,meta_regressor=svr
                                                ,shuffle=True
                                                ,use_features_in_secondary=True,n_jobs=-1
                                                ,random_state=2024)
stack_gen22=StackingCVRegressor(regressors=(XGB,ET,ET,LGBM,RF)
                                            ,meta_regressor=KNN
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen23=StackingCVRegressor(regressors=(ET,RF,LGBM,KNN,svr )
                                            ,meta_regressor=GBDT
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen3=StackingCVRegressor(regressors=(stack_gen21,stack_gen22,stack_gen23)
                                       ,meta_regressor=LR
                                       ,shuffle=True
                                       ,use_features_in_secondary=True
                                       ,n_jobs=-1,random_state=2024).fit(x_train,y_train)
train_pred = stack_gen3.predict(x_train)
train_R=correlation_coefficient(y_train,train_pred)
train_MSE=mean_squared_error(train_pred,y_train)
train_MAE=mean_absolute_error(train_pred,y_train)
test_pred=stack_gen3.predict(x_test)
test_R=correlation_coefficient(y_test,test_pred)
test_mse=mean_squared_error(y_test,test_pred,)
test_mae=mean_absolute_error(y_test,test_pred)
print('-----------Train result--------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(train_R,train_MSE,train_MAE))
print('-----------Test result---------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(test_R,test_mse,test_mae))
#4
stack_gen21=StackingCVRegressor(regressors=(KNN,LGBM,RF,KNN,GBDT)
                                                ,meta_regressor=svr
                                                ,shuffle=True
                                                ,use_features_in_secondary=True,n_jobs=-1
                                                ,random_state=2024)
stack_gen22=StackingCVRegressor(regressors=(XGB,ET,ET,LGBM,RF)
                                            ,meta_regressor=KNN
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen23=StackingCVRegressor(regressors=(KNN,KNN,LGBM,LGBM,RF )
                                            ,meta_regressor=LGBM
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen3=StackingCVRegressor(regressors=(stack_gen21,stack_gen22,stack_gen23)
                                       ,meta_regressor=LR
                                       ,shuffle=True
                                       ,use_features_in_secondary=True
                                       ,n_jobs=-1,random_state=2024).fit(x_train,y_train)
train_pred = stack_gen3.predict(x_train)
train_R=correlation_coefficient(y_train,train_pred)
train_MSE=mean_squared_error(train_pred,y_train)
train_MAE=mean_absolute_error(train_pred,y_train)
test_pred=stack_gen3.predict(x_test)
test_R=correlation_coefficient(y_test,test_pred)
test_mse=mean_squared_error(y_test,test_pred,)
test_mae=mean_absolute_error(y_test,test_pred)
print('-----------Train result--------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(train_R,train_MSE,train_MAE))
print('-----------Test result---------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(test_R,test_mse,test_mae))
#5
stack_gen21=StackingCVRegressor(regressors=(KNN,LGBM,RF,KNN,GBDT)
                                                ,meta_regressor=svr
                                                ,shuffle=True
                                                ,use_features_in_secondary=True,n_jobs=-1
                                                ,random_state=2024)
stack_gen22=StackingCVRegressor(regressors=(XGB,ET,ET,LGBM,RF)
                                            ,meta_regressor=KNN
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen23=StackingCVRegressor(regressors=(KNN, LGBM, svr, RF, KNN )
                                            ,meta_regressor=XGB
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen3=StackingCVRegressor(regressors=(stack_gen21,stack_gen22,stack_gen23)
                                       ,meta_regressor=LR
                                       ,shuffle=True
                                       ,use_features_in_secondary=True
                                       ,n_jobs=-1,random_state=2024).fit(x_train,y_train)
train_pred = stack_gen3.predict(x_train)
train_R=correlation_coefficient(y_train,train_pred)
train_MSE=mean_squared_error(train_pred,y_train)
train_MAE=mean_absolute_error(train_pred,y_train)
test_pred=stack_gen3.predict(x_test)
test_R=correlation_coefficient(y_test,test_pred)
test_mse=mean_squared_error(y_test,test_pred,)
test_mae=mean_absolute_error(y_test,test_pred)
print('-----------Train result--------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(train_R,train_MSE,train_MAE))
print('-----------Test result---------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(test_R,test_mse,test_mae))
#6
stack_gen21=StackingCVRegressor(regressors=(XGB,ET,ET,LGBM,RF)
                                                ,meta_regressor=KNN
                                                ,shuffle=True
                                                ,use_features_in_secondary=True,n_jobs=-1
                                                ,random_state=2024)
stack_gen22=StackingCVRegressor(regressors=(svr,RF,KNN,KNN,LGBM)
                                            ,meta_regressor=RF
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen23=StackingCVRegressor(regressors=(svr,RF,LGBM,svr,RF)
                                            ,meta_regressor=ET
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen3=StackingCVRegressor(regressors=(stack_gen21,stack_gen22,stack_gen23)
                                       ,meta_regressor=LR
                                       ,shuffle=True
                                       ,use_features_in_secondary=True
                                       ,n_jobs=-1,random_state=2024).fit(x_train,y_train)
train_pred = stack_gen3.predict(x_train)
train_R=correlation_coefficient(y_train,train_pred)
train_MSE=mean_squared_error(train_pred,y_train)
train_MAE=mean_absolute_error(train_pred,y_train)
test_pred=stack_gen3.predict(x_test)
test_R=correlation_coefficient(y_test,test_pred)
test_mse=mean_squared_error(y_test,test_pred,)
test_mae=mean_absolute_error(y_test,test_pred)
print('-----------Train result--------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(train_R,train_MSE,train_MAE))
print('-----------Test result---------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(test_R,test_mse,test_mae))
#7
stack_gen21=StackingCVRegressor(regressors=(XGB,ET,ET,LGBM,RF)
                                                ,meta_regressor=KNN
                                                ,shuffle=True
                                                ,use_features_in_secondary=True,n_jobs=-1
                                                ,random_state=2024)
stack_gen22=StackingCVRegressor(regressors=(svr,RF,KNN,KNN,LGBM)
                                            ,meta_regressor=RF
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen23=StackingCVRegressor(regressors=(ET,RF,LGBM,KNN,svr)
                                            ,meta_regressor=GBDT
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen3=StackingCVRegressor(regressors=(stack_gen21,stack_gen22,stack_gen23)
                                       ,meta_regressor=LR
                                       ,shuffle=True
                                       ,use_features_in_secondary=True
                                       ,n_jobs=-1,random_state=2024).fit(x_train,y_train)
train_pred = stack_gen3.predict(x_train)
train_R=correlation_coefficient(y_train,train_pred)
train_MSE=mean_squared_error(train_pred,y_train)
train_MAE=mean_absolute_error(train_pred,y_train)
test_pred=stack_gen3.predict(x_test)
test_R=correlation_coefficient(y_test,test_pred)
test_mse=mean_squared_error(y_test,test_pred,)
test_mae=mean_absolute_error(y_test,test_pred)
print('-----------Train result--------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(train_R,train_MSE,train_MAE))
print('-----------Test result---------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(test_R,test_mse,test_mae))
#8
stack_gen21=StackingCVRegressor(regressors=(XGB,ET,ET,LGBM,RF)
                                                ,meta_regressor=KNN
                                                ,shuffle=True
                                                ,use_features_in_secondary=True,n_jobs=-1
                                                ,random_state=2024)
stack_gen22=StackingCVRegressor(regressors=(svr,RF,KNN,KNN,LGBM)
                                            ,meta_regressor=RF
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen23=StackingCVRegressor(regressors=(KNN,KNN,LGBM,LGBM,RF)
                                            ,meta_regressor=LGBM
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen3=StackingCVRegressor(regressors=(stack_gen21,stack_gen22,stack_gen23)
                                       ,meta_regressor=LR
                                       ,shuffle=True
                                       ,use_features_in_secondary=True
                                       ,n_jobs=-1,random_state=2024).fit(x_train,y_train)
train_pred = stack_gen3.predict(x_train)
train_R=correlation_coefficient(y_train,train_pred)
train_MSE=mean_squared_error(train_pred,y_train)
train_MAE=mean_absolute_error(train_pred,y_train)
test_pred=stack_gen3.predict(x_test)
test_R=correlation_coefficient(y_test,test_pred)
test_mse=mean_squared_error(y_test,test_pred,)
test_mae=mean_absolute_error(y_test,test_pred)
print('-----------Train result--------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(train_R,train_MSE,train_MAE))
print('-----------Test result---------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(test_R,test_mse,test_mae))
#9
stack_gen21=StackingCVRegressor(regressors=(XGB,ET,ET,LGBM,RF)
                                                ,meta_regressor=KNN
                                                ,shuffle=True
                                                ,use_features_in_secondary=True,n_jobs=-1
                                                ,random_state=2024)
stack_gen22=StackingCVRegressor(regressors=(svr,RF,KNN,KNN,LGBM)
                                            ,meta_regressor=RF
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen23=StackingCVRegressor(regressors=(KNN,LGBM,svr,RF,KNN)
                                            ,meta_regressor=XGB
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen3=StackingCVRegressor(regressors=(stack_gen21,stack_gen22,stack_gen23)
                                       ,meta_regressor=LR
                                       ,shuffle=True
                                       ,use_features_in_secondary=True
                                       ,n_jobs=-1,random_state=2024).fit(x_train,y_train)
train_pred = stack_gen3.predict(x_train)
train_R=correlation_coefficient(y_train,train_pred)
train_MSE=mean_squared_error(train_pred,y_train)
train_MAE=mean_absolute_error(train_pred,y_train)
test_pred=stack_gen3.predict(x_test)
test_R=correlation_coefficient(y_test,test_pred)
test_mse=mean_squared_error(y_test,test_pred,)
test_mae=mean_absolute_error(y_test,test_pred)
print('-----------Train result--------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(train_R,train_MSE,train_MAE))
print('-----------Test result---------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(test_R,test_mse,test_mae))
#10
stack_gen21=StackingCVRegressor(regressors=(svr, RF, KNN, KNN,LGBM)
                                                ,meta_regressor=RF
                                                ,shuffle=True
                                                ,use_features_in_secondary=True,n_jobs=-1
                                                ,random_state=2024)
stack_gen22=StackingCVRegressor(regressors=(svr,RF,LGBM,svr,RF )
                                            ,meta_regressor=ET
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen23=StackingCVRegressor(regressors=(ET,RF,LGBM,KNN,svr)
                                            ,meta_regressor=GBDT
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen3=StackingCVRegressor(regressors=(stack_gen21,stack_gen22,stack_gen23)
                                       ,meta_regressor=LR
                                       ,shuffle=True
                                       ,use_features_in_secondary=True
                                       ,n_jobs=-1,random_state=2024).fit(x_train,y_train)
train_pred = stack_gen3.predict(x_train)
train_R=correlation_coefficient(y_train,train_pred)
train_MSE=mean_squared_error(train_pred,y_train)
train_MAE=mean_absolute_error(train_pred,y_train)
test_pred=stack_gen3.predict(x_test)
test_R=correlation_coefficient(y_test,test_pred)
test_mse=mean_squared_error(y_test,test_pred,)
test_mae=mean_absolute_error(y_test,test_pred)
print('-----------Train result--------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(train_R,train_MSE,train_MAE))
print('-----------Test result---------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(test_R,test_mse,test_mae))
#11
stack_gen21=StackingCVRegressor(regressors=(svr,RF,KNN,KNN,LGBM)
                                                ,meta_regressor=RF
                                                ,shuffle=True
                                                ,use_features_in_secondary=True,n_jobs=-1
                                                ,random_state=2024)
stack_gen22=StackingCVRegressor(regressors=(svr,RF,LGBM,svr,RF )
                                            ,meta_regressor=ET
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen23=StackingCVRegressor(regressors=(KNN,KNN,LGBM,LGBM,RF)
                                            ,meta_regressor=LGBM
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen3=StackingCVRegressor(regressors=(stack_gen21,stack_gen22,stack_gen23)#knn,GBDT,ET
                                       ,meta_regressor=LR
                                       ,shuffle=True
                                       ,use_features_in_secondary=True
                                       ,n_jobs=-1,random_state=2024).fit(x_train,y_train)
train_pred = stack_gen3.predict(x_train)
train_R=correlation_coefficient(y_train,train_pred)
train_MSE=mean_squared_error(train_pred,y_train)
train_MAE=mean_absolute_error(train_pred,y_train)
test_pred=stack_gen3.predict(x_test)
test_R=correlation_coefficient(y_test,test_pred)
test_mse=mean_squared_error(y_test,test_pred,)
test_mae=mean_absolute_error(y_test,test_pred)
print('-----------Train result--------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(train_R,train_MSE,train_MAE))
print('-----------Test result---------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(test_R,test_mse,test_mae))
#12
stack_gen21=StackingCVRegressor(regressors=(svr,RF,KNN,KNN,LGBM)
                                                ,meta_regressor=RF
                                                ,shuffle=True
                                                ,use_features_in_secondary=True,n_jobs=-1
                                                ,random_state=2024)
stack_gen22=StackingCVRegressor(regressors=(svr,RF,LGBM,svr,RF)
                                            ,meta_regressor=ET
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen23=StackingCVRegressor(regressors=(KNN,LGBM,svr,RF,KNN)
                                            ,meta_regressor=XGB
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen3=StackingCVRegressor(regressors=(stack_gen21,stack_gen22,stack_gen23)
                                       ,meta_regressor=LR
                                       ,shuffle=True
                                       ,use_features_in_secondary=True
                                       ,n_jobs=-1,random_state=2024).fit(x_train,y_train)
train_pred = stack_gen3.predict(x_train)
train_R=correlation_coefficient(y_train,train_pred)
train_MSE=mean_squared_error(train_pred,y_train)
train_MAE=mean_absolute_error(train_pred,y_train)
test_pred=stack_gen3.predict(x_test)
test_R=correlation_coefficient(y_test,test_pred)
test_mse=mean_squared_error(y_test,test_pred,)
test_mae=mean_absolute_error(y_test,test_pred)
print('-----------Train result--------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(train_R,train_MSE,train_MAE))
print('-----------Test result---------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(test_R,test_mse,test_mae))
#13
stack_gen21=StackingCVRegressor(regressors=(svr,RF,LGBM,svr,RF)
                                                ,meta_regressor=ET
                                                ,shuffle=True
                                                ,use_features_in_secondary=True,n_jobs=-1
                                                ,random_state=2024)
stack_gen22=StackingCVRegressor(regressors=(ET,RF,LGBM,KNN,svr)
                                            ,meta_regressor=GBDT
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen23=StackingCVRegressor(regressors=(KNN,KNN,LGBM,LGBM,RF)
                                            ,meta_regressor=LGBM
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen3=StackingCVRegressor(regressors=(stack_gen21,stack_gen22,stack_gen23)
                                       ,meta_regressor=LR
                                       ,shuffle=True
                                       ,use_features_in_secondary=True
                                       ,n_jobs=-1,random_state=2024).fit(x_train,y_train)
train_pred = stack_gen3.predict(x_train)
train_R=correlation_coefficient(y_train,train_pred)
train_MSE=mean_squared_error(train_pred,y_train)
train_MAE=mean_absolute_error(train_pred,y_train)
test_pred=stack_gen3.predict(x_test)
test_R=correlation_coefficient(y_test,test_pred)
test_mse=mean_squared_error(y_test,test_pred,)
test_mae=mean_absolute_error(y_test,test_pred)
print('-----------Train result--------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(train_R,train_MSE,train_MAE))
print('-----------Test result---------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(test_R,test_mse,test_mae))
#14
stack_gen21=StackingCVRegressor(regressors=(svr,RF,LGBM,svr,RF)
                                                ,meta_regressor=ET
                                                ,shuffle=True
                                                ,use_features_in_secondary=True,n_jobs=-1
                                                ,random_state=2024)
stack_gen22=StackingCVRegressor(regressors=(ET, RF,LGBM,KNN,svr)
                                            ,meta_regressor=GBDT
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen23=StackingCVRegressor(regressors=(KNN, LGBM,svr,RF, KNN)
                                            ,meta_regressor=XGB
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen3=StackingCVRegressor(regressors=(stack_gen21,stack_gen22,stack_gen23)
                                       ,meta_regressor=LR
                                       ,shuffle=True
                                       ,use_features_in_secondary=True
                                       ,n_jobs=-1,random_state=2024).fit(x_train,y_train)
train_pred = stack_gen3.predict(x_train)
train_R=correlation_coefficient(y_train,train_pred)
train_MSE=mean_squared_error(train_pred,y_train)
train_MAE=mean_absolute_error(train_pred,y_train)
test_pred=stack_gen3.predict(x_test)
test_R=correlation_coefficient(y_test,test_pred)
test_mse=mean_squared_error(y_test,test_pred,)
test_mae=mean_absolute_error(y_test,test_pred)
print('-----------Train result--------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(train_R,train_MSE,train_MAE))
print('-----------Test result---------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(test_R,test_mse,test_mae))
#15
stack_gen21=StackingCVRegressor(regressors=(ET,RF,LGBM,KNN,svr)
                                                ,meta_regressor=GBDT
                                                ,shuffle=True
                                                ,use_features_in_secondary=True,n_jobs=-1
                                                ,random_state=2024)
stack_gen22=StackingCVRegressor(regressors=(KNN,KNN,LGBM,LGBM,RF)
                                            ,meta_regressor=LGBM
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen23=StackingCVRegressor(regressors=(KNN, LGBM,svr,RF,KNN)
                                            ,meta_regressor=XGB
                                            ,shuffle=True
                                            ,use_features_in_secondary=True,n_jobs=-1
                                            ,random_state=2024)
stack_gen3=StackingCVRegressor(regressors=(stack_gen21,stack_gen22,stack_gen23)
                                       ,meta_regressor=LR
                                       ,shuffle=True
                                       ,use_features_in_secondary=True
                                       ,n_jobs=-1,random_state=2024).fit(x_train,y_train)
train_pred = stack_gen3.predict(x_train)
train_R=correlation_coefficient(y_train,train_pred)
train_MSE=mean_squared_error(train_pred,y_train)
train_MAE=mean_absolute_error(train_pred,y_train)
test_pred=stack_gen3.predict(x_test)
test_R=correlation_coefficient(y_test,test_pred)
test_mse=mean_squared_error(y_test,test_pred,)
test_mae=mean_absolute_error(y_test,test_pred)
print('-----------Train result--------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(train_R,train_MSE,train_MAE))
print('-----------Test result---------------------------------------------')
print("R,MSE,MAE依次为：{},{},{}".format(test_R,test_mse,test_mae))