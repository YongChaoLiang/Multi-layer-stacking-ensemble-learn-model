import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
#regression model
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,BayesianRidge#线性模型
from sklearn.svm import SVR#SVR
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb#LGBM
from xgboost import XGBRegressor#XGboost
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor
from mlxtend.regressor import StackingCVRegressor#stacking
import scipy.stats as stats
from minepy import MINE
df=pd.read_excel(r"D:\桌面文件\研一上\学习笔记\机器学习\jupyter notebook save path\data\smogn.xlsx")
x=df.iloc[:,:3]
y=df.iloc[:,-1:]
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))
x = x_scaler.fit_transform(x)
y = y_scaler.fit_transform(y)
y=y.ravel()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=2024)
DT=DecisionTreeRegressor(random_state=2024).fit(x_train,y_train)
svr=SVR().fit(x_train,y_train)
KNN=KNeighborsRegressor(n_jobs=-1).fit(x_train,y_train)
LGBM=lgb.LGBMRegressor(random_state=2024).fit(x_train,y_train)
XGB=XGBRegressor(random_state=2024,n_jobs=-1 ).fit(x_train,y_train)
ET=ExtraTreesRegressor(random_state=2024).fit(x_train,y_train)
GBDT=GradientBoostingRegressor(random_state=2024).fit(x_train,y_train)
rf=RandomForestRegressor(random_state=2024).fit(x_train,y_train)
model1_predictions = DT.predict(x_test)
model2_predictions = svr.predict(x_test)
model3_predictions = KNN.predict(x_test)
model4_predictions = LGBM.predict(x_test)
model5_predictions = XGB.predict(x_test)
model6_predictions =ET.predict(x_test)
model7_predictions = GBDT.predict(x_test)
model8_predictions = rf.predict(x_test)
# concat
predictions = [model1_predictions, model2_predictions, model3_predictions,
               model4_predictions, model5_predictions, model6_predictions,
               model7_predictions, model8_predictions
               ]
# MIC
mic_values = []
for i in range(len(predictions)):
    tmp = []
    for j in range(len(predictions)):
        m = MINE()
        m.compute_score(predictions[i], predictions[j])
        tmp.append(m.mic())
    mic_values.append(tmp)
mic_matrix = np.array(mic_values)
# show
fig, ax = plt.subplots(figsize=(6,6))
def plot_confusion_matrix(cm, title, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap,vmin=0,vmax=1)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(8)
    plt.xticks(tick_marks, ['DT','svr','KNN', 'LGBM', 'XGB','ET', 'GBDT', 'rf'], rotation=45)
    plt.yticks(tick_marks, ['DT','svr','KNN', 'LGBM', 'XGB','ET', 'GBDT','rf'])
    plt.tight_layout()
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
plot_confusion_matrix(mic_matrix, title='MIC values between regression models')
def sum_array(arr):
    result = 0
    for num in arr:
        result += num
    return result/10
mic_values=[]
for i in range(8):
    array_sum = sum_array(mic_matrix[i])
    mic_values.append(array_sum)
def correlation_coefficient(y_true, y_pred):
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    return correlation
model_list = [DT,svr,KNN,LGBM,XGB,ET,GBDT,rf]
R_test_list=[]
for model in model_list:
    y_test_pred = model.predict(x_test)
    R_test=correlation_coefficient(y_test, y_test_pred)
    R_test_list.append(R_test)
mic_values= np.array(mic_values)
R_test_list= np.array(R_test_list)
result=R_test_list*0.6+(1-mic_values)*0.4
result.tolist()