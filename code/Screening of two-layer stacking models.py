import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_validate,KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor
from mlxtend.regressor import StackingCVRegressor
import hyperopt
from hyperopt import hp,fmin,tpe,Trials,partial
from hyperopt.early_stop import no_progress_loss
df=pd.read_excel(r"D:\桌面文件\研一上\学习笔记\机器学习\jupyter notebook save path\data\smogn.xlsx")
x=df.iloc[:,:3]
y=df.iloc[:,-1:]
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))
x=x_scaler.fit_transform(x)
y=y_scaler.fit_transform(y)
y=y.ravel()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=2024)
svr=SVR().fit(x_train,y_train)
KNN= KNeighborsRegressor(algorithm='ball_tree'
                         ,leaf_size=17
                         ,n_neighbors=3
                         ,p=6
                         ,n_jobs=-1).fit(x_train,y_train)
LGBM=lgb.LGBMRegressor(learning_rate= 0.3187407985789513
                       ,max_depth=9
                       ,min_child_weight=0.15600579600317482
                       ,min_data_in_leaf=5
                       ,n_estimators=159
                       ,num_leaves=35
                       ,reg_lambda=0.5154249691458345
                       ,subsample=0.42756975409141845
                       ,random_state=2024).fit(x_train,y_train)
XGB=XGBRegressor(colsample_bytree=0.7658291861093494,
                  learning_rate=0.12631595569656445
                  ,max_depth=15
                  ,n_estimators=236
                  ,reg_alpha=0.029890252266918105
                  ,reg_lambda=2.117684588707199
                  ,subsample=0.6689278792614803
                  ,random_state=2024
                  ,n_jobs=-1 ).fit(x_train,y_train)
ET=ExtraTreesRegressor(max_depth=16
                       ,max_features=0.690224732325302
                       ,n_estimators=573,random_state=2024).fit(x_train,y_train)
GBDT=GradientBoostingRegressor(learning_rate=0.22311461706832098
                               ,max_depth=5
                               ,min_samples_leaf=5
                               ,n_estimators=555
                               ,random_state=2024).fit(x_train,y_train)
rf=RandomForestRegressor(max_depth=25
                           ,max_features=0.6850005170903558
                           ,max_samples=0.9747549912335957
                           ,min_samples_split=2
                           ,n_estimators=50
                           ,random_state=2024).fit(x_train,y_train)
#元模型SVR
def hypeopt_objective(params):#传入超参数（字典形式）
    model1 = params["model1"]
    model2 = params["model2"]
    model3 = params["model3"]
    model4 = params["model4"]
    model5 = params["model5"]
#model1
    if model1=='svr':
        base_model1=svr
    elif model1=='KNN':
        base_model1= KNN
    elif model1=='LGBM':
        base_model1=LGBM
    elif model1=='XGB':
        base_model1 =XGB
    elif model1=='ET':
        base_model1=ET
    elif model1=='GBDT':
        base_model1=GBDT
    elif model1=='rf':
        base_model1 =rf
#model2
    if model2=='svr':
        base_model2=svr
    elif model2=='KNN':
        base_model2= KNN
    elif model2=='LGBM':
        base_model2=LGBM
    elif model2=='XGB':
        base_model2 = XGB
    elif model2=='ET':
        base_model2=ET
    elif model2=='GBDT':
        base_model2=GBDT
    elif model2=='rf':
        base_model2 = rf
#model3
    if model3=='svr':
        base_model3=svr
    elif model3=='KNN':
        base_model3= KNN
    elif model3=='LGBM':
        base_model3=LGBM
    elif model3=='XGB':
        base_model3 = XGB
    elif model3=='ET':
        base_model3=ET
    elif model3=='GBDT':
        base_model3=GBDT
    elif model3=='rf':
        base_model3 =rf
#model4
    if model4=='svr':
        base_model4=svr
    elif model4=='KNN':
        base_model4=KNN
    elif model4=='LGBM':
        base_model4=LGBM
    elif model4=='XGB':
        base_model4 =XGB
    elif model4=='ET':
        base_model4=ET
    elif model4=='GBDT':
        base_model4=GBDT
    elif model4=='rf':
        base_model4 =rf
#model5
    if model5=='svr':
        base_model5=svr
    elif model5=='KNN':
        base_model5=KNN
    elif model5=='LGBM':
        base_model5=LGBM
    elif model5=='XGB':
        base_model5 =XGB
    elif model5=='ET':
        base_model5=ET
    elif model5=='GBDT':
        base_model5=GBDT
    elif model5=='rf':
        base_model5 =rf
    stack_gen=StackingCVRegressor(regressors=(base_model1,base_model2,base_model3,base_model4,base_model5)
                                      ,meta_regressor=svr
                                      ,shuffle=True
                                      ,use_features_in_secondary=True
                                      ,n_jobs=-1
                                      ,random_state=2024)
    cv=KFold(n_splits=5,shuffle=True,random_state=2024)
    validation_loss=cross_validate(stack_gen
                               ,x_train
                               ,y_train
                               ,cv=cv
                               ,verbose=False
                               ,n_jobs=-1
                               ,error_score='raise'
                                )
    return -np.mean(validation_loss["test_score"])
param_grid_simple={
                    'model1':hp.choice("model1",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,'model2':hp.choice("model2",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,"model3":hp.choice("model3",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,"model4":hp.choice("model4",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,'model5':hp.choice("model5",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                    }
def param_hypeopt(max_evals):
    trials=Trials()
    early_stop_fn=no_progress_loss(300)
    params_best=fmin(hypeopt_objective
                    ,space=param_grid_simple
                    ,algo=tpe.suggest
                    ,max_evals=max_evals
                    ,verbose=True
                    ,trials=trials
                    ,early_stop_fn=early_stop_fn)
    print("\n","\n","best params:",params_best,"\n")
    return params_best,trials
params_best,trials=param_hypeopt(500)
model_list=['svr','KNN','LGBM','XGB','ET','GBDT','rf']
model1=model_list[params_best["model1"]]
model2=model_list[params_best["model2"]]
model3=model_list[params_best["model3"]]
model4=model_list[params_best["model4"]]
model5=model_list[params_best["model5"]]
print("Metamodel is svr")
print("Model combination: {}, {}, {}, {}, {}".format(model1,model2,model3,model4,model5))
#元模型KNN
def hypeopt_objective(params):#传入超参数（字典形式）
    model1 = params["model1"]
    model2 = params["model2"]
    model3 = params["model3"]
    model4 = params["model4"]
    model5 = params["model5"]
#model1
    if model1=='svr':
        base_model1=svr
    elif model1=='KNN':
        base_model1= KNN
    elif model1=='LGBM':
        base_model1=LGBM
    elif model1=='XGB':
        base_model1 =XGB
    elif model1=='ET':
        base_model1=ET
    elif model1=='GBDT':
        base_model1=GBDT
    elif model1=='rf':
        base_model1 =rf
#model2
    if model2=='svr':
        base_model2=svr
    elif model2=='KNN':
        base_model2= KNN
    elif model2=='LGBM':
        base_model2=LGBM
    elif model2=='XGB':
        base_model2 = XGB
    elif model2=='ET':
        base_model2=ET
    elif model2=='GBDT':
        base_model2=GBDT
    elif model2=='rf':
        base_model2 = rf
#model3
    if model3=='svr':
        base_model3=svr
    elif model3=='KNN':
        base_model3= KNN
    elif model3=='LGBM':
        base_model3=LGBM
    elif model3=='XGB':
        base_model3 = XGB
    elif model3=='ET':
        base_model3=ET
    elif model3=='GBDT':
        base_model3=GBDT
    elif model3=='rf':
        base_model3 =rf
#model4
    if model4=='svr':
        base_model4=svr
    elif model4=='KNN':
        base_model4=KNN
    elif model4=='LGBM':
        base_model4=LGBM
    elif model4=='XGB':
        base_model4 =XGB
    elif model4=='ET':
        base_model4=ET
    elif model4=='GBDT':
        base_model4=GBDT
    elif model4=='rf':
        base_model4 =rf
#model5
    if model5=='svr':
        base_model5=svr
    elif model5=='KNN':
        base_model5=KNN
    elif model5=='LGBM':
        base_model5=LGBM
    elif model5=='XGB':
        base_model5 =XGB
    elif model5=='ET':
        base_model5=ET
    elif model5=='GBDT':
        base_model5=GBDT
    elif model5=='rf':
        base_model5 =rf
    stack_gen=StackingCVRegressor(regressors=(base_model1,base_model2,base_model3,base_model4,base_model5)
                                      ,meta_regressor=KNN
                                      ,shuffle=True
                                      ,use_features_in_secondary=True
                                      ,n_jobs=-1
                                      ,random_state=2024)
    cv=KFold(n_splits=5,shuffle=True,random_state=2024)
    validation_loss=cross_validate(stack_gen
                               ,x_train
                               ,y_train
                               ,cv=cv
                               ,verbose=False
                               ,n_jobs=-1
                               ,error_score='raise'
                                )
    return -np.mean(validation_loss["test_score"])
param_grid_simple={
                    'model1':hp.choice("model1",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,'model2':hp.choice("model2",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,"model3":hp.choice("model3",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,"model4":hp.choice("model4",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,'model5':hp.choice("model5",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                    }
def param_hypeopt(max_evals):
    trials=Trials()
    early_stop_fn=no_progress_loss(300)
    params_best=fmin(hypeopt_objective
                    ,space=param_grid_simple
                    ,algo=tpe.suggest
                    ,max_evals=max_evals
                    ,verbose=True
                    ,trials=trials
                    ,early_stop_fn=early_stop_fn)
    print("\n","\n","best params:",params_best,"\n")
    return params_best,trials
params_best,trials=param_hypeopt(500)
model_list=['svr','KNN','LGBM','XGB','ET','GBDT','rf']
model1=model_list[params_best["model1"]]
model2=model_list[params_best["model2"]]
model3=model_list[params_best["model3"]]
model4=model_list[params_best["model4"]]
model5=model_list[params_best["model5"]]
print("Metamodel is KNN")
print("Model combination: {}, {}, {}, {}, {}".format(model1,model2,model3,model4,model5))
#元模型LGBM
def hypeopt_objective(params):#传入超参数（字典形式）
    model1 = params["model1"]
    model2 = params["model2"]
    model3 = params["model3"]
    model4 = params["model4"]
    model5 = params["model5"]
#model1
    if model1=='svr':
        base_model1=svr
    elif model1=='KNN':
        base_model1= KNN
    elif model1=='LGBM':
        base_model1=LGBM
    elif model1=='XGB':
        base_model1 =XGB
    elif model1=='ET':
        base_model1=ET
    elif model1=='GBDT':
        base_model1=GBDT
    elif model1=='rf':
        base_model1 =rf
#model2
    if model2=='svr':
        base_model2=svr
    elif model2=='KNN':
        base_model2= KNN
    elif model2=='LGBM':
        base_model2=LGBM
    elif model2=='XGB':
        base_model2 = XGB
    elif model2=='ET':
        base_model2=ET
    elif model2=='GBDT':
        base_model2=GBDT
    elif model2=='rf':
        base_model2 = rf
#model3
    if model3=='svr':
        base_model3=svr
    elif model3=='KNN':
        base_model3= KNN
    elif model3=='LGBM':
        base_model3=LGBM
    elif model3=='XGB':
        base_model3 = XGB
    elif model3=='ET':
        base_model3=ET
    elif model3=='GBDT':
        base_model3=GBDT
    elif model3=='rf':
        base_model3 =rf
#model4
    if model4=='svr':
        base_model4=svr
    elif model4=='KNN':
        base_model4=KNN
    elif model4=='LGBM':
        base_model4=LGBM
    elif model4=='XGB':
        base_model4 =XGB
    elif model4=='ET':
        base_model4=ET
    elif model4=='GBDT':
        base_model4=GBDT
    elif model4=='rf':
        base_model4 =rf
#model5
    if model5=='svr':
        base_model5=svr
    elif model5=='KNN':
        base_model5=KNN
    elif model5=='LGBM':
        base_model5=LGBM
    elif model5=='XGB':
        base_model5 =XGB
    elif model5=='ET':
        base_model5=ET
    elif model5=='GBDT':
        base_model5=GBDT
    elif model5=='rf':
        base_model5 =rf
    stack_gen=StackingCVRegressor(regressors=(base_model1,base_model2,base_model3,base_model4,base_model5)
                                      ,meta_regressor=LGBM
                                      ,shuffle=True
                                      ,use_features_in_secondary=True
                                      ,n_jobs=-1
                                      ,random_state=2024)
    cv=KFold(n_splits=5,shuffle=True,random_state=2024)
    validation_loss=cross_validate(stack_gen
                               ,x_train
                               ,y_train
                               ,cv=cv
                               ,verbose=False
                               ,n_jobs=-1
                               ,error_score='raise'
                                )
    return -np.mean(validation_loss["test_score"])
param_grid_simple={
                    'model1':hp.choice("model1",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,'model2':hp.choice("model2",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,"model3":hp.choice("model3",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,"model4":hp.choice("model4",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,'model5':hp.choice("model5",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                    }
def param_hypeopt(max_evals):
    trials=Trials()
    early_stop_fn=no_progress_loss(300)
    params_best=fmin(hypeopt_objective
                    ,space=param_grid_simple
                    ,algo=tpe.suggest
                    ,max_evals=max_evals
                    ,verbose=True
                    ,trials=trials
                    ,early_stop_fn=early_stop_fn)
    print("\n","\n","best params:",params_best,"\n")
    return params_best,trials
params_best,trials=param_hypeopt(500)
model_list=['svr','KNN','LGBM','XGB','ET','GBDT','rf']
model1=model_list[params_best["model1"]]
model2=model_list[params_best["model2"]]
model3=model_list[params_best["model3"]]
model4=model_list[params_best["model4"]]
model5=model_list[params_best["model5"]]
print("Metamodel is LGBM")
print("Model combination: {}, {}, {}, {}, {}".format(model1,model2,model3,model4,model5))
#元模型XGB
def hypeopt_objective(params):#传入超参数（字典形式）
    model1 = params["model1"]
    model2 = params["model2"]
    model3 = params["model3"]
    model4 = params["model4"]
    model5 = params["model5"]
#model1
    if model1=='svr':
        base_model1=svr
    elif model1=='KNN':
        base_model1= KNN
    elif model1=='LGBM':
        base_model1=LGBM
    elif model1=='XGB':
        base_model1 =XGB
    elif model1=='ET':
        base_model1=ET
    elif model1=='GBDT':
        base_model1=GBDT
    elif model1=='rf':
        base_model1 =rf
#model2
    if model2=='svr':
        base_model2=svr
    elif model2=='KNN':
        base_model2= KNN
    elif model2=='LGBM':
        base_model2=LGBM
    elif model2=='XGB':
        base_model2 = XGB
    elif model2=='ET':
        base_model2=ET
    elif model2=='GBDT':
        base_model2=GBDT
    elif model2=='rf':
        base_model2 = rf
#model3
    if model3=='svr':
        base_model3=svr
    elif model3=='KNN':
        base_model3= KNN
    elif model3=='LGBM':
        base_model3=LGBM
    elif model3=='XGB':
        base_model3 = XGB
    elif model3=='ET':
        base_model3=ET
    elif model3=='GBDT':
        base_model3=GBDT
    elif model3=='rf':
        base_model3 =rf
#model4
    if model4=='svr':
        base_model4=svr
    elif model4=='KNN':
        base_model4=KNN
    elif model4=='LGBM':
        base_model4=LGBM
    elif model4=='XGB':
        base_model4 =XGB
    elif model4=='ET':
        base_model4=ET
    elif model4=='GBDT':
        base_model4=GBDT
    elif model4=='rf':
        base_model4 =rf
#model5
    if model5=='svr':
        base_model5=svr
    elif model5=='KNN':
        base_model5=KNN
    elif model5=='LGBM':
        base_model5=LGBM
    elif model5=='XGB':
        base_model5 =XGB
    elif model5=='ET':
        base_model5=ET
    elif model5=='GBDT':
        base_model5=GBDT
    elif model5=='rf':
        base_model5 =rf
    stack_gen=StackingCVRegressor(regressors=(base_model1,base_model2,base_model3,base_model4,base_model5)
                                      ,meta_regressor=XGB
                                      ,shuffle=True
                                      ,use_features_in_secondary=True
                                      ,n_jobs=-1
                                      ,random_state=2024)
    cv=KFold(n_splits=5,shuffle=True,random_state=2024)
    validation_loss=cross_validate(stack_gen
                               ,x_train
                               ,y_train
                               ,cv=cv
                               ,verbose=False
                               ,n_jobs=-1
                               ,error_score='raise'
                                )
    return -np.mean(validation_loss["test_score"])
param_grid_simple={
                    'model1':hp.choice("model1",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,'model2':hp.choice("model2",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,"model3":hp.choice("model3",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,"model4":hp.choice("model4",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,'model5':hp.choice("model5",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                    }
def param_hypeopt(max_evals):
    trials=Trials()
    early_stop_fn=no_progress_loss(300)
    params_best=fmin(hypeopt_objective
                    ,space=param_grid_simple
                    ,algo=tpe.suggest
                    ,max_evals=max_evals
                    ,verbose=True
                    ,trials=trials
                    ,early_stop_fn=early_stop_fn)
    print("\n","\n","best params:",params_best,"\n")
    return params_best,trials
params_best,trials=param_hypeopt(500)
model_list=['svr','KNN','LGBM','XGB','ET','GBDT','rf']
model1=model_list[params_best["model1"]]
model2=model_list[params_best["model2"]]
model3=model_list[params_best["model3"]]
model4=model_list[params_best["model4"]]
model5=model_list[params_best["model5"]]
print("Metamodel is XGB")
print("Model combination: {}, {}, {}, {}, {}".format(model1,model2,model3,model4,model5))
#元模型ET
def hypeopt_objective(params):#传入超参数（字典形式）
    model1 = params["model1"]
    model2 = params["model2"]
    model3 = params["model3"]
    model4 = params["model4"]
    model5 = params["model5"]
#model1
    if model1=='svr':
        base_model1=svr
    elif model1=='KNN':
        base_model1= KNN
    elif model1=='LGBM':
        base_model1=LGBM
    elif model1=='XGB':
        base_model1 =XGB
    elif model1=='ET':
        base_model1=ET
    elif model1=='GBDT':
        base_model1=GBDT
    elif model1=='rf':
        base_model1 =rf
#model2
    if model2=='svr':
        base_model2=svr
    elif model2=='KNN':
        base_model2= KNN
    elif model2=='LGBM':
        base_model2=LGBM
    elif model2=='XGB':
        base_model2 = XGB
    elif model2=='ET':
        base_model2=ET
    elif model2=='GBDT':
        base_model2=GBDT
    elif model2=='rf':
        base_model2 = rf
#model3
    if model3=='svr':
        base_model3=svr
    elif model3=='KNN':
        base_model3= KNN
    elif model3=='LGBM':
        base_model3=LGBM
    elif model3=='XGB':
        base_model3 = XGB
    elif model3=='ET':
        base_model3=ET
    elif model3=='GBDT':
        base_model3=GBDT
    elif model3=='rf':
        base_model3 =rf
#model4
    if model4=='svr':
        base_model4=svr
    elif model4=='KNN':
        base_model4=KNN
    elif model4=='LGBM':
        base_model4=LGBM
    elif model4=='XGB':
        base_model4 =XGB
    elif model4=='ET':
        base_model4=ET
    elif model4=='GBDT':
        base_model4=GBDT
    elif model4=='rf':
        base_model4 =rf
#model5
    if model5=='svr':
        base_model5=svr
    elif model5=='KNN':
        base_model5=KNN
    elif model5=='LGBM':
        base_model5=LGBM
    elif model5=='XGB':
        base_model5 =XGB
    elif model5=='ET':
        base_model5=ET
    elif model5=='GBDT':
        base_model5=GBDT
    elif model5=='rf':
        base_model5 =rf
    stack_gen=StackingCVRegressor(regressors=(base_model1,base_model2,base_model3,base_model4,base_model5)
                                      ,meta_regressor=ET
                                      ,shuffle=True
                                      ,use_features_in_secondary=True
                                      ,n_jobs=-1
                                      ,random_state=2024)
    cv=KFold(n_splits=5,shuffle=True,random_state=2024)
    validation_loss=cross_validate(stack_gen
                               ,x_train
                               ,y_train
                               ,cv=cv
                               ,verbose=False
                               ,n_jobs=-1
                               ,error_score='raise'
                                )
    return -np.mean(validation_loss["test_score"])
param_grid_simple={
                    'model1':hp.choice("model1",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,'model2':hp.choice("model2",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,"model3":hp.choice("model3",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,"model4":hp.choice("model4",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,'model5':hp.choice("model5",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                    }
def param_hypeopt(max_evals):
    trials=Trials()
    early_stop_fn=no_progress_loss(300)
    params_best=fmin(hypeopt_objective
                    ,space=param_grid_simple
                    ,algo=tpe.suggest
                    ,max_evals=max_evals
                    ,verbose=True
                    ,trials=trials
                    ,early_stop_fn=early_stop_fn)
    print("\n","\n","best params:",params_best,"\n")
    return params_best,trials
params_best,trials=param_hypeopt(500)
model_list=['svr','KNN','LGBM','XGB','ET','GBDT','rf']
model1=model_list[params_best["model1"]]
model2=model_list[params_best["model2"]]
model3=model_list[params_best["model3"]]
model4=model_list[params_best["model4"]]
model5=model_list[params_best["model5"]]
print("Metamodel is ET")
print("Model combination: {}, {}, {}, {}, {}".format(model1,model2,model3,model4,model5))
#元模型GBDT
def hypeopt_objective(params):#传入超参数（字典形式）
    model1 = params["model1"]
    model2 = params["model2"]
    model3 = params["model3"]
    model4 = params["model4"]
    model5 = params["model5"]
#model1
    if model1=='svr':
        base_model1=svr
    elif model1=='KNN':
        base_model1= KNN
    elif model1=='LGBM':
        base_model1=LGBM
    elif model1=='XGB':
        base_model1 =XGB
    elif model1=='ET':
        base_model1=ET
    elif model1=='GBDT':
        base_model1=GBDT
    elif model1=='rf':
        base_model1 =rf
#model2
    if model2=='svr':
        base_model2=svr
    elif model2=='KNN':
        base_model2= KNN
    elif model2=='LGBM':
        base_model2=LGBM
    elif model2=='XGB':
        base_model2 = XGB
    elif model2=='ET':
        base_model2=ET
    elif model2=='GBDT':
        base_model2=GBDT
    elif model2=='rf':
        base_model2 = rf
#model3
    if model3=='svr':
        base_model3=svr
    elif model3=='KNN':
        base_model3= KNN
    elif model3=='LGBM':
        base_model3=LGBM
    elif model3=='XGB':
        base_model3 = XGB
    elif model3=='ET':
        base_model3=ET
    elif model3=='GBDT':
        base_model3=GBDT
    elif model3=='rf':
        base_model3 =rf
#model4
    if model4=='svr':
        base_model4=svr
    elif model4=='KNN':
        base_model4=KNN
    elif model4=='LGBM':
        base_model4=LGBM
    elif model4=='XGB':
        base_model4 =XGB
    elif model4=='ET':
        base_model4=ET
    elif model4=='GBDT':
        base_model4=GBDT
    elif model4=='rf':
        base_model4 =rf
#model5
    if model5=='svr':
        base_model5=svr
    elif model5=='KNN':
        base_model5=KNN
    elif model5=='LGBM':
        base_model5=LGBM
    elif model5=='XGB':
        base_model5 =XGB
    elif model5=='ET':
        base_model5=ET
    elif model5=='GBDT':
        base_model5=GBDT
    elif model5=='rf':
        base_model5 =rf
    stack_gen=StackingCVRegressor(regressors=(base_model1,base_model2,base_model3,base_model4,base_model5)
                                      ,meta_regressor=GBDT
                                      ,shuffle=True
                                      ,use_features_in_secondary=True
                                      ,n_jobs=-1
                                      ,random_state=2024)
    cv=KFold(n_splits=5,shuffle=True,random_state=2024)
    validation_loss=cross_validate(stack_gen
                               ,x_train
                               ,y_train
                               ,cv=cv
                               ,verbose=False
                               ,n_jobs=-1
                               ,error_score='raise'
                                )
    return -np.mean(validation_loss["test_score"])
param_grid_simple={
                    'model1':hp.choice("model1",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,'model2':hp.choice("model2",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,"model3":hp.choice("model3",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,"model4":hp.choice("model4",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,'model5':hp.choice("model5",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                    }
def param_hypeopt(max_evals):
    trials=Trials()
    early_stop_fn=no_progress_loss(300)
    params_best=fmin(hypeopt_objective
                    ,space=param_grid_simple
                    ,algo=tpe.suggest
                    ,max_evals=max_evals
                    ,verbose=True
                    ,trials=trials
                    ,early_stop_fn=early_stop_fn)
    print("\n","\n","best params:",params_best,"\n")
    return params_best,trials
params_best,trials=param_hypeopt(500)
model_list=['svr','KNN','LGBM','XGB','ET','GBDT','rf']
model1=model_list[params_best["model1"]]
model2=model_list[params_best["model2"]]
model3=model_list[params_best["model3"]]
model4=model_list[params_best["model4"]]
model5=model_list[params_best["model5"]]
print("Metamodel is GBDT")
print("Model combination: {}, {}, {}, {}, {}".format(model1,model2,model3,model4,model5))
#元模型rf
def hypeopt_objective(params):#传入超参数（字典形式）
    model1 = params["model1"]
    model2 = params["model2"]
    model3 = params["model3"]
    model4 = params["model4"]
    model5 = params["model5"]
#model1
    if model1=='svr':
        base_model1=svr
    elif model1=='KNN':
        base_model1= KNN
    elif model1=='LGBM':
        base_model1=LGBM
    elif model1=='XGB':
        base_model1 =XGB
    elif model1=='ET':
        base_model1=ET
    elif model1=='GBDT':
        base_model1=GBDT
    elif model1=='rf':
        base_model1 =rf
#model2
    if model2=='svr':
        base_model2=svr
    elif model2=='KNN':
        base_model2= KNN
    elif model2=='LGBM':
        base_model2=LGBM
    elif model2=='XGB':
        base_model2 = XGB
    elif model2=='ET':
        base_model2=ET
    elif model2=='GBDT':
        base_model2=GBDT
    elif model2=='rf':
        base_model2 = rf
#model3
    if model3=='svr':
        base_model3=svr
    elif model3=='KNN':
        base_model3= KNN
    elif model3=='LGBM':
        base_model3=LGBM
    elif model3=='XGB':
        base_model3 = XGB
    elif model3=='ET':
        base_model3=ET
    elif model3=='GBDT':
        base_model3=GBDT
    elif model3=='rf':
        base_model3 =rf
#model4
    if model4=='svr':
        base_model4=svr
    elif model4=='KNN':
        base_model4=KNN
    elif model4=='LGBM':
        base_model4=LGBM
    elif model4=='XGB':
        base_model4 =XGB
    elif model4=='ET':
        base_model4=ET
    elif model4=='GBDT':
        base_model4=GBDT
    elif model4=='rf':
        base_model4 =rf
#model5
    if model5=='svr':
        base_model5=svr
    elif model5=='KNN':
        base_model5=KNN
    elif model5=='LGBM':
        base_model5=LGBM
    elif model5=='XGB':
        base_model5 =XGB
    elif model5=='ET':
        base_model5=ET
    elif model5=='GBDT':
        base_model5=GBDT
    elif model5=='rf':
        base_model5 =rf
    stack_gen=StackingCVRegressor(regressors=(base_model1,base_model2,base_model3,base_model4,base_model5)
                                      ,meta_regressor=rf
                                      ,shuffle=True
                                      ,use_features_in_secondary=True
                                      ,n_jobs=-1
                                      ,random_state=2024)
    cv=KFold(n_splits=5,shuffle=True,random_state=2024)
    validation_loss=cross_validate(stack_gen
                               ,x_train
                               ,y_train
                               ,cv=cv
                               ,verbose=False
                               ,n_jobs=-1
                               ,error_score='raise'
                                )
    return -np.mean(validation_loss["test_score"])
param_grid_simple={
                    'model1':hp.choice("model1",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,'model2':hp.choice("model2",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,"model3":hp.choice("model3",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,"model4":hp.choice("model4",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                   ,'model5':hp.choice("model5",['svr','KNN','LGBM','XGB','ET','GBDT','rf'])
                    }
def param_hypeopt(max_evals):
    trials=Trials()
    early_stop_fn=no_progress_loss(300)
    params_best=fmin(hypeopt_objective
                    ,space=param_grid_simple
                    ,algo=tpe.suggest
                    ,max_evals=max_evals
                    ,verbose=True
                    ,trials=trials
                    ,early_stop_fn=early_stop_fn)
    print("\n","\n","best params:",params_best,"\n")
    return params_best,trials
params_best,trials=param_hypeopt(500)
model_list=['svr','KNN','LGBM','XGB','ET','GBDT','rf']
model1=model_list[params_best["model1"]]
model2=model_list[params_best["model2"]]
model3=model_list[params_best["model3"]]
model4=model_list[params_best["model4"]]
model5=model_list[params_best["model5"]]
print("Metamodel is rf")
print("Model combination: {}, {}, {}, {}, {}".format(model1,model2,model3,model4,model5))