import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold,cross_validate,train_test_split
from sklearn.preprocessing import MinMaxScaler
import hyperopt
from hyperopt import hp,fmin,tpe,Trials,partial
from hyperopt.early_stop import no_progress_loss
df=pd.read_excel(r"D:\桌面文件\研一上\学习笔记\机器学习\jupyter notebook save path\data\smogn.xlsx")
x=df.iloc[:,:3]
y=df.iloc[:,-1:]
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))
x = x_scaler.fit_transform(x)
y = y_scaler.fit_transform(y)
y=y.ravel()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2024)
#RF
def hypeopt_objective(params):
    reg=RandomForestRegressor(n_estimators=int(params["n_estimators"])
            ,max_depth=int(params["max_depth"])
            ,max_features=params["max_features"]
            ,min_samples_split=int(params["min_samples_split"])
            ,random_state=2024
            ,verbose=False
            ,n_jobs=-1
            ).fit(x_train,y_train)
    cv=KFold(n_splits=5,shuffle=True,random_state=2024)
    validation_loss=cross_validate(reg
                                   ,x_train,y_train
                                   ,cv=cv
                                   ,verbose=False
                                   ,n_jobs=-1
                                   ,error_score='raise'
                                    )
    return -np.mean(validation_loss["train_score"])
param_grid_simple={
                    'n_estimators':hp.quniform("n_estimators",20,60,1)
                    ,'max_depth':hp.quniform("max_depth",10,30,1)
                    ,"max_features": hp.uniform("max_features",0.6,0.9)
                    ,"min_samples_split":hp.quniform("min_samples_split",2,5,1)
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
                    ,early_stop_fn=early_stop_fn
                    )
    print("\n","\n","best params:",params_best,"\n")
    return params_best,trials
import time
start=time.time()
params_best,trials=param_hypeopt(500)
print("时间:",time.time()-start)
#KNN
def hypeopt_objective(params):
    reg=KNeighborsRegressor(n_neighbors=int(params["n_neighbors"])
                            ,leaf_size=int(params["leaf_size"])
                            ,p=int(params["p"])
                            ,algorithm=params["algorithm"]
                            ,weights=params["weights"]
                            ,n_jobs=-1
            ).fit(x_train,y_train)
    cv=KFold(n_splits=5,shuffle=True,random_state=2024)
    validation_loss=cross_validate(reg
                                   ,x_train,y_train
                                   ,cv=cv
                                   ,verbose=False
                                   ,n_jobs=-1
                                   ,error_score='raise'
                                    )
    return -np.mean(validation_loss["train_score"])
param_grid_simple={
                    'n_neighbors':hp.quniform("n_neighbors",1,6,1)
                    ,'leaf_size':hp.quniform("leaf_size",10,40,1)
                    ,'p':hp.quniform("p",5,10,1)
                    ,'algorithm':hp.choice("algorithm",['auto', 'ball_tree', 'kd_tree', 'brute'])
                    ,'weights': hp.choice('weights', ['uniform', 'distance'])
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
                    ,early_stop_fn=early_stop_fn
                    )
    print("\n","\n","best params:",params_best,"\n")
    return params_best,trials
import time
start=time.time()
params_best,trials=param_hypeopt(500)
print("时间:",time.time()-start)
#ET
def hypeopt_objective(params):
    reg=ExtraTreesRegressor(n_estimators=int(params["n_estimators"])
            ,max_features=params["max_features"]
            ,max_depth=int(params["max_depth"])
            ,random_state=2024
            ,verbose=False
            ,n_jobs=-1
            ).fit(x_train,y_train)
    cv=KFold(n_splits=5,shuffle=True,random_state=2024)
    validation_loss=cross_validate(reg
                                   ,x_train,y_train
                                   ,cv=cv
                                   ,verbose=False
                                   ,n_jobs=-1
                                   ,error_score='raise'
                                    )
    return -np.mean(validation_loss["train_score"])
param_grid_simple={ "n_estimators":hp.quniform("n_estimators",500,600,1)
                    ,"max_features": hp.uniform("max_features",0.7,0.99)
                    ,"max_depth":hp.quniform("max_depth",10,30,1)
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
                    ,early_stop_fn=early_stop_fn
                    )
    print("\n","\n","best params:",params_best,"\n")
    return params_best,trials
params_best,trials=param_hypeopt(500)
#XGBoost
def hypeopt_objective(params):
    reg=xgb.XGBRegressor(n_estimators=int(params["n_estimators"])
            ,learning_rate=params["learning_rate"]
            ,max_depth=int(params["max_depth"])
            ,subsample=params["subsample"]
            ,colsample_bytree=params["colsample_bytree"]
            ,reg_alpha=params["reg_alpha"]
            ,reg_lambda=params["reg_lambda"]
            ,random_state=2024
            ,n_jobs=-1
            ).fit(x_train,y_train)
    cv=KFold(n_splits=5,shuffle=True,random_state=2024)
    validation_loss=cross_validate(reg
                                   ,x_train,y_train
                                   ,cv=cv
                                   ,verbose=False
                                   ,n_jobs=-1
                                   ,error_score='raise'
                                    )
    return -np.mean(validation_loss["train_score"])
param_grid_simple={
                    'n_estimators':hp.quniform("n_estimators",200,300,1)
                    ,'learning_rate':hp.uniform("learning_rate",0.1,0.4)
                    ,'max_depth':hp.quniform("max_depth",1,25,1)
                    ,"subsample": hp.uniform("subsample",0.5,1.0)
                    ,"colsample_bytree":hp.uniform("colsample_bytree",0.5,1.0)
                    ,"reg_alpha": hp.uniform("reg_alpha",0,0.1)
                    ,"reg_lambda": hp.uniform("reg_lambda",1.5,2.5)
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
                    ,early_stop_fn=early_stop_fn
                    )
    print("\n","\n","best params:",params_best,"\n")
    return params_best,trials
import time
start=time.time()
params_best,trials=param_hypeopt(500)
print("时间:",time.time()-start)
#LGBM
def hypeopt_objective(params):
    reg=lgb.LGBMRegressor(num_leaves=int(params["num_leaves"])
            ,max_depth=int(params["max_depth"])
            ,min_child_samples=int(params["min_child_samples"])
            #,min_data_in_leaf=int(params["min_data_in_leaf"])
            ,learning_rate=params["learning_rate"]
            ,n_estimators=int(params["n_estimators"])
            ,min_child_weight=params["min_child_weight"]
            ,subsample=params["subsample"]
            ,reg_lambda=params["reg_lambda"]
            ,random_state=2024
            ,n_jobs=-1
            ).fit(x_train,y_train)
    cv=KFold(n_splits=5,shuffle=True,random_state=2024)
    validation_loss=cross_validate(reg
                                   ,x_train,y_train
                                   ,cv=cv
                                   ,verbose=False
                                   ,n_jobs=-1
                                   ,error_score='raise'
                                    )
    return -np.mean(validation_loss["train_score"])
param_grid_simple={
                    'num_leaves':hp.quniform("num_leaves",25,35,1)
                    ,"learning_rate": hp.uniform("learning_rate",0.3,0.4)
                    ,'max_depth':hp.quniform("max_depth",5,20,1)
                    ,'min_child_samples':hp.quniform("min_child_samples",1,3,1)
                    ,'n_estimators':hp.quniform("n_estimators",100,250,1)
                    ,'min_child_weight': hp.uniform("min_child_weight",0.001,0.2)#数值不对
                    ,'subsample': hp.uniform("subsample",0.1,0.8)
                    ,'reg_lambda':hp.uniform("reg_lambda",0.1,0.6)
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
                    ,early_stop_fn=early_stop_fn
                    )
    print("\n","\n","best params:",params_best,"\n")
    return params_best,trials
import time
start=time.time()
params_best,trials=param_hypeopt(500)
print("时间:",time.time()-start)
#GBDT
def hypeopt_objective(params):
    reg=GradientBoostingRegressor(min_samples_leaf=int(params["min_samples_leaf"])
            ,max_depth=int(params["max_depth"])
            ,learning_rate=params["learning_rate"]
            ,n_estimators=int(params["n_estimators"])
            ,random_state=2024
            ).fit(x_train,y_train)
    cv=KFold(n_splits=5,shuffle=True,random_state=2024)
    validation_loss=cross_validate(reg
                                   ,x_train,y_train
                                   ,cv=cv
                                   ,verbose=False
                                   ,n_jobs=-1
                                   ,error_score='raise'
                                    )
    return -np.mean(validation_loss["train_score"])
param_grid_simple={
                    'min_samples_leaf':hp.quniform("min_samples_leaf",3,7,1)#
                    ,"learning_rate": hp.uniform("learning_rate",0.001,0.3)#
                    ,'max_depth':hp.quniform("max_depth",5,15,1)#
                    ,'n_estimators':hp.quniform("n_estimators",200,600,1)
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
                    ,early_stop_fn=early_stop_fn
                    )
    print("\n","\n","best params:",params_best,"\n")
    return params_best,trials
import time
start=time.time()
params_best,trials=param_hypeopt(500)
print("时间:",time.time()-start)