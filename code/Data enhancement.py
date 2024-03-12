import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde
import smogn
df=pd.read_excel(r"D:\桌面文件\研一上\学习笔记\机器学习\jupyter notebook save path\data\Dmax1527.xlsx")
X1=df.iloc[:,:3].values
Y1=df.iloc[:,-1:].values
#KDE-SMOTE
def kde_smote_oversampling(X, y, k, ratio=1.0, sigma=0.5):
    """
    使用KDE-SMOTE算法对连续数据进行过采样
    参数：
    X：特征数据集，numpy数组
    y：标签数据集，numpy数组
    k：选择近邻样本的数量
    ratio：过采样比例，默认为1.0（即平衡类别）
    sigma：高斯核密度估计的带宽参数
    返回值：
    X_resampled：过采样后的特征数据集，numpy数组
    y_resampled：过采样后的标签数据集，numpy数组
    """
    X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    neighbors = NearestNeighbors(n_neighbors=k+1).fit(X_normalized)
    _, indices = neighbors.kneighbors(X_normalized)
    num_new_samples = int(len(X) * ratio) - len(X)
    new_samples = []
    for i in range(num_new_samples):
        idx = np.random.randint(0, len(X))
        neighbor_idx = np.random.choice(indices[idx, 1:])
        diff = X_normalized[neighbor_idx] - X_normalized[idx]
        kde = gaussian_kde(X_normalized.T, bw_method=sigma)
        new_sample = X_normalized[idx] + kde.resample(size=1)[0] * diff
        new_samples.append(new_sample)
    X_resampled = np.concatenate((X, np.array(new_samples)), axis=0)
    y_resampled = np.concatenate((y, np.array([y[idx]] * num_new_samples)), axis=0)
    return X_resampled, y_resampled
X_resampled, y_resampled = kde_smote_oversampling(X1, Y1, k=3, ratio=2.0, sigma=0.5)
df_features = pd.DataFrame(X_resampled, columns=['Tg', 'Tx', 'Tl'])
df_labels = pd.DataFrame(y_resampled, columns=['Dmax'])
df_combined = pd.concat([df_features, df_labels], axis=1)
df_combined.to_excel('KDE_smote.xlsx', index=False)
#SMOGN
df2=smogn.smoter(data=df,y="Dmax",k=4)
df2.to_excel('smogn.xlsx', index=False)
def PCD(XR, XS):
    """
    计算成对相关差异（PCD）指标
    参数：
    XR：真实数据集，numpy数组类型，形状为(N, M)，N为样本数，M为变量数。
    XS：合成数据集，numpy数组类型，形状为(N, M)，N为样本数，M为变量数。
    返回值：
    PCD值，浮点数类型。
    """
    corr_XR = np.corrcoef(XR.T)
    corr_XS = np.corrcoef(XS.T)
    diff_corr = corr_XR - corr_XS
    PCD = np.linalg.norm(diff_corr, 'fro')
    return PCD
df=pd.read_excel(r"D:\桌面文件\研一上\学习笔记\机器学习\jupyter notebook save path\data\Dmax1527.xlsx")
df=df.values
df1=pd.read_excel("KDE_smote.xlsx")
df1=df1.values
df2=pd.read_excel("smogn.xlsx")
df2=df2.values
# 计算成对相关差异（PCD）
pcd1=PCD(df, df1)
print("KDE_smote的PCD值为",pcd1)
smogn=PCD(df, df2)
print("Smogn的PCD值为",smogn)