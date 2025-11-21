import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from matplotlib import font_manager

# 路径  /home/sharing/disk1/lisongze/cm/upload/Constitution0410.csv

#  设置随机种子
random_seed = 42
np.random.seed(random_seed)

# 统一字体
# font = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc")
# sns.set(font=font.get_name())

# 读取文件  1084×47
df = pd.read_csv("/home/zhangxiaohan/constitution/data/constitution_threecl_fourmodal.csv") 
df = df.drop(["ID", "face_number","tongue_number","pulse_number"], axis=1)   # 按列删除

# 计算每列的标准差并进行升序排序
std = list(df.std().sort_values().items())

# 分别找出特征中std<=0.4和std>0.4且不包含constitution标签的特征名称
feats_small_std=[]  # 27
feats_great_std=[]  # 15
for i in std:
    if i[1]<=0.4 and i[0].find("constitution")<0: # 不包含constitution返回-1
        feats_small_std.append(i[0])
    if i[1]>0.4 and i[0].find("constitution")<0:
        feats_great_std.append(i[0])

# pearson相关系数[-1,1] 表示负相关不相关和正相关 绝对值越大越相关
df_small_std = df[feats_small_std + ["constitution"]]
df_great_std = df[feats_great_std + ["constitution"]]
plt.figure(figsize=(30, 16))
plt.subplot(2, 1, 1)
sns.heatmap(df_small_std.corr(method="pearson")[["constitution"]].T, annot=True, cmap="coolwarm", fmt=".2f")
plt.subplot(2, 1, 2)
sns.heatmap(df_great_std.corr(method="pearson")[["constitution"]].T, annot=True, cmap="coolwarm", fmt=".2f")
# plt.savefig('heatmap')

# 最终筛选出了19个特征
filtered_feats=feats_great_std+['Q8_3','Q5_1','Q12_3','Q12_4','constitution']  # 1084×20  包含了constitution标签
df_filtered=df[filtered_feats]
# print(df_filtered)


# 使用三种方法处理

# 互信息素
X = df_filtered[[col for col in df_filtered.columns if col.find("constitution") < 0]] # 不包含constitution
Y1 = df_filtered["constitution"] # constitution
name = X.columns # 列名
mi_constitution = mutual_info_classif(X, Y1) # 得到互信息 返回一个列表
mi_constitution = list(zip(name, mi_constitution))
mi_constitution.sort(key=lambda x:x[1], reverse=True)  # 降序排序
# print(mi_constitution)

# 随机森林
rf = RandomForestRegressor(n_estimators=20, max_depth=4) 
kfold = KFold(n_splits=5, shuffle=True, random_state=2023)
scores_constitution = []
for column in X.columns:
    tempx = X[column].values.reshape(-1, 1) 
    score1 = cross_val_score(rf, tempx, Y1, scoring="r2",error_score='raise',
                              cv=kfold) 
    scores_constitution.append((column, round(np.mean(score1), 3))) 
scores_constitution.sort(key=lambda x:x[1], reverse=True)

# XGBoost重要性分析
xgb_constitution = XGBRegressor()  # 创建一个XGBoost实例
xgb_constitution.fit(X, Y1)  # 使用数据集来训练
importance_constitution = list(zip(name, xgb_constitution.feature_importances_))  # 获取特征重要性
importance_constitution.sort(key=lambda x:x[1], reverse=True)

# 将列表转换为dataframe
mi_constitution = pd.Series([col[1] for col in mi_constitution], index=[col[0] for col in mi_constitution])  # 把索引的名字都换了一下
scores_constitution = pd.Series([col[1] for col in scores_constitution], index=[col[0] for col in scores_constitution])
importance_constitution = pd.Series([col[1] for col in importance_constitution], index=[col[0] for col in importance_constitution])
# print(mi_constitution)

# 归一化
def to_one(data):
    max_value = data.max()
    min_value = data.min()
    for i in range(len(data)):
        data.iloc[i] = (data.iloc[i] - min_value) / (max_value - min_value)
to_one(mi_constitution)  # 传过去的是原始数据 所有会值会改变
to_one(scores_constitution)
to_one(importance_constitution)


# 使用三种方法归一化 取平均后作为最终结果进行排序
constitution = []
for n in name:
    constitution.append([n, (mi_constitution[n] + scores_constitution[n] + importance_constitution[n]) / 3])
constitution.sort(key=lambda x:x[1], reverse=True)
constitution = pd.Series([col[1] for col in constitution], index=[col[0] for col in constitution])

print(list(constitution.index[:19]))

"""
前19个特征
['Q1', 'Q3_2', 'Q8_3', 'Q12_3', 'Q9', 'Q7', 'Q3_3', 'Q2_1', 'Q5_1', 'Q2_4', 'Q12_4', 'Q10', 'Q11', 'Q15', 'sex', 'Q14_1', 'Q8_5', 'Q8_2', 'age']
前15个特征
['Q1', 'Q3_2', 'Q8_3', 'Q12_3', 'Q9', 'Q7', 'Q3_3', 'Q2_1', 'Q5_1', 'Q2_4', 'Q12_4', 'Q10', 'Q11', 'Q15', 'sex']
"""