import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import os, sys
import tqdm

os.chdir(sys.path[0])  # 确保当前工作目录就是Python脚本所在路径

# 前15组特征
feats15 = ['Q1', 'Q3_2', 'Q8_3', 'Q12_3', 'Q9', 'Q7', 'Q3_3', 'Q2_1', 'Q5_1', 'Q2_4', 'Q12_4', 'Q10', 'Q11', 'Q15', 'sex']
# 前19组特征
feats19 = ['Q1', 'Q3_2', 'Q8_3', 'Q12_3', 'Q9', 'Q7', 'Q3_3', 'Q2_1', 'Q5_1', 'Q2_4', 'Q12_4', 'Q10', 'Q11', 'Q15', 'sex', 'Q14_1', 'Q8_5', 'Q8_2', 'age']
# 全部特征
df = pd.read_csv("/home/zhangxiaohan/constitution/data/constitution_threecl_fourmodal.csv")
df = df.drop(columns=["ID", "face_number","tongue_number","pulse_number"])


# 提取特征和标签

# 15组特征
# features = df[feats15]  # 1084×15 

# 19组特征
features = df[feats19]  # 1084×19 

# 全部特征
# features = df.drop(columns=["constitution"])

# 标签
label_constitution = df["constitution"]  # constitution 气虚标签


# 存储结果的列表
results_constitution = []
results_xueyu = []

# 五折交叉验证
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 定义XGBoost参数
params = {
    'objective': 'binary:logistic',  # objective参数指定了要优化的损失函数
    'eval_metric': 'logloss',  # eval_metric参数指定了模型评估指标 使用对数损失logloss作为评估指标
}

for train_index, test_index in skf.split(features, label_constitution):
    # 划分训练集和测试集
    X_train_constitution, X_test_constitution = features.iloc[train_index], features.iloc[test_index]  #  features 1084×15 
    y_train_constitution, y_test_constitution = label_constitution.iloc[train_index], label_constitution.iloc[test_index]

    # 训练和预测
    model_constitution = xgb.XGBClassifier(**params)
    model_constitution.fit(X_train_constitution, y_train_constitution)
    y_pred_constitution = model_constitution.predict(X_test_constitution)  # y_test_constitution,测试集的真实标签值  y_pred_constitution,测试集的预测标签值

    # 计算评估指标
    acc_constitution = accuracy_score(y_test_constitution, y_pred_constitution)
    f1_constitution = f1_score(y_test_constitution, y_pred_constitution,average='weighted')  #  average='weighted'试一下  [None, 'micro', 'macro', 'weighted']

    # 存储结果
    results_constitution.append([acc_constitution, f1_constitution])

# 将结果保存到CSV文件
df_constitution = pd.DataFrame(results_constitution, columns=["acc", "f1"])
df_constitution.to_csv("/home/zhangxiaohan/constitution/tizhi/data/ask/results/ask_feats19_zxh.csv", index=False)