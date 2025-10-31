# 创建环境并安装依赖
conda env create -f environment.yml
# 激活环境并下载数据集和模型
conda activate myenv  
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
# 生成对比专家模型
./run.sh    
# 生成权重
export MODEL_NAME_1="output/模型1名称" 

export MODEL_NAME_2="output/模型2名称"   

bash script/GranularTokenScoring.sh          
# 使用生成权重进行训练
./run_token.sh  
