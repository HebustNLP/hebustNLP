import os
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import AutoTokenizer, AutoModelForCausalLM
  
# 设置 Hugging Face 的镜像源（可选，用于国内网络加速）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 可选的镜像源

# 设置模型下载的缓存目录为 'autodl-tmp'
cache_dir = os.path.expanduser('/root/autodl-tmp/shiyan/model')

# tokenizer = AutoTokenizer.from_pretrained("THUDM/GLM-4-9B-0414",cache_dir=cache_dir)
# model = AutoModelForCausalLM.from_pretrained("THUDM/GLM-4-9B-0414",cache_dir=cache_dir)
from transformers import AutoModel, AutoTokenizer

# 下载并加载 bert-base-chinese 模型和分词器
model = AutoModel.from_pretrained("bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 保存模型和分词器到本地
model.save_pretrained("./bert-base-chinese",cache_dir=cache_dir)
tokenizer.save_pretrained("./bert-base-chinese",cache_dir=cache_dir)

print("BERT-base-chinese 模型和分词器已下载并保存到本地 ./bert-base-chinese 目录")






# import os
# from huggingface_hub import snapshot_download


# def try_download(repo_id: str, revision: str, local_dir: str, cache_dir: str, use_mirror: bool) -> str:
#     # 关闭离线模式，按需启用镜像
#     os.environ.pop("HF_HUB_OFFLINE", None)
#     os.environ.pop("TRANSFORMERS_OFFLINE", None)
#     if use_mirror:
#         os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
#     else:
#         os.environ.pop("HF_ENDPOINT", None)

#     # 若安装了 hf_transfer，将启用更快的传输
#     os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

#     os.makedirs(local_dir, exist_ok=True)
#     os.makedirs(cache_dir, exist_ok=True)

#     snapshot_path = snapshot_download(
#         repo_id=repo_id,
#         revision=revision,
#         cache_dir=cache_dir,
#         local_dir=local_dir,
#         local_dir_use_symlinks=False,
#         resume_download=True,
#         local_files_only=False,
#     )
#     return snapshot_path


# def main() -> None:
#     repo_id = "bert-base-cased"
#     revision = "main"
#     local_dir = "/root/autodl-tmp/pretrain_models/bert-base-cased"
#     cache_dir = "/root/autodl-tmp/hf-cache"

#     try:
#         snapshot_path = try_download(repo_id, revision, local_dir, cache_dir, use_mirror=False)
#     except Exception as e:
#         print(f"直连失败，将使用镜像重试。原因：{e}")
#         snapshot_path = try_download(repo_id, revision, local_dir, cache_dir, use_mirror=True)

#     print("模型下载完成：", local_dir)
#     print("快照缓存位置：", snapshot_path)


# if __name__ == "__main__":
#     main()
