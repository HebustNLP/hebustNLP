# import jieba
# import numpy as np
# # 优先使用 regex，支持 \p{P}\p{S}，否则 fallback 到内置 re
# try:
#     import regex as re
# except Exception:
#     import re
# import json
# from tqdm import tqdm
# from functools import lru_cache
# import time
# import string

# from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer


# # ========== 加载模型 ==========
# model_path = "/root/autodl-tmp/shiyan/model/models--THUDM--GLM-4-9B-0414/snapshots/645b8482494e31b6b752272bf7f7f273ef0f3caf"

# print("正在加载模型...")
# start_time = time.time()

# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# llm = LLM(
#     model=model_path,
#     trust_remote_code=True,
#     tensor_parallel_size=1   # 如果有多卡，可以改成 GPU 数量
# )

# print(f"模型加载完成，耗时: {time.time() - start_time:.2f}秒")


# # ========== 工具函数 ==========
# def normalize_words(raw_words):
#     """清理分词结果，去掉空白和标点符号"""
#     cleaned = []
#     try:
#         punct_re = re.compile(r'^[\s\p{P}\p{S}]+$')
#     except Exception:
#         puncts = string.punctuation + "，。！？、；：‘’“”（）【】《》——…、·"
#         punct_re = re.compile(rf'^[\s{re.escape(puncts)}]+$')

#     for w in raw_words:
#         if w.strip() == "":
#             continue
#         if punct_re.fullmatch(w):
#             continue
#         cleaned.append(w)
#     return cleaned


# def parse_json_from_text(text):
#     """从模型输出中解析 JSON"""
#     patterns = [
#         r"(\{[\s\S]*?\})",
#         r"```json\n([\s\S]*?)\n```",
#         r"```\n([\s\S]*?)\n```"
#     ]
#     for pattern in patterns:
#         m = re.search(pattern, text)
#         if m:
#             try:
#                 return json.loads(m.group(1))
#             except json.JSONDecodeError:
#                 try:
#                     json_str = m.group(1).replace("'", '"')
#                     json_str = re.sub(r',\s*}', '}', json_str)
#                     json_str = re.sub(r',\s*]', ']', json_str)
#                     return json.loads(json_str)
#                 except:
#                     continue
#     return None


# def heuristic_dep_matrix(words):
#     """启发式依赖矩阵生成（备用方案）"""
#     n = len(words)
#     M = np.eye(n, dtype=int)
#     relations = {}

#     if n >= 2:
#         M[0, 1] = M[1, 0] = 1
#         relations[(0, 1)] = "主谓"

#     for prep in ["在", "从", "到", "向", "对", "关于", "对于"]:
#         if prep in words:
#             i = words.index(prep)
#             if i + 1 < n:
#                 M[i, i + 1] = M[i + 1, i] = 1
#                 relations[(i, i + 1)] = "介宾"
#             if i - 1 >= 0:
#                 M[i - 1, i] = M[i, i - 1] = 1
#                 relations[(i - 1, i)] = "动词-介词"

#     for i in range(n - 1):
#         if (i, i + 1) not in relations:
#             M[i, i + 1] = M[i + 1, i] = 1
#             relations[(i, i + 1)] = "依存"

#     return M, relations


# def create_prompt(sentence, words):
#     """构建提示词"""
#     matrix_size = len(words)
#     return f"""你是一个语义分析专家。请分析以下句子的语义关系：

# 句子: "{sentence}"
# 分词: {words}

# 要求：
# 1. 输出一个严格的JSON对象，格式如下：
# {{"matrix": [[整数矩阵]], "relations": {{"(i,j)": "关系名"}}}}
# 2. matrix必须是{matrix_size}x{matrix_size}的对称矩阵，对角线为1，其他位置0或1
# 3. relations用中文描述词对关系
# 4. 只输出JSON，不要其他内容

# JSON输出："""


# # vLLM 生成参数
# sampling_params = SamplingParams(
#     temperature=0.1,
#     top_p=1.0,
#     max_tokens=256,
#     repetition_penalty=1.1
# )


# @lru_cache(maxsize=1000)
# def analyze_single_sentence(sentence):
#     """分析单个句子"""
#     raw_words = tuple(jieba.cut(sentence))
#     words = normalize_words(raw_words)
#     matrix_size = len(words)

#     prompt = create_prompt(sentence, list(words))
#     outputs = llm.generate([prompt], sampling_params)
#     result_text = outputs[0].outputs[0].text

#     parsed = parse_json_from_text(result_text)
#     if parsed:
#         matrix_raw = parsed.get("matrix")
#         relations_raw = parsed.get("relations")
#         try:
#             mat = np.array(matrix_raw, dtype=int)
#             if mat.shape == (matrix_size, matrix_size):
#                 return list(words), mat.tolist(), relations_raw
#         except Exception as e:
#             print(f"解析矩阵失败: {e}")

#     # 解析失败 → 启发式补全
#     M, rels = heuristic_dep_matrix(list(words))
#     return list(words), M.tolist(), {str(k): v for k, v in rels.items()}


# def main():
#     print("开始处理数据集...")

#     input_path = "/root/autodl-tmp/shiyan/dataset/cluener_public/train.json"
#     output_path = "/root/autodl-tmp/shiyan/dataset/cluener_public/fullSample_process/train.json"

#     with open(input_path, "r", encoding="utf-8") as f:
#         dataset = [json.loads(line) for line in f]

#     print(f"数据集大小: {len(dataset)} 条")

#     sentences = [item["text"] for item in dataset]

#     # 第一阶段：分析句子
#     results = []
#     for sentence in tqdm(sentences, desc="处理句子"):
#         words, matrix, relations = analyze_single_sentence(sentence)
#         results.append((words, matrix, relations))

#     # 第二阶段：整合结果
#     new_dataset = []
#     success_count = 0

#     for idx, (item, (words, matrix, relations)) in tqdm(
#         enumerate(zip(dataset, results)), total=len(results), desc="整合结果"
#     ):
#         if len(words) != len(matrix):
#             print(f"警告: 样本 {idx+1} 的矩阵尺寸不匹配")
#             words = list(jieba.cut(item["text"]))
#             M, rels = heuristic_dep_matrix(words)
#             matrix = M.tolist()
#             relations = {str(k): v for k, v in rels.items()}
#         else:
#             success_count += 1

#         # 🔑 保留原始格式 + 新增语义矩阵信息
#         item["words"] = words
#         item["semantic_matrix"] = matrix
#         item["relations"] = relations
#         new_dataset.append(item)

#         if (idx + 1) % 100 == 0:
#             print(f"\n=== 样本 {idx+1} ===")
#             print("原文:", item["text"])
#             print("分词长度:", len(words))
#             print("矩阵尺寸:", len(matrix), "x", len(matrix[0]) if matrix else 0)

#     print(f"\n处理完成，成功解析: {success_count}/{len(dataset)}")

#     # 第三阶段：保存结果
#     with open(output_path, "w", encoding="utf-8") as f:
#         for item in tqdm(new_dataset, desc="写入文件"):
#             f.write(json.dumps(item, ensure_ascii=False) + "\n")

#     print(f"✅ 结果已保存到: {output_path}")


# if __name__ == "__main__":
#     main()


import jieba
import numpy as np
try:
    import regex as re
except Exception:
    import re
import json
from tqdm import tqdm
import time
import string

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ========== 加载模型 ==========
model_path = "/root/autodl-tmp/shiyan/model/models--THUDM--GLM-4-9B-0414/snapshots/645b8482494e31b6b752272bf7f7f273ef0f3caf"

print("正在加载模型...")
start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    tensor_parallel_size=1   # 多卡可改成 GPU 数量
)

print(f"模型加载完成，耗时: {time.time() - start_time:.2f}秒")

# ========== 工具函数 ==========
def normalize_words(raw_words):
    """清理分词结果，去掉空白和标点符号"""
    cleaned = []
    try:
        punct_re = re.compile(r'^[\s\p{P}\p{S}]+$')
    except Exception:
        puncts = string.punctuation + "，。！？、；：‘’“”（）【】《》——…、·"
        punct_re = re.compile(rf'^[\s{re.escape(puncts)}]+$')

    for w in raw_words:
        if w.strip() == "":
            continue
        if punct_re.fullmatch(w):
            continue
        cleaned.append(w)
    return cleaned


def parse_json_from_text(text):
    """从模型输出中解析 JSON"""
    patterns = [
        r"(\{[\s\S]*?\})",
        r"```json\n([\s\S]*?)\n```",
        r"```\n([\s\S]*?)\n```"
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                try:
                    json_str = m.group(1).replace("'", '"')
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    return json.loads(json_str)
                except:
                    continue
    return None


def heuristic_dep_matrix(words):
    """启发式依赖矩阵生成（备用方案）"""
    n = len(words)
    M = np.eye(n, dtype=int)
    relations = {}

    if n >= 2:
        M[0, 1] = M[1, 0] = 1
        relations[(0, 1)] = "主谓"

    for prep in ["在", "从", "到", "向", "对", "关于", "对于"]:
        if prep in words:
            i = words.index(prep)
            if i + 1 < n:
                M[i, i + 1] = M[i + 1, i] = 1
                relations[(i, i + 1)] = "介宾"
            if i - 1 >= 0:
                M[i - 1, i] = M[i, i - 1] = 1
                relations[(i - 1, i)] = "动词-介词"

    for i in range(n - 1):
        if (i, i + 1) not in relations:
            M[i, i + 1] = M[i + 1, i] = 1
            relations[(i, i + 1)] = "依存"

    return M, relations


def create_prompt(sentence, words):
    """构建提示词"""
    matrix_size = len(words)
    return f"""你是一个语义分析专家。请分析以下句子的语义关系：

句子: "{sentence}"
分词: {words}

要求：
1. 输出一个严格的JSON对象，格式如下：
{{"matrix": [[整数矩阵]], "relations": {{"(i,j)": "关系名"}}}}
2. matrix必须是{matrix_size}x{matrix_size}的对称矩阵，对角线为1，其他位置0或1
3. relations用中文描述词对关系
4. 只输出JSON，不要其他内容


JSON输出："""


# vLLM 生成参数
sampling_params = SamplingParams(
    temperature=0.1,
    top_p=1.0,
    max_tokens=256,
    repetition_penalty=1.1
)


def main():
    print("开始处理数据集...")

    input_path = "/root/autodl-tmp/shiyan/dataset/cluener_public/train.json"
    output_path = "/root/autodl-tmp/shiyan/dataset/cluener_public/fullSample_process/train.json"

    with open(input_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    print(f"数据集大小: {len(dataset)} 条")

    sentences = [item["text"] for item in dataset]

    # 🔥 第一步：批量构建 prompts
    prompts = []
    all_words = []
    for sentence in sentences:
        raw_words = tuple(jieba.cut(sentence))
        words = normalize_words(raw_words)
        all_words.append(words)
        prompts.append(create_prompt(sentence, words))

    # 🔥 第二步：一次性调用 vLLM 批量推理
    print("正在使用 vLLM 批量推理...")
    outputs = llm.generate(prompts, sampling_params)

    # 🔥 第三步：解析输出
    results = []
    for words, out in zip(all_words, outputs):
        text = out.outputs[0].text
        parsed = parse_json_from_text(text)

        if parsed:
            try:
                mat = np.array(parsed.get("matrix"), dtype=int)
                if mat.shape == (len(words), len(words)):
                    results.append((words, mat.tolist(), parsed.get("relations")))
                    continue
            except:
                pass

        # 失败 fallback
        M, rels = heuristic_dep_matrix(words)
        results.append((words, M.tolist(), {str(k): v for k, v in rels.items()}))

    # 🔥 第四步：整合结果
    new_dataset = []
    success_count = 0

    for idx, (item, (words, matrix, relations)) in tqdm(
        enumerate(zip(dataset, results)), total=len(results), desc="整合结果"
    ):
        if len(words) == len(matrix):
            success_count += 1
        else:
            print(f"警告: 样本 {idx+1} 的矩阵尺寸不匹配")
            words = list(jieba.cut(item["text"]))
            M, rels = heuristic_dep_matrix(words)
            matrix = M.tolist()
            relations = {str(k): v for k, v in rels.items()}

        item["words"] = words
        item["semantic_matrix"] = matrix
        item["relations"] = relations
        new_dataset.append(item)

    print(f"\n处理完成，成功解析: {success_count}/{len(dataset)}")

    # 🔥 第五步：保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        for item in tqdm(new_dataset, desc="写入文件"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
