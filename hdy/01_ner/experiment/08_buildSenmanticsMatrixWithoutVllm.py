import jieba
import numpy as np
import regex as re
import json
from tqdm import tqdm
from functools import lru_cache
import time

from transformers import AutoTokenizer, AutoModelForCausalLM

# ========== 加载模型 ==========
model_path = "/root/autodl-tmp/shiyan/model/models--THUDM--GLM-4-9B-0414/snapshots/645b8482494e31b6b752272bf7f7f273ef0f3caf"

print("正在加载模型...")
start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto"  # 自动分配到 GPU 或 CPU
)

print(f"模型加载完成，耗时: {time.time() - start_time:.2f}秒")

# ========== 工具函数 ==========
def parse_json_from_text(text):
    """从文本中解析JSON"""
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

def normalize_relations(rel_raw):
    """标准化关系字典"""
    out = {}
    if not isinstance(rel_raw, dict):
        return out
    for k, v in rel_raw.items():
        if isinstance(k, str):
            mm = re.match(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", k)
            if mm:
                i, j = int(mm.group(1)), int(mm.group(2))
                out[(i, j)] = v
        elif isinstance(k, (list, tuple)) and len(k) >= 2:
            out[(int(k[0]), int(k[1]))] = v
    return out

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
    """创建提示词模板"""
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

def model_generate(prompt, max_new_tokens=256, temperature=0.1, top_p=1.0):
    """使用transformers模型生成"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@lru_cache(maxsize=1000)
def analyze_single_sentence(sentence):
    """分析单个句子（带缓存）"""
    words = tuple(jieba.cut(sentence))
    matrix_size = len(words)
    
    prompt = create_prompt(sentence, list(words))
    
    result_text = model_generate(prompt)
    
    parsed = parse_json_from_text(result_text)
    
    if parsed:
        matrix_raw = parsed.get("matrix")
        relations_raw = parsed.get("relations")
        try:
            mat = np.array(matrix_raw, dtype=int)
            if mat.shape == (matrix_size, matrix_size):
                return list(words), mat.tolist(), relations_raw
        except Exception as e:
            print(f"解析矩阵失败: {e}")
    
    M, rels = heuristic_dep_matrix(list(words))
    return list(words), M.tolist(), {str(k): v for k, v in rels.items()}

def batch_analyze_sentences(sentences, batch_size=4):
    """批量处理句子"""
    all_results = []
    
    for i in tqdm(range(0, len(sentences), batch_size), desc="批量处理"):
        batch_sentences = sentences[i:i+batch_size]
        
        for sentence in batch_sentences:
            words = list(jieba.cut(sentence))
            prompt = create_prompt(sentence, words)
            result_text = model_generate(prompt)
            
            matrix_size = len(words)
            semantic_matrix = None
            relations = None

            parsed = parse_json_from_text(result_text)
            if parsed:
                matrix_raw = parsed.get("matrix")
                relations_raw = parsed.get("relations")
                try:
                    mat = np.array(matrix_raw, dtype=int)
                    if mat.shape == (matrix_size, matrix_size):
                        semantic_matrix = mat.tolist()
                        relations = relations_raw
                except Exception as e:
                    print(f"批次解析失败: {e}")
            
            if semantic_matrix is None:
                M, rels = heuristic_dep_matrix(words)
                semantic_matrix = M.tolist()
                relations = {str(k): v for k, v in rels.items()}
            
            all_results.append((words, semantic_matrix, relations))
    
    return all_results

# ========== 主逻辑 ==========
def main():
    print("开始处理数据集...")
    
    with open("/root/autodl-tmp/shiyan/dataset/result/100.json", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]
    
    print(f"数据集大小: {len(dataset)} 条")
    
    sentences = [item["text"] for item in dataset]
    
    if len(dataset) > 1000:
        print("使用批量处理模式...")
        results = batch_analyze_sentences(sentences, batch_size=4)
    else:
        print("使用单条处理模式（带缓存）...")
        results = []
        for sentence in tqdm(sentences, desc="处理句子"):
            words, matrix, relations = analyze_single_sentence(sentence)
            results.append((words, matrix, relations))
    
    new_dataset = []
    success_count = 0
    
    for idx, (item, (words, matrix, relations)) in enumerate(zip(dataset, results)):
        if len(words) != len(matrix):
            print(f"警告: 样本 {idx+1} 的矩阵尺寸不匹配")
            words = list(jieba.cut(item["text"]))
            M, rels = heuristic_dep_matrix(words)
            matrix = M.tolist()
            relations = {str(k): v for k, v in rels.items()}
        else:
            success_count += 1
        
        item["words"] = words
        item["semantic_matrix"] = matrix
        item["relations"] = relations
        new_dataset.append(item)
        
        if (idx + 1) % 100 == 0:
            print(f"\n=== 样本 {idx+1} ===")
            print("原文:", item["text"])
            print("分词长度:", len(words))
            print("矩阵尺寸:", len(matrix), "x", len(matrix[0]) if matrix else 0)
    
    print(f"\n处理完成，成功解析: {success_count}/{len(dataset)}")
    
    output_path = "/root/autodl-tmp/shiyan/dataset/result/test.json"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in new_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ 结果已保存到: {output_path}")

if __name__ == "__main__":
    main()
