""" 数据预处理(offline, onetime)
将原始的、非标准的大规模语料文件清洗并转换为 PyTorch Dataset 类可直接高效读取的标准 JSONL 格式

功能:
- streaming
- robust cleaning
- chunking

input: jsonl (每行包含 'text' 字段)
output: jsonl (每行 {"text": "..."}，长度 <= chunk_size)
"""
import os
import json
from tqdm import tqdm

def split_text(text: str, chunk_size: int = 512):
    # 将长文本切分为固定长度的 chunk
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def preprocess_pretrain_data(input_file: str, output_file: str, chunk_size: int = 512):
    with open(output_file, "w", encoding="utf-8") as fw:   
        with open(input_file, "r", encoding="utf-8") as fr:  
            for i,line in enumerate(tqdm(fr, desc="processing")):   
                try: 
                    # 逐行读取 + 解析json +空文本过滤
                    sample = json.loads(line)
                    text = sample.get("text", "")
                    if not text.strip(): 
                        continue
                    
                    # 切分并回写
                    chunks = split_text(text, chunk_size) 
                    for chunk in chunks:
                        fw.write(
                            json.dumps({"text": chunk}, ensure_ascii=False) + "\n"
                        )
                        
                except json.JSONDecodeError:
                    print(f"⚠️ 警告：第 {i} 行数据格式错误，已跳过。")
                    continue
                except Exception as e:
                    print(f"❌ 未知错误：{e}")
                    continue

