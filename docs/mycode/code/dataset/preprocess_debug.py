""" 抽样
从原始数据集中抽样(small.jsonl)用来调试训练
"""

# import json

# fw = open("autodl-tmp/dataset/pretrain_data/mobvoi_seq_monkey_general_open_corpus_small.jsonl", "w")
# i = 0
# with open("autodl-tmp/dataset/pretrain_data/mobvoi_seq_monkey_general_open_corpus.jsonl", "r") as f:
#     while i <= 1000000:
#         line = f.readline()
#         fw.write(line)
#         i += 1
# fw.close()

import json
import os

input_file = "/Volumes/T7/datasets/pretrain/mobvoi_seq_monkey_general_open_corpus.jsonl"
output_file = "/Volumes/T7/datasets/pretrain/mobvoi_small.jsonl"

output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    print(f"📂 文件夹不存在，正在创建: {output_dir}")
    os.makedirs(output_dir, exist_ok=True) 



# 读取文件input_file, i.e, fr
# 新建文件output_file,i.e,fw
# 参数 errors="ignore": 遇到坏字节直接扔掉,不然程序卡住
# 结束之后两个都关闭
max_lines = 1000000   
with open(input_file, "r", encoding="utf-8",errors="ignore") as fr, \
    open(output_file, "w", encoding="utf-8",errors="ignore") as fw:

    for i, line in enumerate(fr):
        if i >= max_lines:
            break
        fw.write(line)

print(f"✅ 抽样完成，共 {max_lines} 条")
