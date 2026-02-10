import os
import json
from tqdm import tqdm

# 远程数据集的下载、本地解压
# 使用 os.system 调用 ModelScope官方CLI进行数据下载
os.system(
    "modelscope download --dataset ddzhu123/seq-monkey " # 仓库名:发布者用户名+数据集名
    "mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 "  # 下载仓库中指定文件,压缩包形式
    "--local_dir /Volumes/T7/datasets/pretrain"
)

os.system(
    "tar -xvjf /Volumes/T7/datasets/pretrain/"
    "mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 "
    "-C /Volumes/T7/datasets/pretrain"
)

