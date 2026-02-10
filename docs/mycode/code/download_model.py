""" 
huggingface-cli 下载模型到本地  (1-4)
from_pretrained 加载模型 (5)
加载tokenizer (6)
"""
# import os, subprocess,sys

# # 1. 配置下载路径和环境变量
save_dir = "/Volumes/T7/models/Qwen/Qwen2.5-1.5B"
# os.makedirs(save_dir, exist_ok=True)

# env = os.environ.copy()
# # env["HF_ENDPOINT"] = "https://hf-mirror.com"


# cmd = [
#     "huggingface-cli", "download",
#     "--resume-download", "Qwen/Qwen2.5-1.5B", # 支持断点续传
#     "--local-dir", save_dir
# ]

# # 2. 启动异步子进程
# process = subprocess.Popen(
#     cmd,
#     stdout=subprocess.PIPE,  # 开启输出管道
#     stderr=subprocess.STDOUT, # 强制输出流(stdout/stderr)合并
#     text=True,
#     bufsize=1,
#     env=env
# ) 


    
# # 3. 实时监控输出 (解决 EOF 延迟核心逻辑)
# while True:
#     line = process.stdout.readline()
#     if line:
#         print(line, end="")
#         sys.stdout.flush()
#     elif process.poll() is not None:
#         break

# # 4. 善后状态检查
# process.wait()
# if process.returncode != 0:
#     raise RuntimeError("模型下载失败")
# else:
#     print("✅ 模型下载完成")


# 5. 加载一个预训练好的模型
# 读取本地目录 → 解析 config.json → 构建网络 → 加载权重 model.safetensors
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(save_dir,trust_remote_code=True)

print(model)

# 6. 加载一个预训练好的 tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(save_dir,trust_remote_code=True )
