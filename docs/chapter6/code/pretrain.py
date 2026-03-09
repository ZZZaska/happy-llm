'''
预训练脚本
'''

import logging
import math
import os
import sys
from dataclasses import dataclass, field
# from torchdata.datapipes.iter import IterableWrapper
from itertools import chain
import deepspeed
import contextlib
# =========================================================
# 🧙‍♂️ 强制封印 DeepSpeed 的 no_sync 报错
# 因为 DeepSpeed 自己会处理梯度累积，不需要 Trainer 干涉
# =========================================================
@contextlib.contextmanager
def dummy_no_sync(self):
    yield  # 什么都不做，直接放行
    
# 替换掉底层会报错的那个函数
deepspeed.runtime.engine.DeepSpeedEngine.no_sync = dummy_no_sync
# =========================================================

from typing import Optional,List

import datasets
import pandas as pd
import torch
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
import datetime
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
# import swanlab
from swanlab.integration.huggingface import SwanLabCallback # 不受版本影响的接口


logger = logging.getLogger(__name__)


# 超参类
@dataclass
class ModelArguments:
    """
    规定了模型相关的选项:
        - model_name_or_path: 加载已有模型权重  (继续训练用)
        - config_name: 加载配置文件初始化随机参数 (从零训练用)
        - tokenizer_name: 分词器路径
        - torch_dtype: 数据类型(bfloat16/float16/float32)
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "后训练使用，为预训练模型参数地址"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "预训练使用 Config 文件地址"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "预训练 Tokenizer 地址"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "模型训练使用的数据类型，推荐 bfloat16"
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

# ==============================================================================
# 📦 模块一：定义“点菜单” (接收外部 .sh 脚本传来的参数)
# ==============================================================================

@dataclass
class DataTrainingArguments:
    """
    规定数据集相关的训练参数
        - train_files: Optional[List[str]]   # 训练数据文件路径列表
        - block_size: Optional[int]          # 文本块长度
        - preprocessing_num_workers: Optional[int]  # 数据预处理线程数
    """

    train_files: Optional[List[str]]  = field(default=None, metadata={"help": "训练数据路径"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "设置的文本块长度"
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "预处理使用线程数."},
    )

                
def main():
    #  将pretrain.sh命令行参数解析成三个对象
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments)) 
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

#==============================================================================
# 🕵️‍♂️ 模块二：装上“监控摄像头”与“行车记录仪” (日志与大屏监控)
# ==============================================================================

    # 初始化 SwanLab
    # swanlab.init(project="pretrain", experiment_name="from_scrach")
    swanlab_callback = SwanLabCallback(project="pretrain", experiment_name="from_scrach")
    
    # 设置终端文本日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # 统一日志级别设置为 INFO
    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 检查 checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"输出路径 ({training_args.output_dir}) 非空 "
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"从 {last_checkpoint}恢复训练"
            )

    # 设置随机数种子，保证结果可以复现
    set_seed(training_args.seed)

# ==============================================================================
# 🧠 模块三：召唤“大脑”与“密码本” (模型与分词器初始化)
# ==============================================================================
    # 初始化模型 ： 优先从 config_name 加载，进行随机初始化；如果没有，则从 model_name_or_path 加载预训练权重
    if model_args.config_name is not None:
        # from scrach
        config = AutoConfig.from_pretrained(model_args.config_name)
        logger.warning("你正在从零初始化一个模型")
        logger.info(f"模型参数配置地址：{model_args.config_name}")
        logger.info(f"模型参数：{config}")
        model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"预训练一个新模型 - Total size={n_params/2**20:.2f}M params")
    elif model_args.model_name_or_path is not None:
        logger.warning("你正在初始化一个预训练模型")
        logger.info(f"模型参数地址：{model_args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,trust_remote_code=True)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"继承一个预训练模型 - Total size={n_params/2**20:.2f}M params")
    else:
        logger.error("config_name 和 model_name_or_path 不能均为空")
        raise ValueError("config_name 和 model_name_or_path 不能均为空")

    # 防止 DeepSpeed + Checkpointing 导致计算图断裂
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    
    # 初始化 Tokenizer 分词器
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    logger.info("完成 tokenzier 加载")
    logger.info(f"tokenzier 配置地址：{model_args.tokenizer_name}")

# ==============================================================================
# 🔪 模块四：数据预处理的“暴力美学” (查字典与无缝切块)
# ==============================================================================
    # 加载预训练数据 ds,然后使用加载好的tokenizer中 map函数对数据集 ds 进行处理，最后进行文本切块
    ds = load_dataset('json', data_files=data_args.train_files) # ds: Huggingface Datasetdict
    logger.info("完成训练集加载")
    logger.info(f"训练集地址：{data_args.train_files}")
    logger.info(f'训练文件总数:{len(ds["train"])}')
    # logger.info(f"训练集采样：{ds["train"][0]}")
    column_names = list(ds["train"].features)
    logger.info(f'训练集特征：{column_names}')
    text_column_name = "text" if "text" in column_names else column_names[0] # 有则使用text列，否则使用第一列
    
    """
    仅主进程数据预处理：
        - 定义 tokenize_function 函数，使用 tokenizer 对文本进行编码
        - map 函数加载数据集并行处理

    输入数据集ds, 处理完成后数据集格式：分别是文本 tokenize 之后的数值序列和注意力掩码（标识是否 padding)
                {
                "input_ids": [[...], [...], [...]],
                "attention_mask": [[...], [...], [...]]
                }
                
    """
    def tokenize_function(examples): # examples: {"text": [...], ...}  (原始数据集格式)
        output = tokenizer([item for item in examples[text_column_name]])
        return output
    with training_args.main_process_first(desc="dataset map tokenization"): # 防多卡重复
        tokenized_datasets = ds.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers, # 10个CPU线程批量处理
            remove_columns=column_names, # 删除原文本列，防止OOM
            load_from_cache_file=True, # 使用缓存文件加速重复运行
            desc="Running tokenizer on dataset"
        ) 

    """ 
    CLM 任务（根据上下文预测下一个 token),得到训练数据集  train_dataset
    - 拼接文本
    - 按 block_size 长度切块
    """
    # 最终确定训练时使用的文本块长度 block_size
    # - case 1：外部没有指定 block_size，则默认使用 tokenizer 的 model_max_length (最大1024)
    # - case 2: 外部传值 block_size （自定义）
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "tokenizer 支持大于 1K 的上下文长度，默认设置为 1K"
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"设定的块长为 ({data_args.block_size}) ，大于模型的上下文长度"
                f"将块长设置为模型上下文长度：{tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    
    def group_texts(examples):
        # 拼接：将 batch 中的多个文本样本拼接成一个连续的长序列，并且计算总token
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()} # 字典推导式： tokenized_datasets.keys() = ["input_ids", "attention_mask"]
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # 分块：将拼接后到长序列按照block_size分块
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size # 向下取整到 block_size 的整数倍
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items() # 列表推导式
        }    
        # CLM任务中，labels（正确答案） 就是 input_ids 的一个副本（深拷贝），模型会根据 input_ids 预测下一个 token，计算 loss 时会将 labels 与模型输出对比
        result["labels"] = result["input_ids"].copy() # 自监督学习 = 每个block内部自己内部做 next-token prediction
        return result

    with training_args.main_process_first(desc="文本分块"):
        lm_datasets = tokenized_datasets.map( 
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=True,
            desc=f"文本分块到{block_size}",
            batch_size = 40000,
        )
        logger.info("完成数据预处理")
        train_dataset = lm_datasets["train"]
    
# ==============================================================================
# 🚀 模块五：请出“大管家”一键代打 (Trainer 控制显卡与训练循环)
# ==============================================================================    
    # # ZeRO-2：解决 Deepspeed Nonetype 报错问题 -- 强制注入兼容性参数（依然报错
    # if training_args.gradient_checkpointing:
    #     training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    
    # ZeRO-3: 解决 Deepspeed Nonetype 报错问题
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False} 
        model.config.use_cache = False
    
    logger.info("初始化 Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset= train_dataset, # Trainer支持的数据类型，匹配前面预处理得到的 train_dataset（Dataset格式）
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[swanlab_callback] # 集成 SwanLab 监控训练过程
    )

    # 从 checkpoint 加载
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
            checkpoint = last_checkpoint

    logger.info("开始训练")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model() # 保存结果权重到 output_dir 指定的路径

if __name__ == "__main__":
    main()