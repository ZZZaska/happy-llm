# ==============================================================================
# 🚀 Qwen-1.5B 分布式预训练启动脚本 (PyTorch DDP 原生稳定版)
# 
# 核心变化:
# 彻底抛弃了 DeepSpeed，使用 PyTorch 原生的 torchrun 启动 DDP 训练。
# 没有任何奇奇怪怪的 Hook，绝不报 NoneType 错误！
# ==============================================================================

CUDA_VISIBLE_DEVICES=0,1

# 使用 torchrun 替代 deepspeed，--nproc_per_node=2 代表使用 2 张卡
torchrun --nproc_per_node=2 pretrain.py \
    --config_name /root/autodl-tmp/Qwen2.5-1.5B \
    --tokenizer_name /root/autodl-tmp/Qwen2.5-1.5B \
    --train_files /root/autodl-tmp/datasets/pretrain/mobvoi_small_clean.jsonl \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --do_train \
    --output_dir /root/autodl-tmp/output/pretrain \
    --evaluation_strategy  no \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --warmup_ratio 0.05 \
    --logging_dir /root/autodl-tmp/output/pretrain/logs \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 100 \
    --preprocessing_num_workers 10 \
    --save_total_limit 1 \
    --seed 12 \
    --block_size 2048 \
    --bf16 \
    --report_to none