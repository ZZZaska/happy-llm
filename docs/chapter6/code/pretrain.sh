# ==============================================================================
# 🚀 Qwen-1.5B 分布式预训练启动脚本 (双卡满血版)
# 
# 核心配置说明:
# 1. 硬件环境: 双卡 RTX 3090 （24GB显存） 
# 2. 显存优化: DeepSpeed ZeRO-2 + bf16 + 梯度检查点
# 3. 批次大小: 8(单卡BS) * 8(梯度累积) * 2(GPU数) = 128 (Global Batch Size)
# ==================================================================


# 1. 设置GPU环境变量
#    ↓
# 2. DeepSpeed启动 pretrain.py
#    ↓
# 3. pretrain.py接收所有参数
#    ↓
# 4. 加载模型配置和tokenizer
#    ↓
# 5. 加载训练数据
#    ↓
# 6. 初始化Trainer
#    ↓
# 7. 开始分布式训练
#    ↓
# 8. 定期保存checkpoint和日志
#    ↓
# 9. 训练完成，保存最终模型

CUDA_VISIBLE_DEVICES=0,1

deepspeed pretrain.py \
    --config_name /root/autodl-tmp/Qwen2.5-1.5B \
    --tokenizer_name /root/autodl-tmp/Qwen2.5-1.5B \
    --train_files /root/autodl-tmp/datasets/pretrain/mobvoi_small_clean.jsonl \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
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
    --gradient_checkpointing True \
    --deepspeed "./ds_config_zero3.json" \
    --report_to none \
    
    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \