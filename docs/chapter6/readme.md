# 第六章 基于 transformers 的 LLM 训练

注：本章的核心内容是，基于 transformers 框架实现 LLM 预训练和微调

1. 框架简述：
   1. transformers
   2. deepspeed
   3. peft
   4. wandb
   5. tokenizers
2. 基于 transformers 的 LLM 预训练
   1. 分词器训练
   2. 数据集构建
   3. 模型搭建/继承预训练模型
   4. 构造 Trainer 进行训练
3. 基于 transformers 的 LLM SFT/下游任务微调
   1. 分词器训练
   2. 数据集构建
   3. LoRA 配置
   4. 继承预训练模型
   5. 构造 Trainer 进行训练


   第六章：工业级训练实践  
├── 6.1 预训练（基础训练）  
│   ├── 框架介绍 → 工具准备  
│   ├── 初始化LLM → 模型准备  
│   ├── 数据处理 → 数据准备   
│   ├── Trainer训练 → 单机训练  
│   └── DeepSpeed → 分布式训练
│    
├── 6.2 SFT（指令微调）  
│   ├── Pretrain vs SFT → 理解差异  
│   └── 数据处理 → 核心实现    
│  
└── 6.3 高效微调（资源优化）  
    ├── 方案对比 → 选择依据   
    ├── LoRA原理 → 理论基础   
    ├── 代码实现 → 底层逻辑   
    └── peft使用 → 实践应用   