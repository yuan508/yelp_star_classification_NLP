# 方法说明（Methods）

## SimpleTokenizer（自建分词器 / 编码器）

**SimpleTokenizer** 是一个轻量、可复现的文本到 ID 序列转换器，设计目标是对中文/英文短文本进行快速清洗与离线编码，便于在自定义模型（TextCNN / CNN）中使用。

### 功能与流程

- **文本清洗（clean_text）**
  - 小写化（可选），去除/替换标点与控制字符（可配置 `filters`），压缩多重空白为单个空格。
  - 常用于去掉 HTML 残留、URL 或特殊符号。

- **分词**
  - 当 `char_level=False` 时按空格分词（适合英文或已分词的中文）。
  - 当 `char_level=True` 时按字符切分（适合中文单字级处理）。
  - 可扩展为接入 `jieba` / `spaCy` / `HuggingFace tokenizer`。

- **词表构建（fit_on_texts）**
  - 统计语料词频并按频率倒排选择前 `max_words-1`（保留 index 0 为 PAD，1 为 OOV）。
  - 生成 `word_index`（词→id）与 `index_word`（id→词），并导出 `vocab_size`。

- **序列化（texts_to_sequences）**
  - 把文本映射为 id 列表。
  - 按 `max_sequence_length` 做截断或 pad（pad 填 0）。

---

## 四种模型方法（一句话概览）

- **CNN**：基础一维卷积网络，单/少量卷积核，提取局部 n-gram 特征。  
- **CNN + MLP**：在 CNN 基础上加更深的全连接层（MLP），提升非线性建模能力。  
- **TextCNN**：多尺寸卷积核 (e.g. 3,4,5) + 全局池化（AdaptiveMaxPool），标准文本卷积强基线。  
- **TextCNN + Finetune(Google Word2Vec)**：TextCNN 架构，embedding 层用 GoogleNews Word2Vec 初始化并允许微调（或选择冻结）。  

---

## 1) CNN（基础版）

### 结构要点
- Embedding 层（随机或预训练） → 1D 卷积（单个 kernel_size，例如 3） → ReLU → 全局池化 → FC → 输出 logits。  
- 适合短文本 n-gram 的局部特征提取。  

### 优势
- 结构简单、参数少、训练速度快。  
- 对超短文本（非常短评论或切片）鲁棒，容易调参。  

### 局限
- 单一卷积核难以同时捕获不同粒度的 n-gram（例如 bi-gram 与 tri-gram）。  
- 表达能力受限，可能需要更多 FC 层或更大 filter 数量来提升性能。  

---

## 2) CNN + MLP

### 结构要点
- 与基础 CNN 相同的前端，但在池化后接入一到多个全连接层（MLP）。  
- 通常为：`FC → ReLU → Dropout → FC`，以提升非线性组合能力。  
- MLP 用于把卷积抽取到的局部特征映射到更高维的判别子空间。  

### 优势
- 在同等卷积特征下，MLP 能学到更复杂的特征组合，提高分类边界可分性。  
- 在数据量较大时表现优于简单 CNN。  

### 局限
- 参数更多，训练和过拟合风险增加（需 dropout/正则化）。  
- 若数据量小，可能因为模型容量过大而性能下降。  

---

## 3) TextCNN（推荐基线）

### 结构要点
- Embedding 层 → 多个并行 1D 卷积（kernel sizes = e.g. [3,4,5]）  
- 每路卷积后：`ReLU + AdaptiveMaxPool1d(1)`（得到每路的全局最大响应）  
- concat → Dropout → FC → logits。  
- 典型实现为 Kim Yoon 的 TextCNN（2014）思想。  

### 优势
- 多尺度卷积能同时捕获不同长度的 n-gram（短语与片段）。  
- 全局池化能对变长文本产生固定大小的特征向量、对位置不敏感，适配短/中长文本。  
- 参数可控，泛化较好。  

### 局限
- 相较 Transformer，无法直接捕获长距离依赖（但通常足够短文本情感分类）。  
- 对 embedding 质量依赖性较高：好的 embedding 明显提升性能。  

---

## 4) TextCNN + Finetune(Google Word2Vec)

### 结构要点
- 同 TextCNN，但 embedding 用 **GoogleNews-vectors-negative300.bin**（或其它预训练 Word2Vec）构造 `embedding_matrix`。  
- 两种策略：
  - **冻结 embedding**：训练更快、避免过拟合。  
  - **微调（finetune） embedding**：通常能带来更好性能，尤其在目标域与预训练语料较匹配时。  

### 优势
- 预训练 embedding 带来更富语义的初始化，能在小样本下提升收敛速度和精度。  
- 微调允许网络依据任务调整 embedding 表示，通常能进一步提升性能。  

### 局限
- GoogleNews 的英文模型对中文无用；中文场景需加载中文预训练（如 Tencent/fastText/GloVe/自训练 word2vec）。  
- 预训练模型文件大（内存/加载时间成本），且若域差异大（通用语料 vs Yelp 评论），可能需要更多微调或 domain-specific embedding。  
- 若数据量极少，微调 embedding 可能过拟合，需慎用。  

---

## 四种方法对比（优劣速览）

| 方法 | 训练速度 | 参数量 | 对短文本友好 | 对预训练embedding敏感 | 易过拟合（小数据） | 推荐场景 |
|------|----------|--------|---------------|-----------------------|--------------------|-----------|
| CNN | 非常快 | 低 | 高 | 低 | 低-中 | 资源受限、快速 baseline |
| CNN + MLP | 中 | 中 | 高 | 中 | 中 | 中等数据量，需要更强判别 |
| TextCNN | 中 | 中 | 高 | 中-高 | 中 | 文本分类通用强基线 |
| TextCNN + FinetuneEmb | 慢（加载emb） | 高 | 高 | 很高 | 高（若微调） | 有预训练emb & 中小样本，效果最佳 |
