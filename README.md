# yelp_star_classification_NLP


本项目基于 **PyTorch** 实现了 Yelp 评论星级分类任务，旨在通过用户评论文本预测对应的星级评分。  
项目包含 **CNN、TextCNN** 等多种深度学习模型，并支持使用 **Word2Vec 预训练词向量（可微调）** 提升分类效果。

---

## ✨ 特性
- **多模型实现**：简单 CNN、CNN+MLP、TextCNN，以及 Word2Vec + CNN。
- **词向量支持**：随机初始化或加载预训练 Word2Vec embedding，并可选择微调。
- **模块化代码**：清晰的模型、训练、数据处理结构，便于扩展。
- **完整训练流程**：涵盖数据预处理、训练、验证与评估。

---

# 数据集
Yelp Reviews是Yelp为了学习目的而发布的一个开源数据集。它包含了由数百万用户评论，商业属性和来自多个大都市地区的超过20万张照片。这是一个常用的全球NLP挑战数据集，包含5,200,000条评论，174,000条商业属性。 数据集下载地址为：

	https://www.yelp.com/dataset/download

Yelp Reviews格式分为JSON和SQL两种，以JSON格式为例,其中最重要的review.json,包含评论数据。Yelp Reviews数据量巨大，非常适合验证CNN模型。
# 数据清洗
Yelp Reviews文件格式为JSON和SQL，使用起来并不是十分方便。专门有个开源项目用于解析该JSON文件：

> https://github.com/Yelp/dataset-examples

该项目可以将Yelp Reviews的Yelp Reviews转换成CSV格式，便于进一步处理，该项目的安装非常简便，同步完项目后直接安装即可。

	git clone https://github.com/Yelp/dataset-examples
	python setup.py install

假如需要把review.json转换成CSV格式，命令如下：

	python json_to_csv_converter.py /dataset/yelp/dataset/review.json

命令执行完以后，就会在review.json相同目录下生成对应的CSV文件review.csv。查看该CSV文件的表头，内容如下，其中最重要的两个字段就是text和stars，分别代表评语和打分。

	#CSV格式表头内容：
	#funny,user_id,review_id,text,business_id,stars,date,useful,cool
		
使用pandas读取该CSV文件，开发阶段可以指定仅读取前10000行。
	
	#开发阶段读取前10000行
	df = pd.read_csv(filename,sep=',',header=0,nrows=10000)
---
## 📦 环境依赖
- Python >= 3.8  
- PyTorch >= 1.10  
- NumPy  
- Pandas  
- Scikit-learn  
- Gensim (用于加载 Word2Vec)  

安装依赖：
```bash
pip install -r requirements.txt
````

---

## 🚀 使用方法

### 1. 数据准备

* 数据集：Yelp 评论数据（文本 + 星级标签）
* 预处理：分词、去停用词、构建词向量字典

将处理后的数据放入 `data/` 目录。

### 2. 训练模型

运行训练脚本：

```bash
# 使用基础 TextCNN
python train_yelp_cnn.py --model textcnn

# 使用 Word2Vec embedding (微调)
python train_yelp_cnn.py --model textcnn --embedding word2vec --fine_tune True
```

可选参数：

* `--model`：模型类型 (`cnn`, `textcnn`)
* `--embedding`：embedding 初始化方式 (`random`, `word2vec`)
* `--fine_tune`：是否微调预训练词向量 (`True`/`False`)

### 3. 评估模型

训练完成后，脚本会自动在验证集上评估并输出准确率和F1分数。

---

## 📂 项目结构

```
├── data/                 # Yelp 数据集 (文本 + 标签)
├── models/               # 模型实现 (CNN / TextCNN / Word2Vec+CNN)
│   ├── cnn.py
│   ├── textcnn.py
│   └── utils.py
├── train_yelp_cnn.py     # 训练入口
├── requirements.txt      # 依赖包
└── README.md             # 项目说明
```

---

## 📊 实验结果 (示例)

| 模型            | Embedding  | Fine-tune | 准确率(%) |
|---------------| ---------- | --------- |-------|
| CNN           | Random     | -         | 87.8  |
| CNN+MLP       | Random     | -         | 90.2  |
| TextCNN       | Pretrained | False     | 90.75 |
| TextCNN + W2V | Pretrained | True      | 91.5  |

---

## 📜 License

本项目仅用于学习和研究，禁止用于商业用途。

