# train_yelp_cnn.py
# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import pandas as pd
import re
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 可选 gensim 支持（word2vec）
try:
    from gensim.models import KeyedVectors
    _HAS_GENSIM = True
except Exception:
    _HAS_GENSIM = False

# ---------------- 可配置项 ----------------
yelp_file = "../data/dataset-examples-master/output.csv"
word2vec_file = "../data/GoogleNews-vectors-negative300.bin"  # 如需使用，确认路径正确且已安装 gensim
nrows = 10000  # 调试时用小数据，生产可设为 None 或更大
max_features = 5000
max_document_length = 200
batch_size = 32
NUM_EPOCHS = 15
embedding_dim = 50
filters = 250
dropout_rate = 0.5
kernel_size = 3
hidden_dims = 500
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)
epoch_losses = []
epoch_accs = []


# 模型选择（可改为 'simple_cnn' | 'cnn_mlp' | 'textcnn' | 'textcnn_w2v'）
MODEL_CONFIG = {
    "model_type": "textcnn_w2v",   # 这里改模型类型即可
    "common": {  # 大多数模型都会用到的公共参数
        "embedding_dim": 50,
        "hidden_dim": 128,
        "dropout": 0.5,
        "filters": 250,
        "filter_sizes": (3, 4, 5)
    },
    "extra": {  # 只有某些模型需要的额外参数
        "word2vec_file": "../data/GoogleNews-vectors-negative300.bin",
        "trainable": True
    }
}
# ---------------- 随机种子（可复现） ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------- Tokenizer & Dataset ----------------
class SimpleTokenizer:
    def __init__(self, max_words=5000, oov_token="<OOV>", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\\n\\t',
                 lower=True, char_level=False, max_sequence_length=None):
        self.word_index = {}
        self.index_word = {}
        self.word_counts = defaultdict(int)
        self.max_words = max_words
        self.oov_token = oov_token
        self.filters = filters
        self.lower = lower
        self.char_level = char_level
        self.max_sequence_length = max_sequence_length
        self.oov_index = 1  # OOV token index

    def clean_text(self, text):
        if self.lower:
            text = text.lower()
        for ch in self.filters:
            text = text.replace(ch, ' ')
        text = re.sub('\\s+', ' ', text).strip()
        return text

    def fit_on_texts(self, texts):
        for text in texts:
            cleaned = self.clean_text(str(text))
            words = list(cleaned) if self.char_level else cleaned.split()
            for w in words:
                if w:
                    self.word_counts[w] += 1
        # reserve index 0 for PAD, 1 for OOV
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        self.word_index[self.oov_token] = self.oov_index
        self.index_word[self.oov_index] = self.oov_token
        for i, (word, _) in enumerate(sorted_words[:self.max_words - 1]):
            self.word_index[word] = i + 2
            self.index_word[i + 2] = word
        # safer vocab_size = max index + 1 (include 0 padding)
        max_idx = max(self.word_index.values()) if self.word_index else 1
        self.vocab_size = max_idx + 1

    def texts_to_sequences(self, texts):
        seqs = []
        for text in texts:
            cleaned = self.clean_text(str(text))
            words = list(cleaned) if self.char_level else cleaned.split()
            seq = [self.word_index.get(w, self.oov_index) for w in words if w]
            if self.max_sequence_length:
                if len(seq) > self.max_sequence_length:
                    seq = seq[:self.max_sequence_length]
                else:
                    seq = seq + [0] * (self.max_sequence_length - len(seq))
            seqs.append(seq)
        return seqs

class YelpDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        # ensure tokenizer will pad/truncate to this length
        self.tokenizer.max_sequence_length = max_length
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        tokens = self.tokenizer.texts_to_sequences([text])[0]
        # ensure length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# ---------------- 模型 ----------------
class CNNMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, num_classes=2, filters=250, kernel_size=3, hidden_dims=250):
        super(CNNMLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Conv1d: in_channels = embedding_dim
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=kernel_size, padding=0)
        self.relu = nn.ReLU()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(filters, hidden_dims)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dims, num_classes)
        # IMPORTANT: do NOT apply softmax here; return logits
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)                   # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)                 # (batch, embed_dim, seq_len)
        x = self.conv1d(x)                     # (batch, filters, L_out)
        x = self.relu(x)
        x = self.global_max_pool(x).squeeze(-1)  # (batch, filters)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        logits = self.fc2(x)                   # (batch, num_classes) -- logits
        return logits

class SimpleCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, num_classes=2, filters=250, kernel_size=3):
        super(SimpleCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=filters,
                                kernel_size=kernel_size, padding=0)
        self.relu = nn.ReLU()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.global_max_pool(x).squeeze(-1)
        logits = self.fc(x)
        return logits

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, num_classes=2, filters=100,
                 filter_sizes=(3, 4, 5), hidden_dim=32, dropout=dropout_rate):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # 用参数名 filters（与 build_model 一致）
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=filters,
                      kernel_size=fs, padding=0)
            for fs in filter_sizes
        ])
        self.global_pools = nn.ModuleList([nn.AdaptiveMaxPool1d(1) for _ in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(filters * len(filter_sizes), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)          # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)         # (batch, embed_dim, seq_len)
        conv_outputs = []
        for conv, pool in zip(self.convs, self.global_pools):
            out = conv(x)
            out = self.relu(out)
            out = pool(out)
            out = out.view(out.size(0), -1)
            conv_outputs.append(out)
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits

class TextCNNWithWord2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, num_classes=2, filters=100,
                 filter_sizes=(3,4,5), hidden_dim=32,
                 embedding_matrix=None, trainable=True, dropout=dropout_rate):
        super(TextCNNWithWord2Vec, self).__init__()
        if embedding_matrix is not None:
            emb_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
            self.embedding = nn.Embedding.from_pretrained(
                emb_tensor, freeze=not trainable, padding_idx=0
            )
            self.embedding_dim = emb_tensor.size(1)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.embedding_dim = embedding_dim

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.embedding_dim, out_channels=filters, kernel_size=fs, padding=0)
            for fs in filter_sizes
        ])
        self.global_pools = nn.ModuleList([nn.AdaptiveMaxPool1d(1) for _ in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(filters * len(filter_sizes), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        conv_outputs = []
        for conv, pool in zip(self.convs, self.global_pools):
            out = conv(x)
            out = self.relu(out)
            out = pool(out)
            out = out.view(out.size(0), -1)
            conv_outputs.append(out)
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits

# ---------------- 模型构造函数 ----------------
def build_model(config, vocab_size, num_classes, tokenizer=None):
    model_type = config['model_type']

    if model_type == 'simple_cnn':
        model = SimpleCNN(
            vocab_size=vocab_size,
            num_classes=num_classes
        )
        model_name = "SimpleCNN"

    elif model_type == 'cnn_mlp':
        model = CNNMLP(
            vocab_size=vocab_size,
            embedding_dim=config.get('embedding_dim', 50),
            num_classes=num_classes
        )
        model_name = "CNN_MLP"

    elif model_type == 'textcnn':
        model = TextCNN(
            vocab_size=vocab_size,
            embedding_dim=config.get('embedding_dim', 50),
            filters=config.get('filters', 100),
            filter_sizes=config.get('filter_sizes', (3, 4, 5)),
            hidden_dim=config.get('hidden_dim', 128),
            dropout=config.get('dropout', 0.5),
            num_classes=num_classes
        )
        model_name = "TextCNN"


    elif model_type == 'textcnn_w2v':
        embedding_matrix = None
        w2v_path = config.get('word2vec_file', None)
        if w2v_path and _HAS_GENSIM and tokenizer is not None:
            print(f"Loading Word2Vec from {w2v_path} ...")
            w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
            embedding_dim = w2v.vector_size
            embedding_matrix = np.zeros((vocab_size, embedding_dim))
            for word, idx in tokenizer.word_index.items():
                if word in w2v:
                    embedding_matrix[idx] = w2v[word]
                else:
                    embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
        else:
            print("⚠️ 未提供 word2vec_file 或 gensim 不可用，将使用随机初始化。")
        model = TextCNNWithWord2Vec(
            vocab_size=vocab_size,
            embedding_dim=config.get('embedding_dim', 50),
            filters=config.get('filters', 100),
            filter_sizes=config.get('filter_sizes', (3, 4, 5)),
            hidden_dim=config.get('hidden_dim', 128),
            dropout=config.get('dropout', 0.5),
            num_classes=num_classes,
            embedding_matrix=embedding_matrix,
            trainable=config.get('trainable', True)
        )
        model_name = "TextCNN_Word2Vec"

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, model_name

# ---------------- 数据加载函数 ----------------
def load_reviews(filename, nrows=None):
    df = pd.read_csv(filename, sep=',', header=0, encoding='utf-8', nrows=nrows)
    texts = df['text'].astype(str).tolist()
    stars = df['stars'].tolist()
    plt.figure()
    pd.Series(stars).value_counts().sort_index().plot(kind='bar')
    plt.xlabel('stars'); plt.ylabel('counts'); plt.savefig("yelp_stars.png")
    return texts, stars

# ---------------- 训练 + 验证 + 评估 ----------------
def train_model(model, train_loader, val_loader, criterion,
                optimizer, scheduler, device,
                num_epochs, save_best_path="best_model.pth", early_stop_patience=None):
    """
    说明:
      - scheduler 在外部创建并传入（不要在此再次创建）
      - num_epochs 由外部传入（与脚本顶部的 NUM_EPOCHS 保持一致）
      - early_stop_patience: 可选, 若为 int 则启用早停 (以 val_acc 为监控指标)
    """
    model.to(device)
    best_val_acc = -1.0
    no_improve_count = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for i, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

            if i % 20 == 0:
                print(f"Epoch[{epoch+1}/{num_epochs}] Batch[{i}] loss:{loss.item():.4f} batch_acc:{100*(preds==target).float().mean():.2f}%")

        # 每个 epoch 结束后做验证/调度/记录
        avg_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        epoch_losses.append(avg_loss)
        epoch_accs.append(train_acc)

        # 先在验证集上评估（如果有）
        val_acc = None
        val_f1 = None
        if val_loader is not None:
            val_acc, val_f1, _, _ = evaluate_model(model, val_loader, device)
            print(f"Validation -> acc: {val_acc:.2f}%, f1: {val_f1:.4f}")

        # 更新学习率（按 epoch）
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Epoch {epoch+1} avg_loss: {avg_loss:.4f}, train_acc: {train_acc:.2f}%")

        # 保存最优模型（用 val_acc 作准；若没有 val_loader 则用 train_acc）
        monitor_acc = val_acc if val_acc is not None else train_acc
        if monitor_acc is not None and monitor_acc > best_val_acc:
            best_val_acc = monitor_acc
            torch.save(model.state_dict(), save_best_path)
            print(f"Saved best model to {save_best_path} (acc={best_val_acc:.2f}%)")
            no_improve_count = 0
        else:
            no_improve_count += 1


    return model


def evaluate_model(model, test_loader, device):
    model.eval()
    preds_list = []
    labels_list = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            out = model(data)
            preds = torch.argmax(out, dim=1)
            preds_list.extend(preds.cpu().numpy().tolist())
            labels_list.extend(target.cpu().numpy().tolist())
    all_preds = np.array(preds_list, dtype=int)
    all_labels = np.array(labels_list, dtype=int)
    acc = 100.0 * (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average='micro')
    return acc, f1, all_preds, all_labels

# ---------------- 主流程 ----------------
if __name__ == "__main__":
    # load
    text, stars = load_reviews(yelp_file, nrows)
    labels = [0 if s < 3 else 1 for s in stars]
    print("label distribution:\n", pd.Series(labels).value_counts())

    # tokenizer
    tokenizer = SimpleTokenizer(max_words=max_features, max_sequence_length=max_document_length)
    tokenizer.fit_on_texts(text)
    print("vocab size:", tokenizer.vocab_size)

    # train/val/test split: first 80/20, then split train->train/val
    train_texts, test_texts, train_labels, test_labels = train_test_split(text, labels, test_size=0.2, random_state=SEED, stratify=labels)
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.1, random_state=SEED, stratify=train_labels)

    train_dataset = YelpDataset(train_texts, train_labels, tokenizer, max_document_length)
    val_dataset = YelpDataset(val_texts, val_labels, tokenizer, max_document_length)
    test_dataset = YelpDataset(test_texts, test_labels, tokenizer, max_document_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # build model (factory)
    vocab_size = tokenizer.vocab_size
    num_classes = len(set(labels))

    # 构建模型
    model, model_name = build_model(MODEL_CONFIG, vocab_size, num_classes)
    print("Using model:", model_name)

    # optimizer / loss / scheduler
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 余弦退火调度器（按 epoch 更新）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,  # 一个周期为整个训练周期
        eta_min=1e-6
    )

    # debug one sample
    sample_tokens, sample_label = train_dataset[0]
    print("sample token ids (first 20):", sample_tokens[:20].tolist(), "label:", sample_label.item())

    # train (saves best model to best_model.pth)
    best_path = "best_model.pth"
    model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=NUM_EPOCHS,
        save_best_path="best_model.pth",
    )
    # load best checkpoint for final evaluation (if exists)
    if os.path.exists(best_path):
        print("Loading best model from:", best_path)
        model.load_state_dict(torch.load(best_path, map_location=device))
        model = model.to(device)

    # evaluate on test set
    acc, f1, all_preds, all_labels = evaluate_model(model, test_loader, device)
    print(f"Test acc: {acc:.2f}%, F1: {f1:.4f}")

    # classification report & confusion matrix
    print(classification_report(all_labels, all_preds, digits=4))
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix:\n", cm)

    # 可视化混淆矩阵
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    # 画训练曲线
    plt.figure()
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses, marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Avg Loss'); plt.title('Train Loss')
    plt.savefig('train_loss.png')

    plt.figure()
    plt.plot(range(1, len(epoch_accs)+1), epoch_accs, marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Train Accuracy')
    plt.savefig('train_acc.png')