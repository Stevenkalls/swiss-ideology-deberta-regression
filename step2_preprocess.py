import pandas as pd

# ① 读取原始文件
df = pd.read_csv("model_fintune_dataset.csv")      # 把文件名换成你的实际路径
print(df.head())                           # 确认列名 text / ECON / CULT

# ② 转成 float，防止有字符串混进来
df["ECON"] = df["ECON"].astype(float)
df["CULT"] = df["CULT"].astype(float)

# ③ 如有缺失值可先丢弃或填充
df = df.dropna()                         


from datasets import Dataset, DatasetDict

# 把 Pandas DataFrame → HF Dataset
raw_ds = Dataset.from_pandas(df)

# 80 % 训练、10 % 验证、10 % 测试
dataset = raw_ds.train_test_split(test_size=0.1, seed=42)
train_val = dataset["train"].train_test_split(test_size=0.125, seed=42)  # 0.8*0.125≈0.1

train_ds = train_val["train"]   # 80%
val_ds   = train_val["test"]    # 10%
test_ds  = dataset["test"]      # 10%

print(train_ds, val_ds, test_ds)



from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

def tokenize_function(batch):
    # 1) 文本分词（padding 到 max_length，防止 DataLoader collation 出错）
    encodings = tok(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512,    # 够用了，可按需要调
    )

    # 2) 把 ECON 与 CULT 合在同一个 list → 模型头要 2 维
    encodings["labels"] = [
        [e, c] for e, c in zip(batch["ECON"], batch["CULT"])
    ]

    return encodings

# batched=True 批量映射，速度更快
train_ds = train_ds.map(tokenize_function, batched=True, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(tokenize_function,   batched=True, remove_columns=val_ds.column_names)
test_ds  = test_ds.map(tokenize_function,  batched=True, remove_columns=test_ds.column_names)

# HF 默认把 list 转成 float64；手动改成 float32 以节约显存
import numpy as np
def to_float32(batch):
    batch["labels"] = np.array(batch["labels"], dtype=np.float32)
    return batch

train_ds = train_ds.map(to_float32, batched=True)
val_ds   = val_ds.map(to_float32,   batched=True)
test_ds  = test_ds.map(to_float32,  batched=True)

# 保存到本地，下次直接 load_from_disk()，省去分词时间
train_ds.save_to_disk("ds_train")
val_ds.save_to_disk("ds_val")
test_ds.save_to_disk("ds_test")
