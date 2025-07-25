
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from safetensors.torch import load_file as safe_load  # 用于 *.safetensors

# 导入自定义模型
from step1_regression_model import DebertaDualReg


def parse_args():
    p = argparse.ArgumentParser(description="Batch inference with DeBERTa dual-regression model")
    p.add_argument("--input_file", required=True, help="仅含 text 列的 CSV")
    p.add_argument("--model_dir", default="dual_reg_deberta", help="模型目录或具体 checkpoint 目录")
    p.add_argument("--output_file", default="predictions.csv", help="输出文件名")
    p.add_argument("--batch", type=int, default=16, help="批量大小")
    p.add_argument("--max_length", type=int, default=256, help="分词最大长度")
    return p.parse_args()


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )


def collate(batch):
    # 拼接 batch 字典
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.cat([item[k] for item in batch], dim=0)
    return out


def load_model(model_dir: Path) -> DebertaDualReg:
    """根据目录下文件类型加载权重。"""
    model = DebertaDualReg()

    bin_path = model_dir / "pytorch_model.bin"
    safe_path = model_dir / "model.safetensors"

    if bin_path.exists():
        state = torch.load(bin_path, map_location="cpu")
    elif safe_path.exists():
        state = safe_load(str(safe_path))  # safetensors 返回 dict[str, torch.Tensor]
    else:
        raise FileNotFoundError(f"{model_dir} 中未找到 pytorch_model.bin / model.safetensors")

    model.load_state_dict(state)
    return model


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)

    # ------------- 加载输入文本 -------------
    df = pd.read_csv(args.input_file)
    if "text" not in df.columns:
        raise ValueError("输入 CSV 必须包含列名 text")
    texts = df["text"].tolist()

    # ------------- 初始化 tokenizer & model -------------
    tokenizer = AutoTokenizer.from_pretrained(model_dir)  # tokenizer 同目录
    model = load_model(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # ------------- DataLoader -------------
    ds = TextDataset(texts, tokenizer, args.max_length)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    econ_preds, cult_preds = [], []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch)["logits"].cpu()
            econ_preds.extend(logits[:, 0].tolist())
            cult_preds.extend(logits[:, 1].tolist())

    # ------------- 保存输出 -------------
    out_df = df.copy()
    out_df["ECON"] = econ_preds
    out_df["CULT"] = cult_preds
    out_df.to_csv(args.output_file, index=False, encoding="utf-8-sig")
    print(f" 推理完成 → {args.output_file}，共 {len(out_df)} 行")


if __name__ == "__main__":
    main()


#powershell 运行命令 将断点替换为所需要运行的checkpoint
# python step4_inference.py --input_file data\new_texts.csv --model_dir dual_reg_deberta\checkpoint-断电 --output_file results\predictions.csv --batch 32
