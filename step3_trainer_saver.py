

from pathlib import Path
import argparse
import numpy as np

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ---------------- 导入自定义模型 ----------------
from step1_regression_model import DebertaDualReg  # 步骤 1 定义的类

# ---------------- 参数解析 ----------------

def get_args():
    p = argparse.ArgumentParser(description="Fine‑tune DeBERTa dual‑regression model")

    # 数据目录
    p.add_argument("--train_dir", type=str, default="ds_train", help="HF Dataset 训练集目录")
    p.add_argument("--val_dir",   type=str, default="ds_val",   help="验证集目录")
    p.add_argument("--test_dir",  type=str, default="ds_test",  help="测试集目录 (可选)")

    # 输出与超参
    p.add_argument("--output_dir", type=str, default="dual_reg_deberta", help="权重输出目录")
    p.add_argument("--epochs",     type=int, default=5,          help="训练 epochs")
    p.add_argument("--batch",      type=int, default=8,          help="单 GPU batch")
    p.add_argument("--lr",         type=float, default=2e-5,     help="学习率")
    p.add_argument("--weight_decay", type=float, default=0.01,   help="权重衰减")
    p.add_argument("--warmup_ratio", type=float, default=0.1,    help="LR warm‑up 比例")
    p.add_argument("--fp16", action="store_true", help="使用 FP16 混合精度 (需 GPU)")

    return p.parse_args()

# ---------------- 评估指标 ----------------

def compute_metrics(eval_pred):
    """计算 MSE/MAE（分别对 ECON 与 CULT）"""
    logits, labels = eval_pred

    preds = np.array(logits)
    labels = np.array(labels)

    mse = ((preds - labels) ** 2).mean(axis=0)  # (2,)
    mae = np.abs(preds - labels).mean(axis=0)

    return {
        "mse_econ": float(mse[0]),
        "mse_cult": float(mse[1]),
        "mse_mean": float(mse.mean()),
        "mae_econ": float(mae[0]),
        "mae_cult": float(mae[1]),
        "mae_mean": float(mae.mean()),
    }

# ---------------- 主函数 ----------------

def main():
    args = get_args()

    # ----------- 数据集加载 & 校验 -----------
    train_dir = Path(args.train_dir)
    val_dir   = Path(args.val_dir)
    test_dir  = Path(args.test_dir) if args.test_dir else None

    assert train_dir.exists(), f"训练集路径不存在: {train_dir}"
    assert val_dir.exists(),   f"验证集路径不存在: {val_dir}"

    train_ds = load_from_disk(str(train_dir))
    val_ds   = load_from_disk(str(val_dir))
    test_ds  = load_from_disk(str(test_dir)) if test_dir and test_dir.exists() else None

    # ------------- 分词器 & 数据整理 -------------
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    # collator 自动 padding，并保持 labels 为 float32
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # ------------- 模型初始化 -------------
    model = DebertaDualReg()

    # ------------- 训练参数 -------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=str(Path(args.output_dir) / "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="mse_mean",
        greater_is_better=False,
        seed=42,
        fp16=args.fp16 and torch.cuda.is_available(),
    )

    # ------------- Trainer -------------
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # ------------- 训练 & 保存 -------------
    train_result = trainer.train()

    # 保存最佳模型权重与 tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 记录训练指标
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # ------------- 最终验证集评估 -------------
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # ------------- 可选：测试集评估 -------------
    if test_ds:
        test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    print(" 训练完成，模型已保存到:", args.output_dir)


if __name__ == "__main__":
    main()
