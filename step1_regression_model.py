import torch  # 引入 PyTorch
import torch.nn as nn  # 神经网络模块
from transformers import AutoModel  # 加载预训练模型的高层 API


class DebertaDualReg(nn.Module):
    """
    基于 DeBERTa-v3-base 的 双输出回归模型。

    """

    def __init__(self,
                 pretrained_model_name_or_path: str = "microsoft/deberta-v3-base",
                 dropout_prob: float = 0.1):
        super().__init__()  # 初始化父类

        # ------------------- 预训练编码器 -------------------
        self.backbone = AutoModel.from_pretrained(pretrained_model_name_or_path)
        hidden_size = self.backbone.config.hidden_size  # DeBERTa base 通常是 768

        # ------------------- 回归头 -------------------
        self.reg_head = nn.Sequential(
            nn.Dropout(dropout_prob),  # 随机失活，防止过拟合
            nn.Linear(hidden_size, 2)  # 输出 2 维连续值
        )

    # ------------------------------------------------------------------
    # forward(): 允许 token_type_ids，可兼容 BertTokenizer 的输出
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,  # 新增参数，默认 None；即使不用也能接收
        labels=None,
        **extra_kwargs  # 与 Hugging Face Trainer 兼容，忽略多余参数
    ):
        """前向传播。

        与 `transformers` 的调用方式保持一致，方便直接喂入 tokenizer 输出。
        """
        # 调用 DeBERTa 编码器；若 token_type_ids 为 None，HF 会忽略它
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **extra_kwargs
        )

        # [CLS] 向量 — 用作句子级表示
        pooled = outputs.last_hidden_state[:, 0]

        # 经过回归头得到预测
        logits = self.reg_head(pooled)

        # 训练阶段计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels.float())

        return {"loss": loss, "logits": logits}


# ------------------ 简易脚本测试 ------------------
if __name__ == "__main__":
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    text = "这是一条测试文本。"
    inputs = tok(text, return_tensors="pt")  # 注意：会包含 token_type_ids (全 0)

    model = DebertaDualReg()
    model.eval()

    with torch.no_grad():
        outs = model(**inputs)

    print("预测结果 (y1, y2):", outs["logits"].squeeze().cpu().tolist())