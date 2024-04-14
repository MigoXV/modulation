import torch
import torch.nn as nn


class WaveTransformerModel(nn.Module):
    def __init__(
        self,
        config,

    ):
        
        super(WaveTransformerModel, self).__init__()
        
        d_model = config.d_model
        nhead = config.nhead
        num_layers = config.num_layers
        dim_feedforward = config.dim_feedforward
        dropout = config.dropout
        num_classes = config.num_classes
        
        # 创建一个Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # 使得输入数据的batch维度为第一维
        )
        # 利用编码器层创建一个Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # 全连接层用于分类
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src):
        # src的形状应为 [batch_size, seq_length, d_model]
        # 通过Transformer编码器传递输入
        output = self.transformer_encoder(src)
        # 取最后一个时间步的输出用于分类
        # 这里假设使用序列的最后一个元素进行分类
        output = output[:, -1, :]
        # 通过分类层
        output = self.classifier(output)
        return output
