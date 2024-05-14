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

    def forward(self, *inputs, **kwargs):
        # 假设每个input都是一个完整的src，形状为 [batch_size, seq_length, d_model]
        # 将所有的inputs沿一个新的维度合并，假设它们的批次大小和序列长度都相同
        src = inputs[0] # 沿着批次维度合并
        # 通过Transformer编码器传递合并后的输入
        output = self.transformer_encoder(src)
        # 取每个序列最后一个时间步的输出用于分类
        output = output[:, -1, :]
        # 通过分类层
        output = self.classifier(output)
        # softmax
        output = torch.softmax(output, dim=-1)
        
        return output
