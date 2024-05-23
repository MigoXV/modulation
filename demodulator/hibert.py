import torch
import torch.nn as nn

class HilbertTransform(nn.Module):
    """
    一个实现希尔伯特变换的类。
    
    继承自 `torch.nn.Module`，该类通过 FFT 来获取信号的希尔伯特变换。
    
    属性:
        无额外属性。

    方法:
        forward(signal): 对给定的信号应用希尔伯特变换。

    示例:
        >>> transform = HilbertTransform()
        >>> signal = torch.rand(10)
        >>> transformed_signal = transform(signal)

    注:
        该类仅适用于最后一个维度是时间序列的一维张量。
    """
    def __init__(self):
        """
        初始化 HilbertTransform 类的实例。
        """
        super(HilbertTransform, self).__init__()

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        对输入的信号进行希尔伯特变换。

        Args:
            signal (torch.Tensor): 输入的信号张量。

        Returns:
            torch.Tensor: 希尔伯特变换后的信号张量。
        """
        # 获取信号的长度
        N = signal.size(-1)

        # 计算信号的 FFT
        fft_signal = torch.fft.fft(signal)
        
        # 创建一个长度为 N 的全零复数张量
        h = torch.zeros(N, dtype=torch.complex64)

        # 根据 N 的奇偶性，构造频率响应
        if N % 2 == 0:
            h[0] = h[N//2] = 1
            h[1:N//2] = 2
        else:
            h[0] = 1
            h[1:(N+1)//2] = 2

        # 应用频率响应，并通过 IFFT 返回时域中的希尔伯特变换信号
        hilbert_signal = torch.fft.ifft(fft_signal * h)
        return hilbert_signal