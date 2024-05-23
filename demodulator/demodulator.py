import torch

from .hibert import HilbertTransform

class Demodulator:
    
    def __init__(self):
        
        self.fc=2e6
        self.hilbert_transform = HilbertTransform()
        
    def __call__(self, signal: torch.Tensor, modulation: str) -> torch.Tensor:
        
        if modulation == 'am' or modulation == '2ask':
            return self.demodulate_am(signal)
        elif modulation == 'fm' or modulation == '2fsk':
            return self.demodulate_fm(signal)
        else:
            return self.demodulate_2psk(signal)
        
    def demodulate_am(self, signal: torch.Tensor) -> torch.Tensor:
            
        analytic_signal = self.hilbert_transform(signal)
        envelope = torch.abs(analytic_signal)
        return envelope
    
    def demodulate_fm(self, signal: torch.Tensor) -> torch.Tensor:
        
        analytic_signal = self.hilbert_transform(signal)
        instantaneous_phase = torch.angle(analytic_signal)
        instantaneous_frequency = torch.diff(instantaneous_phase, dim=-1)
        # 保证差分后长度不变
        instantaneous_frequency = torch.cat([instantaneous_frequency, instantaneous_frequency[..., -1:]], dim=-1)
        base_frequency = instantaneous_frequency - self.fc
        return base_frequency
    
    def demodulate_2psk(self, signal: torch.Tensor) -> torch.Tensor:
        
        analytic_signal = self.hilbert_transform(signal)
        instantaneous_phase = torch.angle(analytic_signal)
        return instantaneous_phase
 