import torch

from .hibert import HilbertTransform
from .classifier import Classifier

class Demodulater:
    
    def __init__(self, config, ckpt_path: str):
        
        self.fc=2e6
        
        self.hilbert_transform = HilbertTransform()
        self.classifier = Classifier(config, ckpt_path)
        
    def __call__(self, signal: torch.Tensor, modulation: str) -> torch.Tensor:
        
        if modulation == 'am':
            return self.demodulate_am(signal)
        elif modulation == '2ask':
            return self.demodulate_am(signal)
        elif modulation == 'fm':
            return self.demodulate_fm(signal)
        elif modulation == '2fsk':
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
        base_frequency = instantaneous_frequency - self.fc
        return base_frequency
    
    def demodulate_2psk(self, signal: torch.Tensor) -> torch.Tensor:
        
        analytic_signal = self.hilbert_transform(signal)
        instantaneous_phase = torch.angle(analytic_signal)
        return instantaneous_phase
 