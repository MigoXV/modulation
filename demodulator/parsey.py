import torch
import numpy as np
from .hibert import HilbertTransform
from .classifier import Classifier
from .demodulator import Demodulator

class Parser:
    
    def __init__(self, config, ckpt_path: str):
        
        self.fs = 8e6
        self.fc = 2e6
        self.hilbert_transform = HilbertTransform()
        self.classifier = Classifier(config, ckpt_path)
        self.demodulator = Demodulator()
        
    def parse_modulation(self, signal: torch.Tensor, modulation: str) -> dict:
        
        modulation = self.classifier(signal)
        
        demodulated_signal = self.demodulator(signal, modulation)
        
        if modulation == 'cw':
            return {'modulation': 'CW'}
        
        if modulation == 'am':
            return self.calculate_am_params(signal, demodulated_signal)
        elif modulation == 'fm':
            return self.calculate_fm_params(signal, demodulated_signal)
        elif modulation == '2fsk':
            return self.calculate_2fsk_params(signal, demodulated_signal)
        else:  # Assuming 2PSK or 2ASK
            return self.calculate_2psk_params(signal, demodulated_signal)
    
    def calculate_am_params(self, signal: torch.Tensor, demodulated_signal: torch.Tensor) -> dict:
        amplitude = torch.max(demodulated_signal) - torch.min(demodulated_signal)
        # Calculate frequency using Fourier Transform
        spectrum = torch.fft.fft(signal)
        freq = torch.fft.fftfreq(signal.size(0), d=1/self.fs)
        peak_freq = freq[torch.argmax(torch.abs(spectrum))]
        return {
            'modulation': 'AM',
            'amplitude': amplitude.item(),
            'frequency': peak_freq.item()
        }
    
    def calculate_fm_params(self, signal: torch.Tensor, demodulated_signal: torch.Tensor) -> dict:
        deviation = torch.max(demodulated_signal) - torch.min(demodulated_signal)
        freq_spectrum = torch.fft.fft(demodulated_signal)
        freq = torch.fft.fftfreq(demodulated_signal.size(0), d=1/self.fs)
        peak_freq = freq[torch.argmax(torch.abs(freq_spectrum))]
        modulation_index = deviation / self.fc
        return {
            'modulation': 'FM',
            'frequency': peak_freq.item(),
            'modulation_index': modulation_index.item(),
            'max_deviation': deviation.item()
        }
    
    def calculate_2fsk_params(self, signal: torch.Tensor, demodulated_signal: torch.Tensor) -> dict:
        freqs, power = self.get_freq_power(signal)
        peak_indices = torch.argsort(power, descending=True)[:2]
        f1, f2 = freqs[peak_indices].tolist()
        baud_rate = self.calculate_baud_rate(demodulated_signal)
        return {
            'modulation': '2FSK',
            'baud_rate': baud_rate,
            'frequency_1': f1,
            'frequency_2': f2
        }
    
    def calculate_2psk_params(self, signal: torch.Tensor, demodulated_signal: torch.Tensor) -> dict:
        baud_rate = self.calculate_baud_rate(demodulated_signal)
        return {
            'modulation': '2PSK',
            'baud_rate': baud_rate
        }
    
    def calculate_baud_rate(self, demodulated_signal: torch.Tensor) -> float:
        # Use zero crossings to estimate baud rate
        zero_crossings = ((demodulated_signal[:-1] * demodulated_signal[1:]) < 0).sum().item()
        baud_rate = zero_crossings / (demodulated_signal.size(0) / self.fs / 2)
        return baud_rate
    
    def get_freq_power(self, signal: torch.Tensor) -> tuple:
        spectrum = torch.fft.fft(signal)
        power = torch.abs(spectrum) ** 2
        freqs = torch.fft.fftfreq(signal.size(0), d=1/self.fs)
        return freqs, power
