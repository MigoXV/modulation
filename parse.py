import os
import json
import torch

from demodulator.parser import Parser

def parse_main(ckpt_path:str, config_path: str, config_set: list):
    
    parser = Parser(config_path, ckpt_path)
    
    signal = torch.load('signal.pt')
    
    result = parser.parse_modulation(signal)