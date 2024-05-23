import torch

from model.wave_transformer import WaveTransformerModel


label_dict = {
    "cw": 0,
    "am": 1,
    "fm": 2,
    "2ask": 3,
    "2fsk": 4,
    "2psk": 5,
}

# 反转的字典
label_dict = {
    0: "cw",
    1: "am",
    2: "fm",
    3: "2ask",
    4: "2fsk",
    5: "2psk",

}

class Classifier:

    def __inti__(self, config, ckpt_path: str):

        self.model = WaveTransformerModel(config)
        self.model.load_state_dict(torch.load(ckpt_path))
        
    def __call__(self, signal: torch.Tensor) -> str:
        
        with torch.no_grad():
            label = self.model(signal)
            return label_dict[label.item()]

    