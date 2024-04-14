import torch
import torchaudio


class wave_dataset(torch.utils.data.Dataset):
    
    def __init__(self,waves,labels,config):
        
        self.config = config
        
        self.waves = waves
        self.labels = labels
        
        self.transform = torchaudio.transforms.Spectrogram(
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
        )
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        
        if self.transform:
            wave = self.transform(self.waves[idx])
        else:
            wave = self.waves[idx]
            
        label = self.labels[idx]
        
        return wave,label


        
        