import torch

class wave_dataset(torch.utils.data.Dataset):
    
    def __init__(self,waves,labels,transform=None):
        self.waves = waves
        self.labels = labels
        
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        
        if self.transform:
            wave = self.transform(self.waves[idx])
        else:
            wave = self.waves[idx]
            
        label = self.labels[idx]
        
        return wave,label


        
        