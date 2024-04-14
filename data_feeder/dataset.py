import torch
import torchaudio


class Window(torch.nn.Module):

    def __init__(self, config):
        super(Window, self).__init__()

        self.config = config

        self.window = torch.hann_window(config.win_length).to(config.device)

    def forward(self, x):
        return x * self.window


class transform_wave(torch.nn.Module):

    def __init__(self, config):
        super(transform_wave, self).__init__()

        self.config = config

    def forward(self, x):

        t_f_module = torchaudio.transforms.Spectrogram(
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            window_fn=Window(self.config),
        )

        
        
        return t_f_module(x).T[:,1:]


class wave_dataset(torch.utils.data.Dataset):

    def __init__(self, waves, labels, config):

        self.config = config

        self.waves = waves
        self.labels = labels

        self.transform = transform_wave(config)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        if self.transform:
            wave = self.transform(self.waves[idx])
        else:
            wave = self.waves[idx]

        label = self.labels[idx]

        return wave, label
