import torch.nn as nn
import torch

class CRNN(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super().__init__()
        # small CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1)),  # keep width more
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )
        # compute LSTM input dim: depends on output channels and height reduction
        rnn_input_size = 256 * 8  # if height=32 and two poolings reduce to ~8
        self.rnn = nn.Sequential(
            nn.Linear(rnn_input_size, 512),
            nn.ReLU(inplace=True),
            nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True),
        )
        self.fc = nn.Linear(256*2, num_classes)  # bidirectional

    def forward(self, x):
        # x: B x C x H x W
        b, c, h, w = x.size()
        feat = self.cnn(x)
        # feat: B x C' x H' x W'
        b, c2, h2, w2 = feat.size()
        # collapse height -> feature vector along width
        feat = feat.permute(0, 3, 1, 2).contiguous()  # B, W', C', H'
        feat = feat.view(b, w2, c2 * h2)  # B, W', D
        # pass through linear then LSTM
        lin = self.rnn[0](feat)
        lin = self.rnn[1](lin)
        rnn_out, _ = self.rnn[2](lin)
        logits = self.fc(rnn_out)  # B, W', num_classes
        # for CTC we want T x B x C
        return logits.permute(1, 0, 2)
