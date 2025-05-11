from utils import *


class EncoderBackbone(nn.Module):
    def __init__(self):
        super(EncoderBackbone, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 32, 1)
        self.bn4 = nn.BatchNorm1d(32)
        self.conv5 = nn.Conv1d(32, 2, 1)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        x = x.permute(0,2,1)
        return x
    

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc6 = nn.Linear(256, k*k)

        nn.init.zeros_(self.fc6.weight)
        self.fc6.bias.data.copy_(torch.eye(self.k).view(-1))

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x,2)[0]

        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = self.fc6(x)

        x = x.view(-1, self.k, self.k)

        return x
    

class Encoder(nn.Module):
    def __init__(self, k=3):
        super(Encoder, self).__init__()
        self.tnet = TNet(k=k)
        self.backbone = EncoderBackbone()

    def forward(self, x):
        x_trans = self.tnet(x.permute(0,2,1))
        x_aligned = torch.bmm(x, x_trans)
        out = self.backbone(x_aligned)

        if self.training:
            noise = torch.randn_like(out)*0.01
            out = out + noise

        return out



