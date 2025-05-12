from utils import *

class Decoder(nn.Module):
    def __init__(self, input_dim=2, grid_dim=2, output_dim=3):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim+grid_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 64, 1)
        self.conv4 = nn.Conv1d(64, output_dim, 1)

    def build_grid(self, batch_size, num_points):
        side_len = int(num_points ** 0.5)+1
        x = torch.linspace(-1, 1, steps = side_len)
        y = torch.linspace(-1, 1, steps = side_len)
        grid = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1).reshape(-1,2)
        grid = grid[:num_points]
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)
        return grid
    
    def forward(self, x):
        B, N, _ = x.shape
        device = x.device

        grid = self.build_grid(B, N).to(device)

        folded_input = torch.cat([x, grid], dim=2)
        folded_input = folded_input.permute(0,2,1)

        x = F.relu(self.conv1(folded_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = x.permute(0, 2, 1)

        return x