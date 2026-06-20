import torch
import torch.nn as nn


#  Focus 레이어 (YOLOv5 초기 버전의 입력단)
#   - H,W 크기를 절반으로 줄이면서 채널 4배

class Focus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Focus, self).__init__()
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv(torch.cat([
            x[:, :, ::2, ::2],  # Top-left
            x[:, :, ::2, 1::2],  # Top-right
            x[:, :, 1::2, ::2],  # Bottom-left
            x[:, :, 1::2, 1::2]  # Bottom-right
        ], dim=1))


focus_layer = Focus(3, 32)
input_image = torch.randn(1, 3, 640, 640)
output = focus_layer(input_image)

print(output.shape)