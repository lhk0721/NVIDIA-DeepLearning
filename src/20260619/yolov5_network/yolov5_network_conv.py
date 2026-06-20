import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# 1. 기본 conv 블록 : Conv + BN + SiLu
class Conv(nn.Module):
    def __init__(self, c_in, c_out, k=1, s=1, p=None, g=1):
        super().__init__()
        if p is None:
            p = (k -1) // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class YOLOv5(nn.Module):
    def __init__(self, nc=80):
        super().__init__()

        # Backbone
        # 3입력채널, 64개 출력 채널(특징맵깊이), 3x3커널사이즈, 2 스트라이드
        self.conv1 = Conv(3,64, 3, 2)
        # 64입력채널, 128 출력채널,  3x3커널사이즈, 2 스트라이드
        self.conv2 = Conv(64, 128, 3, 2)

    def forward(self, x):
        # input_size=(1, 3, 640, 640) 가 x 로 전달
        x = self.conv1(x)
        x = self.conv2(x)

        return x



# main 동작
if __name__ == "__main__":
    model = YOLOv5(nc=80)
    #x = torch.randn(1,3,640,640)
    summary(model, input_size=(1, 3, 640, 640))
