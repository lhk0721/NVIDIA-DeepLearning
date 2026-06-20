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


# ----------------------------------------------------
# 2. Focus 레이어 (YOLOv5 초기 버전의 입력단)
#   - H,W 크기를 절반으로 줄이면서 채널 4배
# ----------------------------------------------------
class Focus(nn.Module):
    def __init__(self, c_in, c_out, k=1):
        super().__init__()
        self.conv = Conv(c_in * 4, c_out, k=k)

    def forward(self, x):
        # x: (B, C, H, W)  ==> (1, 3, 640, 640)
        # 4개의 패턴으로 슬라이싱 후 채널 방향으로 concat
        x1 = x[..., ::2, ::2]
        x2 = x[..., 1::2, ::2]
        x3 = x[..., ::2, 1::2]
        x4 = x[..., 1::2, 1::2]
        x = torch.cat([x1, x2, x3, x4], dim=1)
        print('focus forward x : ', x.shape) #  [1, 12, 320, 320]
        return self.conv(x)


class YOLOv5(nn.Module):
    def __init__(self, nc=80):
        super().__init__()

        # Backbone
        #  3 입력채널, 64 출력 채널(특징맵깊이), k=3
        self.focus = Focus(3, 64, k=3) # ==>  [1, 64, 320, 320]
        # 64입력채널, 128개 출력 채널(특징맵깊이), 3x3커널사이즈, 2 스트라이드
        self.conv1 = Conv(64,128, 3, 2) # ==> [1, 128, 160, 160]



    def forward(self, x):
        x = self.focus(x)  # (B, 64, 320, 320)
        # input_size=(1, 3, 640, 640) 가 x 로 전달
        x = self.conv1(x)

        return x



# main 동작
if __name__ == "__main__":
    model = YOLOv5(nc=80)
    #x = torch.randn(1,3,640,640)
    #out = model(x)
    summary(model, input_size=(1, 3, 640, 640))
