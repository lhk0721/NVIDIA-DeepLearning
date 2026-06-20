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

# BottleNeck 1/ 2
# BottleNeck 1 : shortcut 0 (residual)
# BottleNeck 2 : shortcut x
class BottleNeck(nn.Module):
    def __init__(self, c, shortcut=True):
        """
         c: in/out 채널 동일
         shortcut: True -> BottleNeck1, False -> BottleNeck2
        """
        super().__init__()
        self.cv1 = Conv(c, c , k=1, s=1, p=0)
        self.cv2 = Conv(c, c , k=3, s=1, p=1)
        self.use_shortcut = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        if self.use_shortcut:
            return x + y
        else:
            return y



# C3 블록
# - BottleNect1 사용(shortcut = True)
# - BottleNect2 사용(shortcut = False)
class C3(nn.Module):
    def __init__(self, c_in, c_out, n=1, shortcut=True):
        """
              c_in: 입력 채널
              c_out: 출력 채널
              n: BottleNeck 반복 개수 (x3, x6, x9 등)
              shortcut: True -> BottleNeck1, False -> BottleNeck2
        """
        super().__init__()
        c_hidden = c_out // 2  # 128 --> 64 등등..
        self.cv1 = Conv(c_in, c_hidden, k=1, s=1, p=0)
        self.cv2 = Conv(c_in, c_hidden, k=1, s=1, p=0)
        # cv1 --> BottleNeck n개로 전달
        self.m = nn.Sequential(
            *[BottleNeck(c_hidden, shortcut=shortcut) for _ in range(n)]
        )
        # Concat 후 마지막 conv(k1, s1, p0, c_out)
        self.cv3 = Conv(2 * c_hidden, c_out, k=1, s=1, p=0)

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1,y2], dim=1)) # dim=1 : 채널방향으로 cat


# SPPF 블럭
# Conv(1024 -> 512 )  -> MaxPool(k5, s1, p2) X 3  -> Concat         -> Conv(2048 -> 1024 )
#      x              ->   y1, y2, y3             -> [x, y1,y2,y3]  --> Conv() 수행 결과  x 리턴
class SPPF(nn.Module):
    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        c_hidden = c_in // 2  # 1024 --> 512
        self.cv1 = Conv(c_in, c_hidden, k=1, s=1, p=0)
        self.cv2 = Conv(c_hidden * 4, c_out, k=1, s=1, p=0)
        self.k = k

    def forward(self, x):
        x = self.cv1(x)
        y1 = F.max_pool2d(x, kernel_size=self.k, stride=1, padding=self.k//2)
        y2 = F.max_pool2d(y1, kernel_size=self.k, stride=1, padding=self.k // 2)
        y3 = F.max_pool2d(y2, kernel_size=self.k, stride=1, padding=self.k // 2)
        x = torch.cat([x, y1, y2, y3], dim=1)
        x = self.cv2(x)
        return x


class YOLOv5(nn.Module):
    def __init__(self, nc=80):
        super().__init__()

        # --------------------------
        # Backbone (왼쪽 P1 ~ P5, C3, SPPF)
        # --------------------------
        # P1 : 3 x 640 x 640  --> 64 x 320 x 320
        self.p1 = Conv(3, 64, k=6, s=2, p=2)
        # P2 : 64 x 320 x 320 --> 128 x 160 x 160
        self.p2 = Conv(64, 128, k=3, s=2, p=1)

        # C3_1( 128 x 160 x 160, BottleNeck1 x 3 )
        self.c3_1 = C3(128, 128, n=3, shortcut = True)

        # p3 : 128 x 160 x 160 --> 256 x 80 x 80
        self.p3 = Conv(128, 256, k=3, s=2, p=1)

        # C3_2 ( 256 x 80 x 80 , BottleNeck1 x 6 )
        self.c3_2 = C3(256, 256 , n=6, shortcut=True)

        # p4 : 256 x 80 x 80 ==> 512 x 40 x 40
        self.p4 = Conv(256, 512, k=3, s=2, p=1)

        # C3_3 ( 512 x 40 x 40 , BottleNeck1 x 9 )
        self.c3_3 = C3(512, 512, n=9, shortcut=True)

        # p5 : 512 x 40 x 40 ==> 1024 x 20 x 20
        self.p5 = Conv(512,1024, k=3, s=2, p=1)

        # C3_4 ( 1024 x 20 x 20, BottleNeck1 x 3 )
        self.c3_4 = C3(1024, 1024, n=3, shortcut=True)

        # SPPF ( 1024 x 20 x 20 --> 1024 x 20 x 20 )
        self.sppf = SPPF(1024, 1024, k=5)
        # ------------ Backbone 끝 ------------------


    def forward(self, x):
        # --------------------------
        # Backbone (왼쪽 P1 ~ P5, C3, SPPF)
        # --------------------------
        x = self.p1(x)  # ---> [1, 64, 320, 320]
        x = self.p2(x)  # ---> [1, 128, 160, 160]
        # C3_1 동작
        x = self.c3_1(x) # ---> [1, 128, 160, 160]

        # p3 동작
        x = self.p3(x) # -->  [1, 256, 80, 80]

        # C3_2 동작
        p3 = self.c3_2(x)   # 256 x 80 x 80 ( P3 route 출력해서 추후 사용)

        # p4 동작
        x = self.p4(p3) # --> [1, 512, 40, 40]

        # C3_3동작
        p4 = self.c3_3(x)  # 512 x 40 x 40 ( p4 route 출력해서 추후 사용)

        # p5 동작
        x = self.p5(p4)  # -->  [1, 1024, 20, 20]

        # C3_4 동작
        x = self.c3_4(x) # --> [1, 1024, 20, 20]

        # SPPF 동작
        p5 = self.sppf(x)  # --> [1, 1024, 20, 20] p5 route 출력 추후 사용
        # ------------ Backbone 끝 ------------------

        return p5



# main 동작
if __name__ == "__main__":
    model = YOLOv5(nc=80)
    #x = torch.randn(1,3,640,640)
    #out = model(x)

    summary(model, input_size=(1, 3, 640, 640))
