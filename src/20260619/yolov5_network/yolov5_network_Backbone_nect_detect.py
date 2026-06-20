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


# ----------------------------------------------------
# Detect Head (그림 오른쪽 초록색 Conv 1x1)
#   - 각 스케일: Conv(layerscale ,c=(5+nc)*3 , 1)  , 1 ==> 1x1 kernel size
#   - 이 코드는 decode 없이 raw tensor만 뽑음
# ----------------------------------------------------
class Detect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):
        # nc: 클래스 개수
        # anchors: (nl, na*2) 리스트
        # nl: detection 스케일 layer수 ( 3 )
        # na: 각 스케일 당 anchor 수 (3)
        # ch: 각 스케일 feature 채널 수 [256, 512, 1024]

        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors) # detection layer 수(3)
        self.na = len(anchors[0]) // 2   # anchor per scale 수(3)

        # anchor 연산을 위한 tensor 생성 ==> 문서 참조
        anchor_tensor = torch.tensor(anchors, dtype=torch.float32).view(self.nl, self.na, 2)
        self.register_buffer("anchors", anchor_tensor)
        self.register_buffer("anchor_grid", anchor_tensor.clone().view(self.nl, 1, self.na, 1, 1, 2))

        # Conv( ch[i] --> (5+nc)*na )
        self.m = nn.ModuleList( [
            nn.Conv2d(c, self.no * self.na, 1) for c in ch
        ])  # 1 ==> 1x1 커널 사이즈

    def forward(self, x):
        # x: [P3_out(n3), P4_out(n4_2), P5_out(n5_2)]
        z = []
        for i in range(self.nl):
            bs, _, h, w = x[i].shape  # [1, 256, 80, 80] -> [1, 512, 40, 40] -> [1, 1024, 20, 20]
            p = self.m[i](x[i])
            #print('p shape : ', p.shape) # [1, 255, 80, 80] -> [1, 255, 40, 40] -> [1, 255, 20, 20]
            p = p.view(bs, self.na, self.no, h, w)
            #print('p shape : ', p.shape) # [1, 3, 85, 80, 80] -> ....
            # permute는 차원 순서를 재배열
            # 인덱스 번호 ==> 0: B , 1: na, 2: no, 3: H, 4: W

            # 나중에 후처리할 때 편하게 하기 위해
            # [..., 0: 4]: box(tx, ty, tw, th)
            # [..., 4]: objectness
            # [..., 5:]: class score

            p = p.permute(0,1,3,4,2).contiguous() # (B, na, H, W, no)
            z.append(p)
            print(p.shape)
            # z[0]: (B, na, 80, 80, no)
            # z[1]: (B, na, 40, 40, no)
            # z[2]: (B, na, 20, 20, no)
        return z

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

        #  ---------  Neck 시작  ---------------------
        # SPPF출력(1024) 를 512로 줄이는 Conv
        self.p5_conv = Conv(1024, 512, k=1, s=1, p=0)

        # FPN 위로 올릴 때 쓸 C3 ( C3_n1 )
        # BottleNeck2 ( shortcut = False ) 사용
        self.c3_n1 = C3(1024, 512, n=3, shortcut=False) #  # 512 x 40 x 40 out

        # c3_n1 출력을  → p3 으로 올릴 때 사용 ==>  p4_r
        # 512 x 40 x 40 --> 256 x 40 x 40 변환
        self.up_p4_1 = Conv(512, 256, k=1,s=1,p=0)

        #  ( C3_n2 )
        # BottleNeck2 ( shortcut = False ) 사용
        self.c3_n2 = C3(512, 256, n=3, shortcut=False) # 80x80x256 , small Detector용 FM

        # 아래로 bottom
        self.down_n3 = Conv(256, 256, k=3, s=2, p=1)  # 80x80x256 -> 40x40x256

        #  ( C3_n3 )
        # BottleNeck2 ( shortcut = False ) 사용
        self.c3_n3 = C3(512, 512, n=3, shortcut=False)  # 40x40x512 output , (medium Detector용 FM)

        # 아래로 bottom
        self.down_n4 = Conv(512, 512, k=3, s=2, p=1)  # 40x40x512 -> 20x20x512

        #  ( C3_n4 )
        # BottleNeck2 ( shortcut = False ) 사용
        self.c3_n4 = C3(1024,1024, n=3, shortcut=False) # 20x20x1024 output , (large Detector용 FM)

        # --------------------------
        # Detect Head (green Conv 1x1)
        # --------------------------
        anchors = [
            [10, 13, 16, 30, 33, 23],  # P3 (80x80)
            [30, 61, 62, 45, 59, 119],  # P4 (40x40)
            [116, 90, 156, 198, 373, 326],  # P5 (20x20)
        ]

        ch = [256, 512, 1024]  # 각 스케일 채널
        self.detect = Detect(nc = nc, anchors=anchors, ch=ch)

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

        #  ---------  Neck 시작  ---------------------
        # 위로 보내는 로직 ( FPN )
        p5_r = self.p5_conv(p5)  # --> [1, 512, 20, 20]

        #  Upsample 20 x 20 --> 40 x 40
        x = F.interpolate(p5_r, scale_factor=2, mode='nearest') # 512 x 40 x 40
        # 위에서 route 한 p4 ( 512 x 40 x 40 ) 와 병합
        x = torch.cat([x, p4], dim=1) # --> 1024 x 40 x 40

        # c3_n1 동작
        n4 = self.c3_n1(x)  # # 512 x 40 x 40 out
        p4_r = self.up_p4_1(n4)    # 256x40x40 out

        #  Upsample 40 x 40 --> 80 x 80
        x = F.interpolate(p4_r, scale_factor=2, mode='nearest') # 80 x 80 x 256
        x = torch.cat([x, p3], dim=1) # # 80x80x512

        # C3_n2 동작
        n3 = self.c3_n2(x)  # 256x80x80  (small Detector용 FM)

        # bottom : down_n3 동작
        x = self.down_n3(n3)   # 40x40x256 output
        x = torch.cat([x, p4_r], dim=1) # 40x40x256 + 40x40x256 = 40x40x512

        # C3_n3 동작
        n4_2 = self.c3_n3(x)  # 40x40x512  (medium Detector용 FM)

        # bottom : down_n4 동작
        x = self.down_n4(n4_2)  # 20x20x512 output
        x = torch.cat([x, p5_r], dim=1)  # 20x20x512 + 20x20x512 = 20x20x1024

        # 마지막 C3_n4 동작
        n5_2 = self.c3_n4(x)  #  20x20x1024 (large Detector용 FM)
        #  ---------  Neck 끝  ---------------------

        # # -------- Detect Head 시작 --------
        preds = self.detect([n3, n4_2, n5_2])

        return preds



# main 동작
if __name__ == "__main__":
    model = YOLOv5(nc=80)
    #x = torch.randn(1,3,640,640)
    #out = model(x)

    summary(model, input_size=(1, 3, 640, 640))
