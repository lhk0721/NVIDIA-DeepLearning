import numpy as np
import matplotlib as plt

##
arr = np.arange(5,19).reshape(7,2)

# print(arr)
# [[ 5  6]
#  [ 7  8]
#  [ 9 10]
#  [11 12]
#  [13 14]
#  [15 16]
#  [17 18]]

# numpy는 수치 인덱싱밖에 지원하지 않는다. 암묵적인덱스밖에 없어 수치로만 접근해야 한다.
# 연결되어있으면 슬라이싱하면 되지만 떨어져있다면? fancyindexing 사용.

## fancyindexing -> 추출할 위치를 array-like로 전달.

# 튜플은 안됨. 서로 다른 축에 대한 인덱스로 해석할 거임. 리스트로 주자.
# ((1,3,5),0) 와 같이 더 큰 인덱싱 튜플의 한 자리에 들어가면, arraylike로 취급된다. 넘파이에서 최상위 튜플은 축 구분 문법으로 이해되기 때문. 대괄호 안은 항상 객체 하나다.

# arr 

# index = (1,3,5)

index = [1,3,5]
xdata = arr[index,0]
ydata = arr[index,1]


print(xdata) # [7 11 15]

print(ydata) # [8 12 16]