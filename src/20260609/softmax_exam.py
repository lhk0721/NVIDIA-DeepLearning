import numpy as np
# softmax 함수 동작
# 여러개의선행방정식출력값을 0 ~ 1 로압축하고전체합이 1이되도록만듬
# 이를위해지수함수를사용하기때문에정규화된지수함수
mylist = [-6.5, 1.03, 5.16,-2.73, 3.34, 0.33,-0.63] # 예를 들어) 각 클래스별 Z 값
arr = np.array(mylist)

exp_a = np.exp(arr)  # 각 클래스별 Z값 지수함수 적용
sum_exp_a = np.sum(exp_a) # 각 클래스별 지수함수 적용 합 계산
y = exp_a / sum_exp_a # 각 클래스별 지수함수 / 합 ==> 확률 계산
print(y)
y = np.round(y, decimals=3)
print(y)
# ==> 확율이 가장 큰 인덱스를 출력
classes = np.argmax(y)  # 전달해준 argument 중 가장 큰 값의 인덱스를 반환
print(classes)
classdata = ['red', 'blue', 'green', 'black']
print( classdata[np.argmax(y)] )
