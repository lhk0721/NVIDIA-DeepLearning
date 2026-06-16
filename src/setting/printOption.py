import pandas as pd
import numpy as np

pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 1000)
pd.set_option('display.colheader_justify','center')  # 컬럼 중앙 출력
pd.set_option('display.float_format', '{:.3f}'.format) #  float 형식 소숫점 3자리 표현

# 소숫점 이하 3자리까지 출력
np.set_printoptions(precision=3)

np.set_printoptions(threshold=np.inf) #무한으로 출력합니다. (sys.maxsize 크기 만큼 출력

# 소숫점의 과학적 표기법의 사용 억제
np.set_printoptions(suppress=True)


# Dataframe 객체 예쁘게 출력하는 방법
from prettytable import PrettyTable  # 3.3.0 version 설치

def prinf_df(df):
    table = PrettyTable(['']+ list(df.columns))
    for row in df.itertuples():
        table.add_row(row)
    print(str(table))
    print()

prinf_df(df)  # df(데이터프레임) 객체를 테이블 형태로 알아서 예쁘게 출력