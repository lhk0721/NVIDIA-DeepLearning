from konlpy.tag import Okt
import os

okt = Okt()
wordlist = okt.morphs('파이썬을 활용한 한국어 형태소 분석 예제입니다.', norm=True, stem=True)
# print(wordlist)
# ['파이썬', '을', '활용', '한', '한국어', '형태소', '분석', '예제', '이다', '.']

wordlist = okt.pos('파이썬을 활용한 한국어 형태소 분석 예제입니다.', norm=True, stem=True)
# print(wordlist)
# [('파이썬', 'Noun'), ('을', 'Josa'), ('활용', 'Noun'), ('한', 'Josa'), ('한국어', 'Noun'), ('형태소', 'Noun'), ('분석', 'Noun'), ('예제', 'Noun'), ('이다', 'Adjective'), ('.', 'Punctuation')]

nounlist = okt.nouns('파이썬을 활용한 한국어 형태소 분석 예제입니다.')
# print(nounlist)
# ['파이썬', '활용', '한국어', '형태소', '분석', '예제']

nounlist2 = okt.nouns('아자쓰! 버터떡 두쫀쿠 완전 야르다')
# print(nounlist2)
# ['버터', '떡', '두쫀쿠', '완전', '야르']

##

import re
