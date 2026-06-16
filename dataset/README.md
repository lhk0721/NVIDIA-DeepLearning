# 데이터셋 목록

> 이 디렉터리는 용량이 커서 (총 **약 302MB / 10,043개 파일**) Git 저장소에 직접 커밋하지 않습니다.
> 아래 목록을 참고해 각자 데이터를 내려받아 `dataset/` 아래에 배치하세요.

생성일: 2026-06-16

---

## 🖼️ 이미지 데이터셋

| 폴더 | 용량 | 파일 수 | 설명 |
|------|------|---------|------|
| `cats_and_dogs/` | 237MB | 10,006 | 고양이/개 이미지 분류 (CNN) |
| `testimage_dataset-20260616T081915Z-3-001/` | 532KB | 9 | 추론 테스트용 이미지 (cat/dog/mushroom 각 3장) |

### `cats_and_dogs/` 세부 구조
```
cats_and_dogs/
├── training_set/
│   ├── cats/   (4,000장)
│   └── dogs/   (4,000장)
├── test_set/
│   ├── cats/   (1,000장)
│   └── dogs/   (1,000장)
└── temp/       (6장)
```
- 출처: Kaggle "Dogs vs. Cats" (Microsoft / Kaggle)

---

## 📝 텍스트 / NLP 데이터셋

| 파일 | 용량 | 설명 |
|------|------|------|
| `IMDB/IMDB Dataset.csv` | 64MB | IMDB 영화 리뷰 감성분석 (positive/negative 50,000건) |
| `spam/spam.csv` | 472KB | SMS 스팸 분류 (ham/spam) |
| `spam.zip` | 208KB | 위 spam 데이터 압축본 |

---

## 📊 정형 데이터 (CSV / Excel)

| 파일 | 용량 | 설명 |
|------|------|------|
| `Fish.csv` | 6KB | 생선 종 분류/회귀 (무게·길이 등) |
| `Health_info.csv` | 127B | 건강 정보 샘플 |
| `basketball_stat.csv` | 4KB | 농구 선수 스탯 (포지션 분류) |
| `scoreData.csv` | 40B | 점수 샘플 |
| `tips.csv` | 8KB | 식당 팁 데이터 (seaborn tips) |
| `seoul_keumchun_gas_info.csv` | 1KB | 서울 금천구 가스 정보 |
| `서울특별시_지하철 승하차 승객수.csv` | 4KB | 지하철 승하차 인원 |
| `Mydata.xlsx` | 5KB | 실습용 샘플 |
| `mydf.xlsx` | 5KB | 실습용 샘플 |
| `readData.xlsx` | 80B | 읽기 실습용 |
| `population_in_seoul.xls` | 36KB | 서울 인구 통계 |
| `salesfunnel.xlsx` | 6KB | 영업 퍼널 (피벗 실습) |
| `teacher_list_pivot_exam.xlsx` | 11KB | 피벗테이블 실습 |
| `teacher_list_pivot_exam_1.xlsx` | 9KB | 피벗테이블 실습 |
| `naverNews.xlsx` | 6KB | 네이버 뉴스 크롤링 |
| `naverNews_filtering.xlsx` | 5KB | 네이버 뉴스 필터링 결과 |
| `youtube_data.xlsx` | 48KB | 유튜브 데이터 |
| `youtubedata.xlsx` | 7KB | 유튜브 데이터 |
| `youtube_rank_1000.xlsx` | 64KB | 유튜브 랭킹 1000 |
| `국소마취제_groupby.xlsx` | 19KB | 국소마취제 groupby 실습 |
| `반도체_제어_이력.xlsx` | 11KB | 반도체 제어 이력 |

---

## 📦 압축본 / 기타

| 파일 | 용량 | 비고 |
|------|------|------|
| `spam.zip` | 208KB | `spam/`의 원본 압축 |
| `testimage_dataset-...-001.zip` | 524KB | `testimage_dataset.../`의 원본 압축 |
| `*.Zone.Identifier` | - | Windows 다운로드 메타데이터 (불필요, 삭제 가능) |
