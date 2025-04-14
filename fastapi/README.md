### 실행 방법

**0. 필요한 모듈 설치**

```
pip install torch
pip install torch_geometric
```

**1. 현재 위치가 fastapi가 아니라면 fastapi 디렉토리로 이동**

```
cd fastapi
```

**2. raw 데이터 전처리 실행**

```
python src/preprocessor.py
```

`data/raw/train_data.csv` 데이터셋을 `data/processed_train.csv` 로 바꿔준다. 모델 학습 시에는 `data/processed_train.csv`을 이용하고, 출력할 때는 raw도 사용하여 이름 등을 매핑하여 추가 정보를 사용자에게 보여줄 수 있다.
