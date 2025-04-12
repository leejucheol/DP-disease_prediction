import pandas as pd


# 1. 데이터 로드
raw_train_data = pd.read_csv('./data/raw/train_data.csv')
uncharacterized_proteins_data = pd.read_csv('./data/raw/uncharacterized_proteins.csv')

if(not raw_train_data.empty and not uncharacterized_proteins_data.empty):
    print(">>>>>> 데이터를 성공적으로 로드 하였습니다.")

# 2. train_data에 gene_x, gene_y 두 개가 있음. 중복 컬럼 처리
if 'gene_x' in raw_train_data.columns and 'gene_y' in raw_train_data.columns:
    raw_train_data = raw_train_data.drop(columns=['gene_y'])
    raw_train_data = raw_train_data.rename(columns={'gene_x': 'gene'})
    print("🔄 'gene_y' 삭제, 'gene_x'를 'gene'으로 이름 변경했습니다.")

elif 'gene_x' in raw_train_data.columns:
    raw_train_data = raw_train_data.rename(columns={'gene_x': 'gene'})
    print("🔄 'gene_x'를 'gene'으로 이름 변경했습니다.")

elif 'gene_y' in raw_train_data.columns:
    raw_train_data = raw_train_data.rename(columns={'gene_y': 'gene'})
    print("🔄 'gene_y'를 'gene'으로 이름 변경했습니다.")

else:
    print("⚠️ 'gene_x' 또는 'gene_y' 컬럼이 존재하지 않습니다.")

# 3. 병합 전, 안 겹치는 컬럼 확인
# 컬럼 집합 생성
train_cols = set(raw_train_data.columns)
unchar_cols = set(uncharacterized_proteins_data.columns)

# 차이 계산
only_in_train = train_cols - unchar_cols
only_in_unchar = unchar_cols - train_cols

# 결과 출력
if train_cols == unchar_cols:
    print("✅ 두 데이터프레임의 컬럼은 동일합니다.")
else:
    print("⚠️ 두 데이터프레임의 컬럼이 다릅니다.\n")

    if only_in_train:
        print("🔹 raw_train_data에만 있는 컬럼:")
        for col in sorted(only_in_train):
            print(f" - {col}")

    if only_in_unchar:
        print("\n🔹 uncharacterized_proteins_data에만 있는 컬럼:")
        for col in sorted(only_in_unchar):
            print(f" - {col}")