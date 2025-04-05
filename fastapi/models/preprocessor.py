import pandas as pd

# 파일 경로
path_large = "./data/train_data.csv"
path_small = "./data/train_data_small.csv"

# 데이터 불러오기
train_data = pd.read_csv(path_large)
train_data_small = pd.read_csv(path_small)

# # 고유 인덱스 컬럼 추가
# train_data.insert(0, "index", range(len(train_data)))
# train_data_small.insert(0, "index", range(len(train_data_small)))

# # 덮어쓰기 저장
# train_data.to_csv(path_large, index=False)
# train_data_small.to_csv(path_small, index=False)

# print("✅ index 컬럼 추가 및 저장 완료!")

# drop_cols = [
#     "organismScientificNameT", "tax_id", "taxId", "organismCommonNames", "isAMdata",
#     "Protein_ID_Formatted", "isReviewed", "isReferenceProteome", "geneSynonyms"
# ]

# # ✅ dump 파일은 drop 컬럼 + index 만 추출
# train_data_dump = train_data[["index"] + drop_cols]
# train_data_small_dump = train_data_small[["index"] + drop_cols]

# # ✅ 원본은 drop 컬럼 제거 후 저장
# train_data_cleaned = train_data.drop(columns=drop_cols)
# train_data_small_cleaned = train_data_small.drop(columns=drop_cols)

# # # 파일 저장
# # train_data_dump.to_csv("./data/train_data_dump.csv", index=False)
# # train_data_small_dump.to_csv("./data/train_data_dump_small.csv", index=False)
# train_data_cleaned.to_csv(path_large, index=False)
# train_data_small_cleaned.to_csv(path_small, index=False)

# print("✅ index 포함된 dump 저장 + 컬럼 제거 후 원본 덮어쓰기 완료!")

# "9606." 접두어 제거 함수
def remove_prefix(df):
    df["protein1"] = df["protein1"].str.replace("9606.", "", regex=False)
    df["protein2"] = df["protein2"].str.replace("9606.", "", regex=False)
    return df

# 적용
train_data = remove_prefix(train_data)
train_data_small = remove_prefix(train_data_small)

# 덮어쓰기 저장
train_data.to_csv(path_large, index=False)
train_data_small.to_csv(path_small, index=False)

print("✅ protein1, protein2 컬럼에서 '9606.' 제거 완료 및 저장됨!")