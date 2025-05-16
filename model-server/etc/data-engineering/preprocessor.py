import pandas as pd

train_data = pd.read_csv('./data/raw/train_data.csv')

print(f"\nraw train data 전체 컬럼:\n{train_data.columns}\n")

# protein id 부분에 있는 "9606." 문자열 제거
def processing_protein_and_save(df):
    df["protein1"] = df["protein1"].str.replace("9606.", "", regex=False)
    df["protein2"] = df["protein2"].str.replace("9606.", "", regex=False)

# df, main_path: 드롭 후 최종 테이블 저장 경로
def delete_columns_and_save(df, main_path, small_sample_path=None):
    # 삭제할 컬럼들 정의
    cols_to_drop = [
        'PubMed_IDs', 'entryId', 'sequenceChecksum', 'sequenceVersionDate',
        'uniprotDescription', 'organismScientificName', 'globalMetricValue',
        'uniprotStart', 'uniprotEnd', 'modelCreatedDate','uniprotSequence',
        'latestVersion', 'allVersions', '_version_', 'proteinShortNames',
        'uniprotAccession_unchar', 'entry_name', 'protein_name', 'organism',
        'protein_existence', 'sequence_version', 'uniprotAccession', 'uniprotId',
        'gene_x', 'gene_y', 'organismScientificNameT', 'tax_id', 'taxId', 'organismCommonNames', 'isAMdata', 
        'Protein_ID_Formatted', 'isReviewed', 'isReferenceProteome', 'geneSynonyms'
    ]

    # 유지할 컬럼 정의
    cols_to_keep = [col for col in df.columns if col not in cols_to_drop]
    
    print("유지할 컬럼:", cols_to_keep)
    print("삭제할 컬럼:", cols_to_drop)

    # protein 컬럼 가공
    processing_protein_and_save(df)
    print('protein 컬럼을 가공하였습니다.')

    df_main = pd.DataFrame(df[cols_to_keep].copy())

    # 잘 저장되었는지 확인
    print(f"main columns shape: {df_main.shape}")

    df_main.to_csv(main_path, index=False)
    print(f"✅ 전체 데이터 저장 완료: {main_path}")

    # 무작위 1% 샘플 저장
    if small_sample_path:
        df_small = df_main.sample(frac=0.01, random_state=42)
        df_small.to_csv(small_sample_path, index=False)
        print(f"✅ 무작위 1% 샘플 저장 완료: {small_sample_path}")

# 실행
delete_columns_and_save(
    train_data,
    main_path="./data/processed_train.csv",
    small_sample_path="./data/processed_train_small.csv"
)
