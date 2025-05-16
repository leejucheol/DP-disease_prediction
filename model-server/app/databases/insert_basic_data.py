"""
기본 데이터 삽입 스크립트
"""
import os
import pandas as pd
import numpy as np
import csv
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.databases.database_connect import engine, AsyncSessionLocal
from app.schema.models import Protein, Disease, ProteinDisease
from sqlalchemy.exc import IntegrityError

async def load_csv_data():
    """CSV 파일에서 데이터를 로드합니다."""
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    # 파일 경로 생성
    disease_file = os.path.join(base_path, 'disease.csv')
    protein_file = os.path.join(base_path, 'protein.csv')
    protein_disease_file = os.path.join(base_path, 'protein_disease.csv')
    
    # 데이터 로드
    try:
        # Disease 데이터 로드
        disease_df = pd.read_csv(disease_file, encoding='utf-8')
        disease_df.columns = ['disease_id', 'disease_name']  # 열 이름 재설정

        # Protein 데이터 로드
        protein_df = pd.read_csv(protein_file, encoding='utf-8')
        protein_df.columns = ['uniprot_id', 'sequence']  # 열 이름 재설정        # Protein-Disease 매핑 데이터 로드
        protein_disease_df = pd.read_csv(protein_disease_file, encoding='utf-8')
        print(f"Protein-Disease CSV 열: {protein_disease_df.columns.tolist()}")  # 열 이름 확인
        if 'UniProt_ID' in protein_disease_df.columns and 'Disease ID' in protein_disease_df.columns:
            protein_disease_df = protein_disease_df[['UniProt_ID', 'Disease ID']]  # 필요한 열만 선택
            protein_disease_df.columns = ['uniprot_id', 'disease_id']  # 열 이름 재설정

        return disease_df, protein_df, protein_disease_df
    
    except Exception as e:
        print(f"CSV 파일 로드 중 오류 발생: {e}")
        return None, None, None

async def check_data_exists():
    """데이터베이스에 데이터가 이미 존재하는지 확인합니다."""
    async with AsyncSessionLocal() as session:
        # Disease 테이블 데이터 확인
        disease_count = await session.execute(select(Disease).limit(1))
        disease_exists = disease_count.scalars().first() is not None
        
        # Protein 테이블 데이터 확인
        protein_count = await session.execute(select(Protein).limit(1))
        protein_exists = protein_count.scalars().first() is not None
        
        return disease_exists and protein_exists

async def insert_basic_data():
    """기본 데이터를 데이터베이스에 삽입합니다."""
    # 데이터가 이미 존재하는지 확인
    data_exists = await check_data_exists()
    if data_exists:
        print("데이터가 이미 데이터베이스에 존재합니다.")
        return
    
    # CSV 파일에서 데이터 로드
    disease_df, protein_df, protein_disease_df = await load_csv_data()
    if disease_df is None or protein_df is None or protein_disease_df is None:
        print("데이터 로드 실패")
        return
    
    try:
        async with AsyncSessionLocal() as session:
            # Disease 데이터 삽입
            print("질병 데이터 삽입 중...")
            disease_count = 0
            for _, row in disease_df.iterrows():
                try:
                    disease = Disease(
                        disease_id=row['disease_id'],
                        disease_name=row['disease_name']
                    )
                    session.add(disease)
                    disease_count += 1
                    # 100개 단위로 커밋하여 메모리 사용량 관리
                    if disease_count % 100 == 0:
                        await session.commit()
                except IntegrityError:
                    await session.rollback()
                    print(f"중복된 질병 ID 건너뜀: {row['disease_id']}")
                except Exception as e:
                    await session.rollback()
                    print(f"질병 데이터 삽입 오류: {e}")
            
            # 남은 데이터 커밋
            await session.commit()
            
            # Protein 데이터 삽입
            print("단백질 데이터 삽입 중...")
            protein_count = 0
            inserted_count = 0
            skipped_count = 0
            
            # 이미 존재하는 단백질 ID 확인
            existing_proteins = {}
            result = await session.execute(select(Protein.uniprot_id))
            for row in result.scalars():
                existing_proteins[row] = True
                
            for _, row in protein_df.iterrows():
                protein_count += 1
                # 이미 존재하는 단백질이면 건너뜀
                if row['uniprot_id'] in existing_proteins:
                    skipped_count += 1
                    if skipped_count % 100 == 0:
                        print(f"중복 단백질 건너뜀: {skipped_count}개")
                    continue
                
                try:
                    protein = Protein(
                        uniprot_id=row['uniprot_id'],
                        sequence=row['sequence']
                    )
                    session.add(protein)
                    inserted_count += 1
                    existing_proteins[row['uniprot_id']] = True
                    
                    # 100개 단위로 커밋하여 메모리 사용량 관리
                    if inserted_count % 100 == 0:
                        await session.commit()
                        print(f"단백질 {inserted_count}개 삽입 완료")
                except IntegrityError:
                    await session.rollback()
                    skipped_count += 1
                except Exception as e:
                    await session.rollback()
                    print(f"단백질 데이터 삽입 오류: {row['uniprot_id']} - {e}")
            
            # 남은 데이터 커밋
            await session.commit()
            print(f"단백질 데이터 삽입 완료: 총 {protein_count}개 중 {inserted_count}개 삽입, {skipped_count}개 건너뜀")
            
            # Protein-Disease 매핑 데이터 삽입
            print("단백질-질병 매핑 데이터 삽입 중...")
            mapping_count = 0
            mapping_success = 0
            mapping_failed = 0
            
            for _, row in protein_disease_df.iterrows():
                mapping_count += 1
                try:
                    # 단백질이 존재하는지 확인
                    protein_exists = await session.execute(
                        select(Protein).where(Protein.uniprot_id == row['uniprot_id'])
                    )
                    protein = protein_exists.scalar_one_or_none()
                    
                    # disease_id가 nan인 경우에도 허용
                    disease_exists = True
                    if isinstance(row['disease_id'], str) or not np.isnan(row['disease_id']):
                        disease_check = await session.execute(
                            select(Disease).where(Disease.disease_id == row['disease_id'])
                        )
                        disease_exists = disease_check.scalar_one_or_none() is not None
                    
                    # 단백질이 존재하면 매핑 추가 (disease_id가 nan이어도 허용)
                    if protein:
                        protein_disease = ProteinDisease(
                            uniprot_id=row['uniprot_id'],
                            disease_id=row['disease_id'] if isinstance(row['disease_id'], str) or not np.isnan(row['disease_id']) else None
                        )
                        session.add(protein_disease)
                        mapping_success += 1
                        
                        # 100개 단위로 커밋
                        if mapping_success % 100 == 0:
                            await session.commit()
                            print(f"매핑 {mapping_success}개 삽입 완료")
                    else:
                        mapping_failed += 1
                        if mapping_failed % 100 == 0:
                            print(f"매핑 실패 (존재하지 않는 참조): {mapping_failed}개")
                except IntegrityError:
                    await session.rollback()
                    mapping_failed += 1
                except Exception as e:
                    await session.rollback()
                    mapping_failed += 1
                    print(f"매핑 데이터 삽입 실패: {e} - UniProt ID: {row['uniprot_id']}, Disease ID: {row.get('disease_id', 'None')}")
            
            # 남은 데이터 커밋
            await session.commit()
            print(f"매핑 데이터 삽입 완료: 총 {mapping_count}개 중 {mapping_success}개 성공, {mapping_failed}개 실패")
            
        print("기본 데이터 삽입 완료")
    
    except Exception as e:
        print(f"데이터 삽입 중 오류 발생: {e}")
