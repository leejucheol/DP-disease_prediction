# 단백질 질병 예측 API 서버

이 문서는 단백질 질병 예측 모델 서버의 작동 방식과 주요 기능에 대한 설명이다.

## 서버 실행 방법

모델 서버는 FastAPI 기반으로 구현되어 있으며 다음 명령어로 실행할 수 있다:

```bash
cd model-server
uvicorn app.main:app --reload
```

기본적으로 서버는 8000번 포트에서 실행된다. 실행 후 `http://localhost:8000` 또는 `http://127.0.0.1:8000`으로 접속할 수 있다.

## 서버 동작 과정

### 초기화 과정

`main.py` 실행 시 다음 과정이 순차적으로 진행된다:

1. **데이터베이스 연결 확인**:

    - 실행 파일: `app/databases/database_connect.py`
    - 클래스: `Base` (SQLAlchemy의 declarative_base)
    - 함수/변수: `engine` (데이터베이스 연결 엔진), `get_db` (데이터베이스 세션 획득 함수)
    - 동작: SQLAlchemy를 통해 MySQL 데이터베이스에 비동기 연결을 설정하고 세션을 생성한다.

2. **테이블 존재 여부 확인**:

    - 실행 함수: `main.py`의 `check_tables_exist()` 함수
    - 동작: SQL 쿼리 `SHOW TABLES`를 실행해 필요한 테이블들이 데이터베이스에 존재하는지 확인한다.
    - 확인 테이블: 'protein', 'disease', 'protein_disease', 'predict_protein', 'model_prediction_run', 'predict_disease'

3. **테이블 생성**:

    - 실행 파일: `main.py`의 `lifespan` 함수 내 코드
    - 사용 기능: `Base.metadata.create_all` (SQLAlchemy ORM)
    - 관련 모델 클래스: `app/schema/models.py`에 정의된 `Protein`, `Disease`, `ProteinDisease`, `PredictProtein`, `ModelPredictionRun`, `PredictDisease`
    - 동작: 필요한 테이블이 없을 경우 모델 클래스 정의를 기반으로 테이블 스키마를 생성한다.

4. **기본 데이터 삽입**:
    - 실행 파일: `app/databases/insert_basic_data.py`
    - 함수: `insert_basic_data()`
    - 동작: CSV 파일에서 기본 단백질 및 질병 데이터를 읽어 데이터베이스에 삽입한다.
    - 데이터 소스: `app/data` 폴더의 CSV 파일들 (protein.csv, disease.csv, protein_disease.csv)

### API 엔드포인트

서버는 다음과 같은 API 엔드포인트를 제공한다:

1. **루트 엔드포인트 (`/`)**:

    - 정의 위치: `app/main.py`의 `root()` 함수
    - 데코레이터: `@app.get("/")`
    - 서버 연결 상태를 확인하는 엔드포인트다.
    - 응답: `{"message": "단백질 예측 프로그램 연결 성공"}`

2. **단백질 질병 예측 엔드포인트 (`/proteins/predict`)**:
    - 정의 위치: `app/routers/proteins.py`의 `predict_disease()` 함수
    - 라우터: `router = APIRouter(prefix="/proteins", tags=["proteins"])`
    - 데코레이터: `@router.post("/predict", response_model=DiseasePredictionResponse)`
    - 입력 모델: `app/schema/sequence_input_dto.py`의 `SequenceInput` 클래스
    - 출력 모델: `app/schema/sequence_input_dto.py`의 `DiseasePredictionResponse` 클래스
    - 기능: 입력된 단백질 시퀀스를 분석하여 관련 질병을 예측한다.
    - 내부 처리:
        - 단백질 정보를 데이터베이스에 저장 (`Protein` 테이블)
        - GCN 모델을 사용하여 질병 예측 (`predict_top5_diseases` 함수 호출)
        - 예측 실행 정보 저장 (`ModelPredictionRun` 테이블)
        - 예측된 단백질 정보 저장 (`PredictProtein` 테이블)
        - 예측된 질병 정보 저장 (`PredictDisease` 테이블)
    - HTTP 메서드: POST
    - 요청 형식: JSON 객체 `{ "sequence": "단백질 시퀀스 문자열" }`
    - 응답: 관련 질병 예측 결과(상위 5개)와 확률을 반환한다.

## 예측 모델 동작 방식

예측 모델은 다음과 같은 단계로 동작한다:

1. **ESM 모델 사용**:

    - 실행 파일: `app/models/gcn_v0_1_0.py`
    - 사용 모델: `esm2_t6_8M_UR50D` (ESM 모델)
    - 주요 함수: `sequence_to_graph_with_esm()`
    - 동작: 단백질 시퀀스를 ESM(Evolutionary Scale Modeling) 모델에 입력해서 각 아미노산 서열의 임베딩 벡터를 얻는다.

2. **그래프 변환**:

    - 실행 함수: `sequence_to_graph_with_esm()` 내부 로직
    - 사용 라이브러리: PyTorch Geometric의 `Data` 클래스
    - 동작: 임베딩된 단백질 시퀀스를 노드(아미노산)와 엣지(아미노산 간 연결)로 구성된 그래프 형태로 변환한다.
    - 그래프 구성 방식: 아미노산 시퀀스에서 인접한 아미노산끼리 엣지로 연결한다.

3. **GCN 모델**:

    - 실행 함수: `predict_top5_diseases()`
    - 모델 클래스: `gcn_v0_1_0.py` 내 정의된 GCN 모델
    - 사용 레이어: PyTorch Geometric의 `GCNConv` (그래프 합성곱 레이어)
    - 동작: 그래프 기반 합성곱 신경망(GCN)을 통해 단백질 그래프를 분석하고 질병과의 연관성 점수를 계산한다.

4. **결과 반환**:
    - 실행 함수: `predict_top5_diseases()` 결과 부분
    - 동작: 모델이 예측한 모든 질병 중에서 확률이 가장 높은 상위 5개 질병을 선택하고, 질병 ID, 질병 이름, 예측 확률을 포함한 결과를 반환한다.
    - API 응답 처리: `app/routers/proteins.py`의 `predict_disease` 함수에서 모델 예측 결과를 받아 API 응답 형식으로 변환한다.

## 데이터베이스 구조

서버는 다음과 같은 테이블 구조를 사용한다:

1. **protein**: 단백질 정보 저장 (uniprot_id, sequence)
2. **disease**: 질병 정보 저장 (disease_id, disease_name)
3. **protein_disease**: 단백질-질병 간 관계 저장
4. **predict_protein**: 예측에 사용된 단백질 정보
5. **model_prediction_run**: 모델 예측 실행 정보
6. **predict_disease**: 예측된 질병 정보

## 예측 결과 저장

사용자가 단백질 시퀀스 예측을 요청하면 다음 정보가 데이터베이스에 저장된다:

1. 입력된 단백질 시퀀스
2. 예측 실행 정보(날짜, 모델 버전 등)
3. 예측된 질병 목록과 순위

## 기술 스택

-   **FastAPI**: RESTful API 서버 프레임워크
-   **SQLAlchemy**: ORM(Object-Relational Mapping) 라이브러리
-   **PyTorch & PyTorch Geometric**: 딥러닝 및 그래프 신경망 라이브러리
-   **ESM**: 단백질 시퀀스 임베딩 모델
-   **MySQL**: 데이터베이스

## 오류 처리

서버는 예측 과정에서 발생할 수 있는 오류를 처리하며, 오류 발생 시 데이터베이스 rollback을 수행하고 적절한 HTTP 오류 응답을 반환한다.
