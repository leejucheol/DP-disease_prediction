```
model-server/
├── app/                        ← 애플리케이션 주요 코드
│   ├── __init__.py
│   ├── main.py                 ← FastAPI 앱 인스턴스 생성, 이벤트 설정
│   ├── config.py               ← 환경변수·설정(Pydantic BaseSettings)
│   ├── database.py             ← DB 엔진·세션, Redis 클라이언트 초기화
│   ├── dependencies.py         ← 공통 Depends 함수 모음
│   ├── routers/                ← API 엔드포인트 모듈
│   │   ├── __init__.py
│   │   ├── protein.py
│   │   └── disease.py
│   ├── models/                 ← SQLAlchemy ORM 모델 정의
│   │   ├── __init__.py
│   │   ├── protein.py
│   │   ├── disease.py
│   │   └── prediction.py
│   ├── schemas/                ← Pydantic 요청·응답 스키마
│   │   ├── __init__.py
│   │   ├── protein.py
│   │   └── disease.py
│   ├── services/               ← 비즈니스 로직 (모델 호출, 캐시 등)
│   │   ├── __init__.py
│   │   ├── predict.py
│   │   └── batch.py
│   └── utils/                  ← 유틸리티 함수·공통 모듈
│       ├── __init__.py
│       └── logger.py
│
├── scripts/                    ← 배치 작업·마이그레이션 스크립트
│   ├── seed_db.py
│   └── batch_predict.py
│
├── tests/                      ← 단위/통합 테스트
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_protein.py
│   └── test_predict.py
│
├── data/                       ← 원본·가공 데이터 (git 제외)
│
├── model/                      ← 학습된 모델 파일(.pt, .joblib 등)
│
├── .env                        ← 환경변수
├── requirements.txt            ← 의존성 목록
├── Dockerfile                  ← 컨테이너 이미지 정의
├── docker-compose.yml          ← 개발/운영 환경 구성
└── README.md

```

```
uvicorn app.main:app --reload
```
