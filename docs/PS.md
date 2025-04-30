# Project Server Structure

## Server Response

### 단백질 입력 → OUTPUT

```json
{
  "input":      { "type": "protein", "id": "P12345" },
  "proteins": [
    { "id": "P67890", "score": 0.92 },
    …
  ],
  "diseases": [
    { "id": "D00123", "score": 0.88 },
    …
  ]
}

```

### 질병 입력 → OUTPUT

```json
{
  "input":      { "type": "disease", "id": "D00123" },
  "proteins": [
    { "id": "P12345", "score": 0.91 },
    …
  ]
}

```

---

## Server API

| Method | URL                                  | Query Params                                                                               | 설명                                                                  |
| ------ | ------------------------------------ | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| GET    | `/proteins/{protein_id}/predictions` | `model_version` (optional)<br>`top_k_proteins` (default=5)<br>`top_k_diseases` (default=5) | 지정한 단백질에 대해<br>• 관련 단백질 Top-k<br>• 관련 질병 Top-k 반환 |
| GET    | `/diseases/{disease_id}/predictions` | `model_version` (optional)<br>`top_k_proteins` (default=5)                                 | 지정한 질병에 대해<br>• 관련 단백질 Top-k 반환                        |
| GET    | `/proteins`                          | `page`, `size`, `search`                                                                   | 단백질 목록 조회 (페이징, 이름 검색 등)                               |
| GET    | `/diseases`                          | `page`, `size`, `search`                                                                   | 질병 목록 조회 (페이징, 이름 검색 등)                                 |
| GET    | `/model-versions`                    | —                                                                                          | 사용 가능한 모델 버전 목록 조회                                       |
| GET    | `/model-versions/{version_id}`       | —                                                                                          | 특정 모델 버전 메타데이터 조회                                        |
| GET    | `/predictions`                       | `protein_id`, `disease_id`, `model_version`,<br>`top_k_proteins`, `top_k_diseases`         | 범용 예측 조회 엔드포인트 (단백질·질병 둘 다 파라미터로)              |
| POST   | `/predictions/batch`                 | — (body에 ID 리스트 및 옵션)                                                               | 배치 예측 요청 (여러 단백질·질병 한 번에 처리)                        |
| GET    | `/health`                            | —                                                                                          | 서비스 상태(헬스체크)                                                 |
