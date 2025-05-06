from fastapi import APIRouter, Query

router = APIRouter()

@router.get("/proteins/{protein_id}/predictions")
async def get_protein_predictions(
    protein_id: str,
    model_version: str = Query(None),
    top_k_proteins: int = Query(5),
    top_k_diseases: int = Query(5),
):
    """
    지정한 단백질에 대해 관련 단백질 Top-k와 관련 질병 Top-k 반환
    """
    return {
        "protein_id": protein_id,
        "model_version": model_version,
        "top_k_proteins": top_k_proteins,
        "top_k_diseases": top_k_diseases,
        "proteins": [{"id": "P67890", "score": 0.92}],
        "diseases": [{"id": "D00123", "score": 0.88}],
    }


@router.get("/diseases/{disease_id}/predictions")
async def get_disease_predictions(
    disease_id: str,
    model_version: str = Query(None),
    top_k_proteins: int = Query(5),
):
    """
    지정한 질병에 대해 관련 단백질 Top-k 반환
    """
    return {
        "disease_id": disease_id,
        "model_version": model_version,
        "top_k_proteins": top_k_proteins,
        "proteins": [{"id": "P12345", "score": 0.91}],
    }


@router.get("/proteins")
async def get_proteins(page: int = Query(1), size: int = Query(10), search: str = Query(None)):
    """
    단백질 목록 조회 (페이징, 이름 검색 등)
    """
    return {
        "page": page,
        "size": size,
        "search": search,
        "proteins": [{"id": "P12345", "name": "Protein A"}],
    }


@router.get("/diseases")
async def get_diseases(page: int = Query(1), size: int = Query(10), search: str = Query(None)):
    """
    질병 목록 조회 (페이징, 이름 검색 등)
    """
    return {
        "page": page,
        "size": size,
        "search": search,
        "diseases": [{"id": "D00123", "name": "Disease A"}],
    }


@router.get("/model-versions")
async def get_model_versions():
    """
    사용 가능한 모델 버전 목록 조회
    """
    return {"model_versions": ["v1.0", "v1.1", "v2.0"]}


@router.get("/model-versions/{version_id}")
async def get_model_version_metadata(version_id: str):
    """
    특정 모델 버전 메타데이터 조회
    """
    return {"version_id": version_id, "metadata": {"description": "Model version details"}}


@router.get("/predictions")
async def get_predictions(
    protein_id: str = Query(None),
    disease_id: str = Query(None),
    model_version: str = Query(None),
    top_k_proteins: int = Query(5),
    top_k_diseases: int = Query(5),
):
    """
    범용 예측 조회 엔드포인트 (단백질·질병 둘 다 파라미터로)
    """
    return {
        "protein_id": protein_id,
        "disease_id": disease_id,
        "model_version": model_version,
        "top_k_proteins": top_k_proteins,
        "top_k_diseases": top_k_diseases,
        "predictions": {"proteins": [], "diseases": []},
    }


@router.post("/predictions/batch")
async def batch_predictions(batch_request: dict):
    """
    배치 예측 요청 (여러 단백질·질병 한 번에 처리)
    """
    return {"batch_request": batch_request, "predictions": []}


@router.get("/health")
async def health_check():
    """
    서비스 상태(헬스체크)
    """
    return {"status": "healthy"}