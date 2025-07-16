# Dockerfile

# 베이스 이미지 선택 (예: Python 3.10)
FROM python:3.10

# 작업 디렉토리 생성
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 포트 오픈 (필요시)
EXPOSE 8000

# 실행 명령어 (FastAPI 예시)
CMD ["python", "main.py"]
