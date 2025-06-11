# 🐍 Python 3.10 슬림 이미지 사용
FROM python:3.10-slim

# 📦 시스템 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 📁 작업 디렉토리 설정
WORKDIR /app

# 📋 requirements.txt 복사 및 의존성 설치
COPY requirements.txt .

# 🚀 의존성 설치 (캐시 최적화)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 📂 애플리케이션 코드 복사
COPY . .

# 🔧 환경 변수 설정
ENV FLASK_APP=run.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# 🌐 포트 노출
EXPOSE 5000

# 👤 non-root 사용자 생성 및 전환
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# 🏃 애플리케이션 실행
CMD ["python", "run.py"] 