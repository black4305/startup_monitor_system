# 🚀 AI 지원사업 모니터링 시스템

한국 정부기관의 창업 지원사업을 실시간으로 크롤링하고 AI로 분석하여 맞춤형 추천을 제공하는 Flask 기반 웹 애플리케이션입니다.

## 📋 주요 기능

- 🔍 **실시간 크롤링**: 정부 지원사업 사이트 자동 모니터링
- 🤖 **AI 분석**: BERT + Sentence Transformer + 강화학습 모델로 관련성 평가
- 📊 **대시보드**: 지원사업 현황 및 통계 시각화
- 🎯 **자동 필터링**: 스팸/광고 자동 제거, 창업 관련 사업만 추출
- 💾 **캐싱**: Redis 기반 5분 캐시로 성능 최적화

## 🏗️ 프로젝트 구조

```
startup_monitor_system/
├── core/                      # 핵심 비즈니스 로직
│   ├── ai_engine.py          # AI 통합 엔진
│   ├── ai_models.py          # AI 모델 관리자
│   ├── app.py                # Flask 애플리케이션
│   ├── config.py             # 설정 관리
│   ├── crawler.py            # 웹 크롤러
│   ├── database.py           # Supabase 연결
│   ├── database_postgresql.py # PostgreSQL 연결 (선택)
│   ├── routes.py             # API 라우트
│   └── services.py           # 비즈니스 서비스
├── models/                    # AI 모델 파일
│   └── apple_silicon_production_model.pkl
├── static/                    # 정적 파일 (CSS, JS)
├── templates/                 # HTML 템플릿
├── deploy/                    # 배포 가이드
│   └── ncp-detailed-infrastructure-guide.md
├── docker-compose.yml         # 로컬 개발용
├── docker-compose.prod.yml    # 프로덕션용
├── requirements.txt           # Python 패키지
└── run.py                     # 진입점
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/your-username/startup_monitor_system.git
cd startup_monitor_system

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일을 편집하여 필요한 값 입력
```

### 2. 로컬 실행

```bash
# 개발 서버 실행
python run.py

# Docker로 실행
docker-compose up
```

브라우저에서 http://localhost:5001 접속

## 📦 환경변수 설명

### `.env` (로컬 개발용)
- 개발 환경에서 사용하는 설정
- Supabase 또는 로컬 PostgreSQL 연결 정보
- 디버그 모드 활성화

### `.env.production` (프로덕션용)
- 실제 서비스 배포 시 사용
- 보안 강화 설정
- 성능 최적화 설정

## 🐳 Docker 사용법

### 로컬 개발
```bash
docker-compose up        # 전체 서비스 실행
docker-compose down      # 서비스 중지
```

### 프로덕션 배포
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 🤖 AI 모델

1. **BERT** (`klue/bert-base`): 한국어 텍스트 분석
2. **Sentence Transformer**: 문장 유사도 계산
3. **강화학습 모델**: 사용자 피드백 기반 최적화

## 📚 상세 문서

- [NCP 배포 가이드](deploy/ncp-detailed-infrastructure-guide.md)
- [Claude Code 사용 가이드](CLAUDE.md)

## 🛠️ 기술 스택

- **Backend**: Flask, Python 3.10+
- **AI/ML**: PyTorch, Transformers, Stable-Baselines3
- **Database**: Supabase/PostgreSQL
- **Cache**: Redis
- **Container**: Docker, Docker Compose
- **Cloud**: Naver Cloud Platform

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.