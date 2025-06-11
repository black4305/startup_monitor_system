# 🤖 AI 지원사업 모니터링 시스템 - 상세 기술 문서

## 📋 프로젝트 개요

**AI 기반 창업 지원사업 모니터링 및 분석 시스템**은 한국의 다양한 정부기관과 지방자치단체에서 발표하는 창업 지원사업 공고를 실시간으로 수집, 분석, 분류하여 사용자에게 맞춤형 추천을 제공하는 지능형 시스템입니다.

### 핵심 가치 제안
- 🎯 **개인화된 추천**: 사용자 프로필 기반 맞춤형 지원사업 추천
- 🔍 **실시간 모니터링**: 24/7 자동 크롤링 및 업데이트
- 🤖 **AI 기반 분석**: 다중 AI 모델을 통한 정확한 관련성 분석
- 📊 **강화학습 최적화**: 사용자 피드백을 통한 지속적 성능 개선

## 🏗️ 시스템 아키텍처

### 1. 전체 시스템 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    웹 인터페이스 계층                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   대시보드   │  │ 프로그램 목록│  │  관리 패널   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                       라우팅 계층                           │
│         Flask Blueprint 기반 RESTful API                    │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                      서비스 계층                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ProgramService│  │DashboardSvc │  │  AIService  │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                       AI 엔진 계층                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   AI 모델   │  │   크롤러    │  │피드백핸들러  │          │
│  │   매니저    │  │   시스템    │  │(강화학습)   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                      데이터베이스 계층                       │
│              Supabase PostgreSQL + REST API                │
└─────────────────────────────────────────────────────────────┘
```

### 2. 핵심 모듈 구조

#### **core/** 디렉토리 구성
- `ai_engine.py` (723줄): 통합 AI 엔진 메인 컨트롤러
- `database.py` (1034줄): Supabase 데이터베이스 매니저
- `crawler.py` (832줄): 웹 크롤링 시스템
- `ai_models.py` (326줄): AI 모델 관리자
- `deep_learning_engine.py` (480줄): 딥러닝 모델
- `reinforcement_learning_optimizer.py` (1210줄): 강화학습 최적화
- `feedback_handler.py` (462줄): 사용자 피드백 처리
- `routes.py` (520줄): 웹 라우팅
- `services.py` (217줄): 비즈니스 로직 계층
- `config.py` (175줄): 통합 설정 관리

## 📁 프로젝트 구조

```
startup_monitor_system/
├── core/                     # 핵심 모듈
│   ├── __init__.py
│   ├── config.py            # 설정 관리
│   ├── database.py          # 데이터베이스 연결
│   ├── ai_engine.py         # AI 엔진
│   ├── ai_models.py         # AI 모델 관리자
│   ├── crawler.py           # 웹 크롤링 시스템
│   ├── deep_learning_engine.py
│   ├── reinforcement_learning_optimizer.py
│   ├── feedback_handler.py  # 피드백 처리
│   ├── services.py          # 비즈니스 로직
│   ├── routes.py            # 웹 라우팅
│   └── app.py               # 메인 애플리케이션
├── models/                   # AI 모델 저장소
│   ├── apple_silicon_production_model.pkl (505MB)
│   └── apple_silicon_production_model_metadata.json
├── templates/                # HTML 템플릿
├── static/                   # 정적 파일
├── notebooks/                # Jupyter 노트북
├── data/                     # 데이터 파일
├── docs/                     # 문서
├── run.py                    # 실행 스크립트
├── requirements.txt          # 의존성 패키지
└── README.md                 # 본 문서
```

## 🤖 AI 모델 아키텍처

### 1. 다중 AI 모델 시스템

시스템은 **4가지 주요 AI 모델**을 통합하여 사용합니다:

#### **A. BERT 기반 언어 모델**
- **모델**: `klue/bert-base` (한국어 특화)
- **용도**: 텍스트 임베딩 및 의미 분석
- **가중치**: 60% (BERT_WEIGHT = 0.6)

#### **B. Sentence Transformer**
- **모델**: `paraphrase-multilingual-MiniLM-L12-v2`
- **용도**: 문장 유사도 계산 및 관련성 분석
- **기능**: 지원사업 기준 문장과의 유사도 측정

#### **C. Apple Silicon 최적화 딥러닝 모델**
- **파일**: `apple_silicon_production_model.pkl` (505MB)
- **메타데이터**: 정확도 100%, 훈련 샘플 803개, 테스트 샘플 201개
- **호환성**: CPU, MPS(Apple Silicon), CUDA
- **가중치**: 80% (COLAB_MODEL_WEIGHT = 0.8)

##### **pkl 파일 구성 (apple_silicon_production_model.pkl)**

이 pkl 파일은 **Google Colab에서 훈련된 고성능 딥러닝 모델**을 포함하며, Apple Silicon Mac에서 최적화된 실행을 위해 설계되었습니다.

**파일 내부 구조**:
```python
model_data = {
    # 🧠 핵심 신경망 모델
    'model_state_dict': # PyTorch 모델 가중치 (8층 딥러닝 네트워크)
    'model_structure': # DeepSupportClassifier 인스턴스
    
    # 🔧 특성 추출 시스템
    'feature_extractor': # AppleSiliconFeatureExtractor 인스턴스
    
    # 📊 모델 상태 정보
    'is_fitted': True,
    'device': 'cpu',  # Apple Silicon 호환성을 위한 CPU 저장
    'model_type': 'AppleSiliconDeepLearningModel',
    'version': '1.0',
    'created_at': '2025-06-08T07:36:10.901706',
    
    # 🎯 성능 지표
    'training_accuracy': 1.0,
    'training_samples': 803,
    'test_samples': 201
}
```

**신경망 아키텍처 (DeepSupportClassifier)**:
```
입력층 → 1024 → 512 → 256 → 128 → 64 → 32 → 출력층(2)
       ↓     ↓     ↓     ↓    ↓    ↓
    BatchNorm + ReLU + Dropout (각 층마다)
```

**특성 추출 시스템 (AppleSiliconFeatureExtractor)**:
- **TF-IDF 벡터화**: 5000차원, 1-3 n-gram
- **Sentence Transformer**: 384차원 임베딩 (다국어 MiniLM)
- **수동 특성**: 11개 도메인 특화 특성
  - 텍스트 길이, 단어 수, 키워드 개수
  - 지원사업 관련 키워드 포함 여부
  - 한글 비율, 숫자 개수 등

**학습 하이퍼파라미터**:
- **최적화기**: AdamW (lr=0.001, weight_decay=1e-5)
- **배치 크기**: 32
- **에포크**: 100회 (조기 종료 포함)
- **정규화**: Dropout 0.3, BatchNorm, Gradient Clipping
- **스케줄링**: ReduceLROnPlateau (patience=5)

**Apple Silicon 최적화 특징**:
- **pickle protocol 4**: 호환성 보장
- **CPU 기반 저장**: MPS 가속 지원하면서도 범용 호환성 유지
- **메모리 효율성**: 배치 처리 및 gradient clipping
- **다중 플랫폼 지원**: CPU/MPS/CUDA 자동 전환

#### **D. 강화학습 최적화 모델**
- **라이브러리**: Stable-Baselines3 + Gymnasium
- **알고리즘**: PPO (Proximal Policy Optimization)
- **용도**: 사용자 피드백 기반 추천 성능 최적화

### 2. AI 점수 계산 시스템

```python
# 최종 AI 점수 = 가중 평균
final_score = (
    bert_score * BERT_WEIGHT +
    sentence_score * (1 - BERT_WEIGHT) +
    deep_learning_score * COLAB_MODEL_WEIGHT +
    personalized_bonus +
    content_quality_bonus +
    relevance_bonus
)
```

#### **점수 구성 요소**:
1. **기본 AI 점수**: BERT + Sentence Transformer 조합
2. **딥러닝 점수**: Apple Silicon 최적화 모델
3. **개인화 점수**: 사용자 프로필 매칭
4. **콘텐츠 품질 보너스**: 텍스트 완성도 평가
5. **관련성 보너스**: 키워드 매칭도

### 3. 모델 성능 최적화

#### **디바이스 최적화**:
```python
# Apple Silicon (M1/M2)
DEVICE = torch.device('mps')
MAX_CPU_THREADS = 2
REQUEST_DELAY = 2.0

# NVIDIA GPU
DEVICE = torch.device('cuda')

# CPU 전용
DEVICE = torch.device('cpu')
MAX_CPU_THREADS = 4
REQUEST_DELAY = 3.0
```

## 🕷️ 웹 크롤링 시스템

### 1. 실시간 스트리밍 크롤링

#### **핵심 특징**:
- **비동기 처리**: `async/await` 기반 고성능 크롤링
- **실시간 분석**: 크롤링과 동시에 AI 분석 및 DB 저장
- **스트리밍 콜백**: 진행상황 실시간 모니터링

#### **크롤링 프로세스**:
```python
async def crawl_websites_streaming():
    1. 사이트 목록 조회 (우선순위/지역 필터링)
    2. 각 사이트별 비동기 크롤링
    3. 프로그램 발견 즉시 AI 분석
    4. 점수 임계값(55점) 이상만 DB 저장
    5. 실시간 통계 업데이트
```

#### **성능 설정**:
- **동시 요청 수**: GPU 사용시 10개, CPU 사용시 1개
- **요청 지연**: GPU 2초, CPU 3초
- **최대 프로그램/사이트**: 50개
- **점수 임계값**: 55점 (MIN_SCORE_THRESHOLD)

### 2. 사이트 관리 시스템

#### **크롤링 대상 사이트**:
- 중앙부처 (중기부, 과기부, 산업부 등)
- 지방자치단체 (서울, 부산, 광주 등)
- 공공기관 (KOTRA, KISA, 창진원 등)
- 대학 및 연구기관

#### **필터링 기능**:
- 지역별 필터링 (region)
- 우선순위별 처리 (priority)
- 사이트 활성화 상태 관리

## 💾 데이터베이스 아키텍처

### 1. Supabase PostgreSQL 구조

#### **주요 테이블**:

**A. support_programs** (지원사업 프로그램)
```sql
- external_id: VARCHAR (URL 기반 해시, Primary Key)
- title: VARCHAR(500)
- content: TEXT(2000)
- url: VARCHAR(1000)
- organization: VARCHAR(100)
- ai_score: FLOAT
- support_type: VARCHAR(100)
- application_deadline: DATE
- is_active: BOOLEAN
- created_at: TIMESTAMP
```

**B. user_feedback** (사용자 피드백)
```sql
- id: BIGSERIAL
- program_external_id: VARCHAR
- action: VARCHAR ('interested', 'not_interested', 'deleted')
- reason: TEXT
- confidence: FLOAT
- created_at: TIMESTAMP
```

**C. ai_learning_stats** (AI 학습 통계)
```sql
- id: BIGSERIAL
- accuracy_before: FLOAT
- accuracy_after: FLOAT
- feedback_count: INTEGER
- retrain_reason: TEXT
- created_at: TIMESTAMP
```

**D. crawling_sites** (크롤링 사이트 관리)
```sql
- site_id: VARCHAR
- name: VARCHAR
- url: VARCHAR
- priority: INTEGER
- region: VARCHAR
- is_enabled: BOOLEAN
- last_crawl_at: TIMESTAMP
```

### 2. 데이터 처리 파이프라인

#### **삽입/업데이트 프로세스**:
1. URL 기반 해시로 `external_id` 생성
2. 데이터 정제 및 길이 제한 적용
3. `upsert` 방식으로 중복 방지
4. 실시간 통계 업데이트

#### **성능 최적화**:
- **페이지네이션**: 50개씩 분할 조회
- **캐싱**: 5분간 메모리 캐시
- **인덱싱**: external_id, ai_score, created_at

## 🧠 강화학습 시스템

### 1. 학습 아키텍처

#### **환경 정의**:
```python
class RecommendationEnvironment:
    State: 사용자 프로필 + 프로그램 특성
    Action: 추천 여부 (0 또는 1)
    Reward: 사용자 피드백 기반 보상
    - 관심있음: +1.0
    - 관심없음: -0.5
    - 삭제: -1.0
```

#### **학습 알고리즘**:
- **PPO (Proximal Policy Optimization)**
- **학습 주기**: 피드백 3개마다 재훈련
- **신뢰도 임계값**: 0.7
- **관심도 임계값**: 0.3

### 2. 피드백 학습 프로세스

```python
def record_user_feedback():
    1. 사용자 액션 기록 (관심/무관심/삭제)
    2. 학습 패턴 분석 및 저장
    3. 피드백 개수 확인 (MIN_FEEDBACK_FOR_RETRAIN)
    4. 임계값 도달시 자동 재훈련
    5. 성능 지표 업데이트
```

## 🌐 웹 애플리케이션 구조

### 1. Flask 기반 웹 서버

#### **라우팅 구조**:
```python
# 메인 페이지
GET  /              → 대시보드
GET  /programs      → 프로그램 목록 (페이지네이션)
GET  /admin         → 관리자 페이지

# API 엔드포인트
GET  /api/programs               → 프로그램 데이터 조회
POST /api/feedback              → 사용자 피드백 등록
GET  /api/learning-status       → AI 학습 상태
POST /api/retrain              → 모델 재훈련
POST /api/start-crawling       → 크롤링 시작
```

#### **서비스 계층 분리**:
- **ProgramService**: 프로그램 CRUD 및 캐싱
- **DashboardService**: 대시보드 데이터 집계
- **AIService**: AI 모델 관리 및 학습

### 2. 사용자 인터페이스

#### **대시보드 기능**:
- 📊 전체 통계 (프로그램 수, 정확도 등)
- 🏆 최신 추천 프로그램 (AI 점수 순)
- 📈 실시간 크롤링 진행 상황
- 🤖 AI 학습 상태 모니터링

#### **프로그램 목록 기능**:
- 📄 페이지네이션 (50개씩)
- 🔍 검색 및 필터링
- ⭐ AI 점수 기반 정렬
- 🔗 원본 링크 유효성 검사

## 🚀 설치 및 실행

### 1. 가상환경 활성화
```bash
conda activate school_class
```

### 2. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 환경변수 설정
```bash
cp env.example .env
# .env 파일을 편집하여 Supabase 설정을 입력하세요
```

### 4. 애플리케이션 실행

#### 메인 실행 방법 (권장)
```bash
python run.py
```

#### 직접 실행
```bash
python core/app.py
```

## ⚙️ 시스템 설정 및 운영

### 1. 환경 설정

#### **필수 환경변수**:
```bash
# Supabase 데이터베이스
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-key
SUPABASE_SERVICE_KEY=your-service-key

# Flask 웹서버
FLASK_HOST=0.0.0.0
FLASK_PORT=5001
FLASK_DEBUG=false
SECRET_KEY=ai_support_monitor_2025

# 성능 튜닝
MAX_CONCURRENT_REQUESTS=10
CRAWL_DELAY_SECONDS=2.0
```

#### **기본 사용자 프로필**:
```python
DEFAULT_USER_PROFILE = {
    "business_type": "AI 가족 맞춤형 여행 큐레이션 서비스",
    "stage": "예비창업",
    "region": "광주",
    "keywords": ["AI", "인공지능", "여행", "창업", "지원사업"],
    "funding_needs": ["자금지원", "입주프로그램", "멘토링"]
}
```

### 2. 성능 모니터링

#### **로깅 시스템**:
- **콘솔 로그**: 개발/디버깅용 실시간 출력
- **Supabase 로그**: 중요 시스템 이벤트 영구 저장
- **카테고리**: SYSTEM, CRAWLING, AI_LEARNING, USER_ACTION, ERROR

#### **메트릭스**:
- 크롤링 성공률
- AI 모델 정확도
- 사용자 만족도
- 시스템 응답 시간

## 🌐 웹 인터페이스

서버 실행 후 브라우저에서 접속:
- **대시보드**: http://localhost:5001
- **프로그램 목록**: http://localhost:5001/programs
- **관리자 패널**: http://localhost:5001/admin

## 📊 주요 기능

- 🔍 AI 기반 지원사업 분석
- 📈 실시간 데이터 모니터링
- 🤖 딥러닝 모델을 통한 분류
- 📱 반응형 웹 대시보드
- 🍎 Apple Silicon 최적화

## 🔧 기술 스택

- **Backend**: Flask, Python 3.9+
- **Database**: Supabase (PostgreSQL)
- **AI/ML**: PyTorch, Transformers, Scikit-learn
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Apple Silicon 호환

## 🔧 의존성 및 호환성

### 1. 핵심 라이브러리

#### **AI/ML 스택**:
```
torch>=1.12.0 (딥러닝 프레임워크)
transformers>=4.20.0 (자연어 처리)
sentence-transformers>=2.2.0 (문장 임베딩)
scikit-learn>=1.1.0 (전통적 ML)
stable-baselines3>=1.6.0 (강화학습)
```

#### **웹/데이터베이스**:
```
Flask==3.0.0 (웹 프레임워크)
supabase>=2.15.0 (데이터베이스 클라이언트)
aiohttp>=3.8.0 (비동기 HTTP)
beautifulsoup4==4.12.2 (HTML 파싱)
```

### 2. 플랫폼 호환성

#### **지원 플랫폼**:
- ✅ **Apple Silicon (M1/M2)**: MPS 가속 지원
- ✅ **NVIDIA GPU**: CUDA 11.7+ 지원
- ✅ **Intel CPU**: 멀티스레딩 최적화
- ✅ **Google Colab**: 클라우드 환경 호환

#### **Python 요구사항**:
- Python 3.8+ (권장: 3.11)
- 메모리: 최소 4GB (권장: 8GB+)
- 저장공간: 2GB+ (모델 파일 포함)

## 📝 환경변수

```bash
# Flask 설정
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
PORT=5001

# Supabase 데이터베이스 설정
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-anon-key

# AI 모델 설정
AI_MODEL_PATH=models/

# 로깅 설정
LOG_LEVEL=INFO

# 캐시 설정
CACHE_DURATION=300
```

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

---

🎉 **리팩토링 완료!** 더 깔끔하고 유지보수하기 쉬운 코드베이스로 개선되었습니다. 

이 시스템은 **리팩토링된 모듈화 구조**로 설계되어 유지보수성과 확장성을 모두 갖춘 **프로덕션 레디** 상태의 AI 서비스입니다. 