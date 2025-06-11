# 🚀 AI 기반 창업 지원사업 모니터링 시스템

## 📋 프로젝트 개요

**Supabase 기반 실시간 AI 창업 지원사업 모니터링 및 개인화 추천 시스템**

- 🌐 **486개 사이트** 실시간 크롤링으로 지원사업 정보 수집
- 🤖 **사용자 맞춤 Colab 모델** 활용한 고도화된 AI 분석 
- 🧠 **강화학습 기반** 개인화 추천 및 실시간 학습
- 💾 **Supabase 기반** 실시간 데이터베이스 및 웹 대시보드
- ⚡ **고속 병렬 처리**로 전체 사이트 동시 크롤링
- 📊 **구조화된 로그 시스템**으로 오류 추적 및 성능 분석

---

## 🏗️ **리팩토링 완료된 최종 아키텍처**

### 🎯 **핵심 4개 파일 구조**

```
startup_monitor_system/
├── 🤖 core/                          # 핵심 모듈
│   ├── config.py                     # 🔧 통합 설정 + 로그 정책
│   ├── database.py                   # 💾 Supabase DB 전체 관리  
│   ├── ai_engine.py                  # 🧠 AI 크롤링+분석+학습 통합
│   └── __init__.py                   # 📦 모듈 초기화
├── 🌐 main.py                        # 🖥️ Flask 웹 대시보드 서버
├── 📁 templates/                     # 🎨 HTML 템플릿
├── 📁 static/                        # 💄 CSS/JS 정적 파일
├── 📊 data/core/                     # 📄 486개 사이트 CSV (필수)
│   └── complete_startup_sites.csv
├── 🤖 models/                        # 🧠 AI 모델 저장소
│   ├── colab_sentence_model/         # 사용자 맞춤 Colab 모델
│   ├── sentence_transformers/        # 문장 임베딩 모델
│   └── transformers/                 # BERT 모델
├── 🗂️ backup_old_files/             # 기존 복잡한 파일들 백업
├── 📋 requirements.txt               # Python 의존성
├── 🏗️ project_structure.md          # 🔥 프로젝트 문서 (본 파일)
├── 🗄️ supabase_schema.sql           # DB 스키마
└── 📖 SUPABASE_SETUP_GUIDE.md       # Supabase 설정 가이드
```

### 🔥 **리팩토링 성과**
- **복잡도 감소**: 10개+ 파일 → **4개 핵심 파일**
- **기능 100% 보존**: 486사이트 크롤링, AI 분석, 강화학습, 웹 대시보드
- **성능 개선**: 모든 모델 서버 시작시 로드, 지연 없는 크롤링
- **사용자 모델 우선**: Colab 모델 80% 가중치로 높은 정확도

---

## 🧠 **AI 엔진 아키텍처**

### 🤖 **모델 구성**
```
AI Engine (core/ai_engine.py)
├── 🥇 Colab 모델 (우선 80% 가중치)
│   ├── 📝 문장 임베딩 모델 (사용자 맞춤)
│   ├── 🌳 RandomForest 모델  
│   └── 📈 GradientBoosting 모델
├── 🥈 기본 모델 (보완 20% 가중치)
│   ├── 🤖 BERT 모델 (klue/bert-base)
│   └── 🌍 다국어 문장 변환기
└── 🧠 강화학습 시스템
    ├── 사용자 피드백 실시간 학습
    ├── 삭제/유지/조회 패턴 분석
    └── 3개 피드백마다 자동 재훈련
```

### ⚡ **크롤링 시스템**
- **순차 모드**: 안정적인 단일 사이트 크롤링
- **고속 모드**: 비동기 병렬 처리 (최대 10개 동시)
- **상세 오류 기록**: URL 문제, 타임아웃, HTTP 오류 등 모두 Supabase에 기록

---

## 📊 **구조화된 로그 시스템**

### 🎯 **로그 분류 체계**

#### 💾 **Supabase DB 로그** (90일 보관)
- **SYSTEM**: 시스템 시작/종료, 설정 변경
- **CRAWLING**: 크롤링 시작/완료, 사이트별 결과
- **AI_LEARNING**: 모델 재훈련, 정확도 변화  
- **USER_ACTION**: 사용자 피드백, 삭제/관심
- **ERROR**: 모든 오류 (URL 문제, 타임아웃 등)

#### 🖥️ **콘솔 로그** (즉시 소멸)
- 실시간 디버깅 정보
- 서버 상태 모니터링

### 🔍 **URL 오류 추적 시스템**
```json
{
  "site_name": "사이트명",
  "site_url": "문제 URL", 
  "error_type": "TIMEOUT|HTTP_ERROR|CONNECTION_ERROR|...",
  "error_message": "상세 오류 메시지",
  "suggested_action": "해결 제안사항",
  "status_code": "HTTP 상태코드 (해당시)"
}
```

---

## 🗄️ **Supabase 데이터베이스 구조**

### 📊 **핵심 테이블**
- `support_programs` - 크롤링된 지원사업 데이터
- `user_feedback` - 사용자 피드백 (삭제/관심 등)
- `learning_patterns` - AI 학습 패턴 데이터
- `system_logs` - 구조화된 시스템 로그
- `program_stats` - 프로그램 통계
- `ai_learning_stats` - AI 학습 통계  
- `system_settings` - 시스템 설정
- `crawling_sites` - 486개 크롤링 사이트 정보 (CSV에서 이동)

### 🔒 **보안 정책**
- Row Level Security (RLS) 활성화
- 인증된 사용자만 데이터 접근
- 실시간 구독 정책 설정

---

## 🚀 **실행 방법**

### 1️⃣ **의존성 설치**
```bash
pip install -r requirements.txt
```

### 2️⃣ **환경변수 설정**
```bash
# env_setup_guide.md 참고하여 .env 파일 생성
# Supabase URL, API 키 등 설정
```

### 3️⃣ **Supabase 설정**
```bash
# Supabase 프로젝트 생성 후
# SUPABASE_SETUP_GUIDE.md 참조하여 설정
```

### 4️⃣ **CSV → Supabase 마이그레이션**
```bash
# 486개 크롤링 사이트를 Supabase로 이동
python migrate_csv_to_supabase.py
```

### 5️⃣ **서버 시작**
```bash
python main.py
```

### 6️⃣ **웹 대시보드 접속**
```
http://localhost:5001
```

---

## 🎯 **주요 기능**

### 🌐 **웹 대시보드**
- 실시간 크롤링 모니터링
- AI 분석 결과 시각화
- 사용자 피드백 인터페이스
- 시스템 상태 및 성능 지표

### 🤖 **AI 분석**
- 사용자 맞춤 Colab 모델 우선 사용
- 실시간 강화학습으로 정확도 지속 개선
- 개인화된 추천 및 필터링

### 📊 **모니터링**
- 486개 사이트 실시간 상태 추적
- URL 오류 상세 분석 및 해결 제안
- 크롤링 성능 및 성공률 통계

---

## 🔧 **기술 스택**

### 🎨 **Frontend**
- HTML5, CSS3, JavaScript
- Bootstrap (반응형 디자인)
- Chart.js (데이터 시각화)

### ⚙️ **Backend** 
- Python 3.8+
- Flask (웹 프레임워크)
- asyncio/aiohttp (비동기 처리)
- BeautifulSoup4 (웹 스크래핑)

### 🤖 **AI/ML**
- HuggingFace Transformers (BERT)
- sentence-transformers (문장 임베딩)
- scikit-learn (머신러닝)
- 사용자 맞춤 Colab 모델

### 💾 **Database**
- Supabase (실시간 PostgreSQL)
- 구조화된 로그 시스템
- Row Level Security (RLS)

---

## 📈 **성능 최적화**

### ⚡ **속도 개선**
- 서버 시작시 모든 모델 로드
- 비동기 병렬 크롤링 (고속모드)
- 지연 로딩 제거로 즉시 응답

### 🔍 **정확도 향상**  
- Colab 모델 80% 가중치
- 실시간 강화학습
- 사용자 피드백 기반 지속 개선

### 📊 **모니터링 강화**
- 상세 오류 추적 시스템
- URL 문제 자동 감지
- 성능 지표 실시간 수집

---

## 🛠️ **개발자 가이드**

### 📁 **코드 구조**
```python
# 1. 설정 (core/config.py)
Config.SUPABASE_LOG_CATEGORIES  # 로그 분류
Config.COLAB_MODEL_PATH         # 사용자 모델 경로

# 2. 데이터베이스 (core/database.py)  
db = get_db()
db.log_system_event()           # 구조화된 로그
db.insert_program()             # 프로그램 저장

# 3. AI 엔진 (core/ai_engine.py)
ai = get_ai()
ai.crawl_and_analyze_sites()    # 크롤링 + AI 분석
ai.record_user_feedback()       # 강화학습

# 4. 웹 서버 (main.py)
flask app                       # 대시보드 서버
```

### 🔧 **확장 방법**
1. **새 사이트 추가**: Supabase `crawling_sites` 테이블에 직접 추가
2. **AI 모델 개선**: Colab에서 새 모델 훈련 후 경로 설정
3. **새 기능 추가**: 각 핵심 파일에 메서드 추가

---

## 📞 **지원 및 문의**

### 📖 **문서**
- `SUPABASE_SETUP_GUIDE.md` - Supabase 설정 가이드
- `supabase_schema.sql` - 데이터베이스 스키마
- 본 파일 - 전체 프로젝트 구조

### 🔧 **트러블슈팅**
- Supabase 연결 오류 → 환경변수 확인
- 모델 로드 실패 → Colab 모델 경로 확인  
- 크롤링 오류 → system_logs 테이블에서 상세 분석

**🎉 리팩토링 완료! 복잡한 시스템을 단순하고 강력한 4개 파일로 통합했습니다.**