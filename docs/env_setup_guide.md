# 🔧 환경변수 설정 가이드

## 📁 .env 파일 생성 방법

### 1️⃣ .env 파일 생성
```bash
# 프로젝트 루트에 .env 파일 생성
touch .env
```

### 2️⃣ .env 파일 내용
```bash
# 🗄️ Supabase 설정 (필수)
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-key-here
SUPABASE_SERVICE_KEY=your-service-role-key-here

# 🔐 보안 키
SECRET_KEY=ai-startup-monitor-secret-key-2025

# 🌐 Flask 설정
FLASK_ENV=development
FLASK_DEBUG=true
FLASK_HOST=0.0.0.0
FLASK_PORT=5001

# 🤖 AI 모델 설정
COLAB_MODEL_PATH=models/colab_sentence_model
MAX_CONCURRENT_REQUESTS=10
CRAWL_DELAY_SECONDS=1

# 📊 로그 설정
LOG_RETENTION_DAYS=90
AUTO_CLEANUP_ENABLED=true
```

## 🔑 Supabase 키 찾는 방법

### 1️⃣ Supabase Dashboard 접속
1. [https://supabase.com/dashboard](https://supabase.com/dashboard) 접속
2. 프로젝트 선택

### 2️⃣ API 키 복사
1. **Settings** → **API** 메뉴 접속
2. **Project URL** 복사 → `SUPABASE_URL`에 입력
3. **anon** **public** 키 복사 → `SUPABASE_KEY`에 입력  
4. **service_role** **secret** 키 복사 → `SUPABASE_SERVICE_KEY`에 입력

### 3️⃣ 설정 확인
```bash
# .env 파일이 제대로 읽히는지 확인
python -c "
from core.config import Config
print(f'✅ Supabase URL: {Config.SUPABASE_URL[:20]}...')
print(f'✅ Supabase Key: {Config.SUPABASE_KEY[:20]}...')
"
```

## 🚀 마이그레이션 실행

### 1️⃣ CSV → Supabase 마이그레이션
```bash
# CSV 파일을 Supabase로 이동
python migrate_csv_to_supabase.py
```

### 2️⃣ 서버 실행
```bash
# 웹 대시보드 실행
python main.py
```

## ✅ 환경변수 설정 체크리스트

- [ ] `.env` 파일 생성됨
- [ ] `SUPABASE_URL` 설정됨
- [ ] `SUPABASE_KEY` 설정됨  
- [ ] `SUPABASE_SERVICE_KEY` 설정됨
- [ ] Supabase 테이블 생성됨 (`supabase_schema.sql` 실행)
- [ ] CSV 마이그레이션 완료됨
- [ ] 서버 정상 실행됨

## 📝 주의사항

1. **보안**: `.env` 파일은 절대 Git에 커밋하지 마세요
2. **백업**: CSV 파일은 자동으로 백업됩니다  
3. **테스트**: 마이그레이션 후 Supabase Dashboard에서 데이터 확인
4. **오류**: 환경변수 관련 오류 시 위 설정을 다시 확인

## 🎯 다음 단계

1. SUPABASE_SETUP_GUIDE.md 참고하여 Supabase 설정
2. 마이그레이션 실행
3. 웹 대시보드 접속 (http://localhost:5001)
4. 크롤링 및 AI 분석 기능 테스트 