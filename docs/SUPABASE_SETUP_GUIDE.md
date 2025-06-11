# 🗄️ Supabase 설정 가이드

## 📋 개요

**AI 창업 지원사업 모니터링 시스템**을 위한 완전한 Supabase 설정 가이드입니다.

---

## 🚀 1단계: Supabase 프로젝트 생성

### 1.1 프로젝트 생성
1. [Supabase](https://supabase.com)에 접속
2. "New Project" 클릭
3. 프로젝트 정보 입력:
   - **Name**: `startup-monitor-system`
   - **Database Password**: 강력한 패스워드 설정
   - **Region**: `Northeast Asia (Seoul)` 선택
4. "Create new project" 클릭

### 1.2 환경변수 설정
프로젝트 생성 후 다음 정보를 메모:
```bash
# core/config.py에 추가할 정보
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_KEY = "your-anon-key"
```

---

## 🏗️ 2단계: 데이터베이스 스키마 설정

### 2.1 SQL 스키마 실행
Supabase Dashboard → SQL Editor에서 `supabase_schema.sql` 내용을 실행:

```sql
-- 1. 지원사업 프로그램 테이블
CREATE TABLE IF NOT EXISTS support_programs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    external_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    organization TEXT,
    category TEXT,
    support_type TEXT,
    target_audience TEXT,
    support_amount TEXT,
    application_deadline DATE,
    program_period TEXT,
    contact_info TEXT,
    url TEXT,
    ai_score DECIMAL DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. 사용자 피드백 테이블  
CREATE TABLE IF NOT EXISTS user_feedback (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    program_external_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('delete', 'interest', 'view')),
    reason TEXT,
    confidence DECIMAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (program_external_id) REFERENCES support_programs(external_id)
);

-- 3. AI 학습 패턴 테이블
CREATE TABLE IF NOT EXISTS learning_patterns (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    pattern_type TEXT NOT NULL CHECK (pattern_type IN ('keyword', 'site', 'score_range')),
    category TEXT NOT NULL,
    pattern_key TEXT NOT NULL,
    reason TEXT,
    frequency INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(pattern_type, category, pattern_key, reason)
);

-- 4. 시스템 로그 테이블 (구조화된 로그)
CREATE TABLE IF NOT EXISTS system_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    log_level TEXT NOT NULL CHECK (log_level IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    category TEXT NOT NULL CHECK (category IN ('SYSTEM', 'CRAWLING', 'AI_LEARNING', 'USER_ACTION', 'ERROR')),
    message TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 5. 프로그램 통계 테이블
CREATE TABLE IF NOT EXISTS program_stats (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    total_programs INTEGER DEFAULT 0,
    active_programs INTEGER DEFAULT 0,
    daily_new_programs INTEGER DEFAULT 0,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 6. AI 학습 통계 테이블
CREATE TABLE IF NOT EXISTS ai_learning_stats (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    total_feedback INTEGER DEFAULT 0,
    positive_feedback INTEGER DEFAULT 0,
    negative_feedback INTEGER DEFAULT 0,
    accuracy_percentage DECIMAL DEFAULT 0,
    last_training_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 7. 시스템 설정 테이블
CREATE TABLE IF NOT EXISTS system_settings (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    setting_key TEXT UNIQUE NOT NULL,
    setting_value JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 8. 크롤링 사이트 테이블 (CSV에서 이동)
CREATE TABLE IF NOT EXISTS crawling_sites (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    selector TEXT NOT NULL,
    region TEXT,
    priority INTEGER DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    enabled BOOLEAN DEFAULT true,
    category TEXT,
    description TEXT,
    
    -- 크롤링 통계
    total_crawls INTEGER DEFAULT 0,
    successful_crawls INTEGER DEFAULT 0,
    failed_crawls INTEGER DEFAULT 0,
    last_crawl_at TIMESTAMP WITH TIME ZONE,
    last_success_at TIMESTAMP WITH TIME ZONE,
    last_error TEXT,
    
    -- 메타데이터
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### 2.2 인덱스 생성
성능 최적화를 위한 인덱스:

```sql
-- 프로그램 검색 최적화
CREATE INDEX IF NOT EXISTS idx_support_programs_active ON support_programs(is_active);
CREATE INDEX IF NOT EXISTS idx_support_programs_score ON support_programs(ai_score DESC);
CREATE INDEX IF NOT EXISTS idx_support_programs_created ON support_programs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_support_programs_deadline ON support_programs(application_deadline);

-- 피드백 분석 최적화
CREATE INDEX IF NOT EXISTS idx_user_feedback_action ON user_feedback(action);
CREATE INDEX IF NOT EXISTS idx_user_feedback_created ON user_feedback(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_feedback_program ON user_feedback(program_external_id);

-- 학습 패턴 조회 최적화
CREATE INDEX IF NOT EXISTS idx_learning_patterns_type ON learning_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_learning_patterns_frequency ON learning_patterns(frequency DESC);

-- 시스템 로그 조회 최적화
CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(log_level);
CREATE INDEX IF NOT EXISTS idx_system_logs_category ON system_logs(category);
CREATE INDEX IF NOT EXISTS idx_system_logs_created ON system_logs(created_at DESC);

-- 텍스트 검색 최적화 (simple 사용 - 한국어 설정 미지원)
CREATE INDEX IF NOT EXISTS idx_support_programs_title_search ON support_programs USING GIN(to_tsvector('simple', title));
CREATE INDEX IF NOT EXISTS idx_support_programs_content_search ON support_programs USING GIN(to_tsvector('simple', content));

-- 크롤링 사이트 조회 최적화
CREATE INDEX IF NOT EXISTS idx_crawling_sites_enabled ON crawling_sites(enabled);
CREATE INDEX IF NOT EXISTS idx_crawling_sites_priority ON crawling_sites(priority DESC);
CREATE INDEX IF NOT EXISTS idx_crawling_sites_region ON crawling_sites(region);
CREATE INDEX IF NOT EXISTS idx_crawling_sites_category ON crawling_sites(category);
CREATE INDEX IF NOT EXISTS idx_crawling_sites_last_crawl ON crawling_sites(last_crawl_at DESC);
```

---

## 🔒 3단계: Row Level Security (RLS) 설정

### 3.1 RLS 활성화
```sql
-- 모든 테이블에 RLS 활성화
ALTER TABLE support_programs ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE learning_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE program_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_learning_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE crawling_sites ENABLE ROW LEVEL SECURITY;
```

### 3.2 읽기 정책 생성
```sql
-- 프로그램 읽기 (활성 프로그램만)
CREATE POLICY "support_programs_read" ON support_programs
FOR SELECT USING (is_active = true);

-- 피드백 읽기 (모든 피드백)
CREATE POLICY "user_feedback_read" ON user_feedback
FOR SELECT USING (true);

-- 학습 패턴 읽기 (모든 패턴)
CREATE POLICY "learning_patterns_read" ON learning_patterns
FOR SELECT USING (true);

-- 시스템 로그 읽기 (최근 30일)
CREATE POLICY "system_logs_read" ON system_logs
FOR SELECT USING (created_at > NOW() - INTERVAL '30 days');

-- 통계 읽기 (모든 통계)
CREATE POLICY "program_stats_read" ON program_stats
FOR SELECT USING (true);

CREATE POLICY "ai_learning_stats_read" ON ai_learning_stats
FOR SELECT USING (true);

-- 설정 읽기 (모든 설정)
CREATE POLICY "system_settings_read" ON system_settings
FOR SELECT USING (true);

-- 크롤링 사이트 읽기 (활성 사이트만)
CREATE POLICY "crawling_sites_read" ON crawling_sites
FOR SELECT USING (enabled = true);
```

### 3.3 쓰기 정책 생성
```sql
-- 프로그램 삽입/업데이트
CREATE POLICY "support_programs_write" ON support_programs
FOR ALL USING (true);

-- 피드백 삽입
CREATE POLICY "user_feedback_write" ON user_feedback
FOR INSERT WITH CHECK (true);

-- 학습 패턴 쓰기
CREATE POLICY "learning_patterns_write" ON learning_patterns
FOR ALL USING (true);

-- 시스템 로그 삽입
CREATE POLICY "system_logs_write" ON system_logs
FOR INSERT WITH CHECK (true);

-- 통계 업데이트
CREATE POLICY "program_stats_write" ON program_stats
FOR ALL USING (true);

CREATE POLICY "ai_learning_stats_write" ON ai_learning_stats
FOR ALL USING (true);

-- 설정 업데이트
CREATE POLICY "system_settings_write" ON system_settings
FOR ALL USING (true);

-- 크롤링 사이트 쓰기 (모든 작업 허용)
CREATE POLICY "crawling_sites_write" ON crawling_sites
FOR ALL USING (true);
```

---

## ⚡ 4단계: 실시간 구독 설정

### 4.1 Realtime 활성화
```sql
-- 실시간 구독이 필요한 테이블에 대해 Realtime 활성화
ALTER publication supabase_realtime ADD TABLE support_programs;
ALTER publication supabase_realtime ADD TABLE user_feedback;
ALTER publication supabase_realtime ADD TABLE system_logs;
ALTER publication supabase_realtime ADD TABLE program_stats;
ALTER publication supabase_realtime ADD TABLE ai_learning_stats;
```

### 4.2 구독 정책 설정
```sql
-- 프로그램 실시간 구독 정책
CREATE POLICY "support_programs_realtime" ON support_programs
FOR SELECT USING (is_active = true);

-- 로그 실시간 구독 정책 (ERROR 레벨만)
CREATE POLICY "system_logs_realtime" ON system_logs
FOR SELECT USING (log_level IN ('ERROR', 'CRITICAL'));

-- 통계 실시간 구독 정책
CREATE POLICY "stats_realtime" ON program_stats
FOR SELECT USING (true);

CREATE POLICY "ai_stats_realtime" ON ai_learning_stats
FOR SELECT USING (true);
```

---

## 🔧 5단계: 데이터베이스 함수 생성

### 5.1 프로그램 검색 함수
```sql
-- 텍스트 검색 함수 (simple 언어 설정 사용)
CREATE OR REPLACE FUNCTION search_programs(
    search_query text,
    search_limit integer DEFAULT 25
)
RETURNS TABLE (
    id uuid,
    external_id text,
    title text,
    content text,
    organization text,
    ai_score decimal,
    created_at timestamptz,
    rank real
)
LANGUAGE sql
STABLE
AS $$
    SELECT 
        p.id,
        p.external_id,
        p.title,
        p.content,
        p.organization,
        p.ai_score,
        p.created_at,
        ts_rank(
            to_tsvector('simple', p.title || ' ' || COALESCE(p.content, '') || ' ' || COALESCE(p.organization, '')),
            plainto_tsquery('simple', search_query)
        ) as rank
    FROM support_programs p
    WHERE 
        p.is_active = true
        AND (
            to_tsvector('simple', p.title || ' ' || COALESCE(p.content, '') || ' ' || COALESCE(p.organization, ''))
            @@ plainto_tsquery('simple', search_query)
        )
    ORDER BY rank DESC, p.ai_score DESC, p.created_at DESC
    LIMIT search_limit;
$$;
```

### 5.2 통계 업데이트 함수
```sql
-- 프로그램 통계 업데이트 함수
CREATE OR REPLACE FUNCTION update_program_stats()
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO program_stats (total_programs, active_programs, daily_new_programs, updated_at)
    VALUES (
        (SELECT COUNT(*) FROM support_programs),
        (SELECT COUNT(*) FROM support_programs WHERE is_active = true),
        (SELECT COUNT(*) FROM support_programs WHERE created_at::date = CURRENT_DATE),
        NOW()
    )
    ON CONFLICT (id) DO UPDATE SET
        total_programs = EXCLUDED.total_programs,
        active_programs = EXCLUDED.active_programs,
        daily_new_programs = EXCLUDED.daily_new_programs,
        updated_at = EXCLUDED.updated_at;
END;
$$;

-- AI 학습 통계 업데이트 함수
CREATE OR REPLACE FUNCTION update_ai_learning_stats()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    total_count integer;
    positive_count integer;
    negative_count integer;
BEGIN
    SELECT COUNT(*) INTO total_count FROM user_feedback;
    SELECT COUNT(*) INTO positive_count FROM user_feedback WHERE action = 'interest';
    SELECT COUNT(*) INTO negative_count FROM user_feedback WHERE action = 'delete';
    
    INSERT INTO ai_learning_stats (
        total_feedback, 
        positive_feedback, 
        negative_feedback, 
        accuracy_percentage,
        updated_at
    )
    VALUES (
        total_count,
        positive_count,
        negative_count,
        CASE WHEN total_count > 0 THEN (positive_count::decimal / total_count * 100) ELSE 0 END,
        NOW()
    )
    ON CONFLICT (id) DO UPDATE SET
        total_feedback = EXCLUDED.total_feedback,
        positive_feedback = EXCLUDED.positive_feedback,
        negative_feedback = EXCLUDED.negative_feedback,
        accuracy_percentage = EXCLUDED.accuracy_percentage,
        updated_at = EXCLUDED.updated_at;
END;
$$;

-- 크롤링 통계 업데이트 함수
CREATE OR REPLACE FUNCTION update_crawling_stats(
    site_id uuid,
    success boolean,
    error_message text DEFAULT NULL
)
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE crawling_sites
    SET 
        total_crawls = total_crawls + 1,
        successful_crawls = CASE WHEN success THEN successful_crawls + 1 ELSE successful_crawls END,
        failed_crawls = CASE WHEN NOT success THEN failed_crawls + 1 ELSE failed_crawls END,
        last_crawl_at = NOW(),
        last_success_at = CASE WHEN success THEN NOW() ELSE last_success_at END,
        last_error = CASE WHEN NOT success THEN error_message ELSE NULL END,
        updated_at = NOW()
    WHERE id = site_id;
END;
$$;
```

### 5.3 자동 업데이트 트리거
```sql
-- 프로그램 통계 자동 업데이트 트리거
CREATE OR REPLACE FUNCTION trigger_update_program_stats()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    PERFORM update_program_stats();
    RETURN NULL;
END;
$$;

CREATE TRIGGER update_program_stats_trigger
    AFTER INSERT OR UPDATE OR DELETE ON support_programs
    FOR EACH STATEMENT
    EXECUTE FUNCTION trigger_update_program_stats();

-- AI 학습 통계 자동 업데이트 트리거
CREATE OR REPLACE FUNCTION trigger_update_ai_stats()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    PERFORM update_ai_learning_stats();
    RETURN NULL;
END;
$$;

CREATE TRIGGER update_ai_stats_trigger
    AFTER INSERT OR UPDATE OR DELETE ON user_feedback
    FOR EACH STATEMENT
    EXECUTE FUNCTION trigger_update_ai_stats();
```

---

## 🧹 6단계: 데이터 정리 작업

### 6.1 로그 정리 함수
```sql
-- 오래된 로그 정리 함수 (90일 이상)
CREATE OR REPLACE FUNCTION cleanup_old_logs()
RETURNS integer
LANGUAGE plpgsql
AS $$
DECLARE
    deleted_count integer;
BEGIN
    DELETE FROM system_logs 
    WHERE created_at < NOW() - INTERVAL '90 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- 정리 작업 로그 기록
    INSERT INTO system_logs (log_level, category, message, details)
    VALUES (
        'INFO', 
        'SYSTEM', 
        '오래된 로그 정리 완료',
        jsonb_build_object('deleted_count', deleted_count)
    );
    
    RETURN deleted_count;
END;
$$;
```

---

## 🔑 7단계: API 키 및 보안 설정

### 7.1 환경변수 설정
Python 프로젝트의 `core/config.py`에 추가:

```python
import os
from pathlib import Path

class Config:
    # Supabase 설정
    SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://your-project-id.supabase.co')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY', 'your-anon-key')
    
    # 보안 설정
    SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY', 'your-service-key')  # 관리자용
```

### 7.2 .env 파일 생성
프로젝트 루트에 `.env` 파일 생성:

```bash
# Supabase 설정
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-key

# 보안 키
SECRET_KEY=your-flask-secret-key
```

---

## 📊 8단계: 초기 데이터 설정

### 8.1 기본 설정 데이터
```sql
-- 시스템 기본 설정 삽입
INSERT INTO system_settings (setting_key, setting_value, description) VALUES
('max_concurrent_requests', '10', '최대 동시 크롤링 요청 수'),
('crawl_delay_seconds', '1', '사이트간 크롤링 지연 시간'),
('ai_score_threshold', '0.5', 'AI 점수 임계값'),
('log_retention_days', '90', '로그 보관 기간 (일)'),
('auto_cleanup_enabled', 'true', '자동 정리 기능 활성화');

-- 초기 통계 데이터
INSERT INTO program_stats (total_programs, active_programs, daily_new_programs)
VALUES (0, 0, 0);

INSERT INTO ai_learning_stats (total_feedback, positive_feedback, negative_feedback, accuracy_percentage)
VALUES (0, 0, 0, 0);
```

---

## ✅ 9단계: 설정 확인

### 9.1 연결 테스트
Python에서 연결 확인:

```python
from supabase import create_client, Client

# 연결 테스트
def test_supabase_connection():
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # 간단한 쿼리 테스트
        result = supabase.table('system_settings').select('*').limit(1).execute()
        
        if result.data:
            print("✅ Supabase 연결 성공!")
            return True
        else:
            print("❌ 데이터 조회 실패")
            return False
            
    except Exception as e:
        print(f"❌ Supabase 연결 실패: {e}")
        return False

# 테스트 실행
test_supabase_connection()
```

---

## 🎯 완료 체크리스트

### ✅ 필수 설정
- [ ] Supabase 프로젝트 생성
- [ ] 데이터베이스 스키마 생성 (7개 테이블)
- [ ] 인덱스 생성 (성능 최적화)
- [ ] RLS 정책 설정 (보안)
- [ ] 실시간 구독 활성화
- [ ] 데이터베이스 함수 생성
- [ ] 환경변수 설정
- [ ] 연결 테스트 완료

### ✅ 선택 설정
- [ ] 자동 정리 스케줄링 (Pro 플랜)
- [ ] 테스트 데이터 삽입
- [ ] 모니터링 대시보드 설정
- [ ] 백업 정책 설정

---

## 🚨 중요 보안 주의사항

1. **API 키 보안**
   - `.env` 파일을 반드시 `.gitignore`에 추가
   - Service Key는 서버 환경에서만 사용
   - 정기적으로 키 로테이션

2. **RLS 정책**
   - 모든 테이블에 적절한 RLS 정책 설정
   - 최소 권한 원칙 적용
   - 정기적으로 정책 검토

3. **데이터 백업**
   - 정기적인 데이터베이스 백업
   - 중요 데이터는 별도 저장소에 보관

---

## 🎯 다음 단계

✅ **모든 설정이 완료되었습니다!**

1. **환경변수 설정**: `env_setup_guide.md` 참고하여 `.env` 파일 생성
2. **CSV 마이그레이션**: `python migrate_csv_to_supabase.py` 실행 (486개 사이트)
3. **웹 대시보드 실행**: `python main.py`
4. **브라우저 접속**: http://localhost:5001
5. **크롤링 시작**: "크롤링 시작" 버튼 클릭
6. **AI 분석 확인**: 실시간으로 데이터가 Supabase에 저장됨

## 📊 Supabase Dashboard에서 확인할 수 있는 것들

- **support_programs**: 크롤링된 지원사업 정보
- **user_feedback**: 사용자 피드백 (삭제/관심/조회)
- **learning_patterns**: AI 학습 패턴
- **system_logs**: 시스템 로그 (90일 보관)
- **program_stats & ai_learning_stats**: 실시간 통계
- **crawling_sites**: 486개 크롤링 사이트 정보 (CSV에서 이동됨)

**🎉 Supabase 설정 완료! 이제 AI 모니터링 시스템이 완전히 작동할 준비가 되었습니다.** 