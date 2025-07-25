-- AI 지원사업 모니터링 시스템 PostgreSQL 테이블 생성 스크립트

-- 1. 지원 프로그램 테이블
CREATE TABLE IF NOT EXISTS support_programs (
    external_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    content TEXT,
    url VARCHAR(1000) UNIQUE NOT NULL,
    organization VARCHAR(100),
    support_type VARCHAR(100),
    target_audience VARCHAR(200),
    application_deadline DATE,
    ai_score FLOAT DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    crawled_from VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. 사용자 피드백 테이블
CREATE TABLE IF NOT EXISTS user_feedback (
    id BIGSERIAL PRIMARY KEY,
    program_external_id VARCHAR(255) REFERENCES support_programs(external_id),
    action VARCHAR(50) NOT NULL CHECK (action IN ('interested', 'not_interested', 'deleted')),
    reason TEXT,
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. AI 학습 통계 테이블
CREATE TABLE IF NOT EXISTS ai_learning_stats (
    id BIGSERIAL PRIMARY KEY,
    accuracy_before FLOAT,
    accuracy_after FLOAT,
    feedback_count INTEGER,
    programs_analyzed INTEGER,
    retrain_reason TEXT,
    model_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. 크롤링 사이트 관리 테이블
CREATE TABLE IF NOT EXISTS crawling_sites (
    site_id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    url VARCHAR(500) NOT NULL,
    list_selector VARCHAR(500),
    title_selector VARCHAR(500),
    link_selector VARCHAR(500),
    priority INTEGER DEFAULT 5,
    region VARCHAR(50),
    category VARCHAR(100),
    is_enabled BOOLEAN DEFAULT true,
    last_crawl_at TIMESTAMP WITH TIME ZONE,
    crawl_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. 시스템 로그 테이블
CREATE TABLE IF NOT EXISTS system_logs (
    id BIGSERIAL PRIMARY KEY,
    category VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 6. 학습 패턴 테이블
CREATE TABLE IF NOT EXISTS learning_patterns (
    id BIGSERIAL PRIMARY KEY,
    pattern_type VARCHAR(50) NOT NULL,
    category VARCHAR(100),
    pattern_key VARCHAR(200) NOT NULL,
    value FLOAT DEFAULT 0,
    occurrences INTEGER DEFAULT 1,
    confidence FLOAT DEFAULT 0.5,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(pattern_type, category, pattern_key)
);

-- 7. 시스템 설정 테이블
CREATE TABLE IF NOT EXISTS system_settings (
    id BIGSERIAL PRIMARY KEY,
    setting_key VARCHAR(100) UNIQUE NOT NULL,
    setting_value TEXT,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스 생성
CREATE INDEX idx_programs_ai_score ON support_programs(ai_score DESC);
CREATE INDEX idx_programs_created_at ON support_programs(created_at DESC);
CREATE INDEX idx_programs_active ON support_programs(is_active);
CREATE INDEX idx_feedback_program_id ON user_feedback(program_external_id);
CREATE INDEX idx_feedback_created_at ON user_feedback(created_at DESC);
CREATE INDEX idx_sites_enabled ON crawling_sites(is_enabled);
CREATE INDEX idx_logs_category ON system_logs(category);
CREATE INDEX idx_logs_created_at ON system_logs(created_at DESC);
CREATE INDEX idx_patterns_lookup ON learning_patterns(pattern_type, category, pattern_key);
CREATE INDEX idx_settings_key ON system_settings(setting_key);

-- 트리거: updated_at 자동 업데이트
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_programs_updated_at BEFORE UPDATE ON support_programs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sites_updated_at BEFORE UPDATE ON crawling_sites
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_settings_updated_at BEFORE UPDATE ON system_settings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();