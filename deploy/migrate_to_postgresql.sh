#!/bin/bash
# PostgreSQL 마이그레이션 스크립트

# 색상 코드
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 PostgreSQL 마이그레이션 시작${NC}"

# 1. PostgreSQL 설치
echo -e "${YELLOW}1. PostgreSQL 설치 중...${NC}"
sudo apt update
sudo apt install -y postgresql postgresql-contrib

# 2. PostgreSQL 서비스 확인
echo -e "${YELLOW}2. PostgreSQL 서비스 확인...${NC}"
sudo systemctl status postgresql --no-pager

# 3. 데이터베이스 및 사용자 생성
echo -e "${YELLOW}3. 데이터베이스 설정 중...${NC}"

# 비밀번호 생성 (또는 직접 입력)
DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
echo -e "${GREEN}생성된 DB 비밀번호: ${DB_PASSWORD}${NC}"
echo "이 비밀번호를 .env.production에 저장하세요!"

# PostgreSQL 명령 실행
sudo -u postgres psql << EOF
-- 사용자 생성
CREATE USER aimonitor WITH PASSWORD '${DB_PASSWORD}';

-- 데이터베이스 생성
CREATE DATABASE ai_monitor OWNER aimonitor;

-- 권한 부여
GRANT ALL PRIVILEGES ON DATABASE ai_monitor TO aimonitor;

-- 확인
\l
\du
EOF

# 4. 테이블 생성
echo -e "${YELLOW}4. 테이블 생성 중...${NC}"
sudo -u postgres psql -d ai_monitor < /home/ubuntu/create_tables.sql

# 5. 데이터 임포트
echo -e "${YELLOW}5. 데이터 임포트 중...${NC}"
sudo -u postgres psql -d ai_monitor < /home/ubuntu/supabase_backup.sql

# 6. 권한 재설정
echo -e "${YELLOW}6. 권한 설정 중...${NC}"
sudo -u postgres psql -d ai_monitor << EOF
GRANT ALL ON ALL TABLES IN SCHEMA public TO aimonitor;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO aimonitor;
GRANT ALL ON SCHEMA public TO aimonitor;
EOF

# 7. 데이터 확인
echo -e "${YELLOW}7. 데이터 확인...${NC}"
sudo -u postgres psql -d ai_monitor << EOF
SELECT 'support_programs' as table_name, COUNT(*) as count FROM support_programs
UNION ALL
SELECT 'user_feedback', COUNT(*) FROM user_feedback
UNION ALL
SELECT 'ai_learning_stats', COUNT(*) FROM ai_learning_stats
UNION ALL
SELECT 'crawling_sites', COUNT(*) FROM crawling_sites;
EOF

# 8. PostgreSQL 설정 (외부 연결 허용 - 선택사항)
echo -e "${YELLOW}8. PostgreSQL 설정...${NC}"
# Docker 컨테이너에서 접속할 수 있도록 설정
sudo sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/" /etc/postgresql/*/main/postgresql.conf
echo "host    all             all             172.0.0.0/8            md5" | sudo tee -a /etc/postgresql/*/main/pg_hba.conf

# PostgreSQL 재시작
sudo systemctl restart postgresql

echo -e "${GREEN}✅ 마이그레이션 완료!${NC}"
echo -e "${YELLOW}다음 단계:${NC}"
echo "1. .env.production 파일 수정:"
echo "   USE_POSTGRESQL=true"
echo "   DATABASE_URL=postgresql://aimonitor:${DB_PASSWORD}@localhost:5432/ai_monitor"
echo "2. Docker 컨테이너 재시작"