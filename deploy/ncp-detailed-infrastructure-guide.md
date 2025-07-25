# NCP 상세 인프라 설정 가이드 (단순화 구성 - NAT Gateway 제외)

## 🏗️ 전체 아키텍처 구성도 (비용 최적화)

```
┌─────────────────────────────────────────────────────────┐
│                     Internet                            │
└────────────────┬───────────────────────────────────────┘
                 │
           ┌─────┴─────┐
           │  공인 IP  │
           │101.x.x.x  │
           └─────┬─────┘
                 │
┌────────────────┴────────────────────────────────────────┐
│                    VPC (10.0.0.0/16)                    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │          Public Subnet (10.0.1.0/24)            │  │
│  │                                                 │  │
│  │  ┌────────────────────────────────────────┐    │  │
│  │  │     High Memory-m2 Server              │    │  │
│  │  │     (2vCPU, 16GB RAM)                  │    │  │
│  │  │                                        │    │  │
│  │  │  ┌─────────────┐  ┌─────────────┐    │    │  │
│  │  │  │ Flask App   │  │ PostgreSQL  │    │    │  │
│  │  │  │ (Docker)    │  │ (Docker)    │    │    │  │
│  │  │  └─────────────┘  └─────────────┘    │    │  │
│  │  │                                        │    │  │
│  │  │  ┌─────────────┐  ┌─────────────┐    │    │  │
│  │  │  │ Redis Cache │  │ AI Models   │    │    │  │
│  │  │  │ (Docker)    │  │ (3개 모델)  │    │    │  │
│  │  │  └─────────────┘  └─────────────┘    │    │  │
│  │  └────────────────────────────────────────┘    │  │
│  └─────────────────────────────────────────────────┘  │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │       Security Group (포트별 엄격한 제어)        │  │
│  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

💰 비용 절감: 단일 서버 + Public Subnet으로 NAT Gateway 비용(월 32,400원) 제거
```

## 🔐 STEP 7: SSL/TLS 설정 (무료)

### 7.1 Let's Encrypt 무료 SSL
```bash
# 서버 접속 후 실행
sudo apt update
sudo apt install -y certbot

# Nginx 설치 (리버스 프록시용)
sudo apt install -y nginx

# SSL 인증서 발급
sudo certbot certonly --standalone \
  -d your-domain.com \
  --agree-tos \
  --email your-email@example.com

# 자동 갱신 설정
sudo systemctl enable certbot.timer
```

### 7.2 Nginx 설정
```nginx
# /etc/nginx/sites-available/ai-monitor
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📊 STEP 8: 모니터링 설정

### 8.1 NCP Cloud Insight (기본 모니터링)
```
Management & Governance → Cloud Insight → [에이전트 설치]

# 서버에서 실행
curl -s http://repo.ncloud.com/cw-agent/scripts/install.sh | sudo bash
sudo /opt/NCP/cw_agent/cw_agent start

모니터링 항목 (자동):
- CPU 사용률
- 메모리 사용률
- 디스크 사용률
- 네트워크 I/O

비용: 무료 (기본 메트릭)
```


### 8.2 간단한 자체 모니터링
```python
# core/routes.py에 추가
@bp.route('/health')
def health_check():
    try:
        # DB 체크
        db.get_programs(limit=1)
        # 메모리 체크
        import psutil
        memory = psutil.virtual_memory()
        
        return jsonify({
            "status": "healthy",
            "memory_percent": memory.percent,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500
```

## 🔄 STEP 9: 백업 전략

### 9.1 서버 이미지 백업 (주 1회)
```
수동 백업:
Server → Server → ai-monitor-server 선택 → [서버 이미지 생성]

설정:
- 이미지 이름: ai-monitor-backup-$(date +%Y%m%d)
- 설명: 주간 백업

비용:
- 50GB 이미지: 시간당 2원
- 월 4개 보관: 약 5,760원/월

자동화 (선택):
# crontab -e
0 3 * * 0 /home/ubuntu/create_snapshot.sh
```

### 9.2 데이터 백업 스크립트
```bash
#!/bin/bash
# /home/ubuntu/backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/mnt/data/backup"

# 1. PostgreSQL 백업 (Docker 컨테이너)
docker exec postgres pg_dump -U dbuser startup_monitor | gzip > $BACKUP_DIR/db_$DATE.sql.gz

# 2. AI 모델 백업 (변경시에만)
if [ -n "$(find /app/models -mtime -1)" ]; then
    tar czf $BACKUP_DIR/models_$DATE.tar.gz /app/models/
fi

# 3. Object Storage 업로드 (AWS CLI 사용)
export AWS_ACCESS_KEY_ID=your_ncp_access_key
export AWS_SECRET_ACCESS_KEY=your_ncp_secret_key
export AWS_DEFAULT_REGION=kr-standard

aws --endpoint-url=https://kr.object.ncloudstorage.com \
    s3 cp $BACKUP_DIR/db_$DATE.sql.gz s3://ai-monitor-backup/

# 4. 로컬 백업 정리 (7일 이상)
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete

# 5. 백업 결과 로깅
echo "[$(date)] 백업 완료: db_$DATE.sql.gz" >> /var/log/backup.log
```

### 9.3 백업 자동화
```bash
# crontab 설정
crontab -e

# 매일 새벽 3시 백업
0 3 * * * /home/ubuntu/backup.sh

# 매주 일요일 서버 이미지 생성 알림
0 2 * * 0 echo "서버 이미지 백업 필요" | mail -s "Backup Reminder" your@email.com
```

## 💰 STEP 10: 시간제 자동화 설정

### 10.1 NCP CLI 설치 (로컬 컴퓨터)
```bash
# Mac
brew install ncloud-cli

# Linux
curl -O https://www.ncloud.com/api/support/download/cli/ncloud_cli_linux_amd64.tar.gz
tar -xvf ncloud_cli_linux_amd64.tar.gz
sudo mv ncloud /usr/local/bin/

# 인증 설정
ncloud configure
# Access Key ID: [NCP 콘솔에서 생성]
# Secret Access Key: [NCP 콘솔에서 생성]
# Region: KR
```

### 10.2 서버 시작/중지 스크립트
```bash
#!/bin/bash
# /home/local/ncp_scheduler.sh

SERVER_NAME="ai-monitor-server"

# 서버 인스턴스 번호 조회
get_server_id() {
    ncloud server getServerInstanceList \
        --serverName $SERVER_NAME \
        --query "serverInstanceList[0].serverInstanceNo" \
        --output text
}

# 서버 시작
start_server() {
    SERVER_ID=$(get_server_id)
    echo "서버 시작: $SERVER_ID"
    ncloud server startServerInstances --serverInstanceNoList $SERVER_ID
    
    # 시작 대기 (약 2분)
    sleep 120
    
    # 알림 (선택)
    curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
        -H 'Content-type: application/json' \
        -d '{"text":"🚀 AI Monitor 서버가 시작되었습니다!"}'
}

# 서버 중지
stop_server() {
    SERVER_ID=$(get_server_id)
    echo "서버 중지: $SERVER_ID"
    ncloud server stopServerInstances --serverInstanceNoList $SERVER_ID
    
    # 알림
    curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
        -H 'Content-type: application/json' \
        -d '{"text":"💤 AI Monitor 서버가 중지되었습니다!"}'
}

case "$1" in
    start) start_server ;;
    stop) stop_server ;;
    *) echo "Usage: $0 {start|stop}" ;;
esac
```

### 10.3 Crontab 설정 (로컬 컴퓨터)
```bash
# crontab -e

# 평일 오전 9시 서버 시작
0 9 * * 1-5 /home/local/ncp_scheduler.sh start

# 평일 오후 6시 서버 중지  
0 18 * * 1-5 /home/local/ncp_scheduler.sh stop

# 토요일 오후 2시 시작 (필요시)
0 14 * * 6 /home/local/ncp_scheduler.sh start

# 토요일 오후 6시 중지
0 18 * * 6 /home/local/ncp_scheduler.sh stop
```

## 📈 STEP 11: 성능 최적화

### 11.1 Docker Compose 최적화
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  postgres:
    image: postgres:13-alpine
    container_name: postgres
    environment:
      POSTGRES_DB: startup_monitor
      POSTGRES_USER: dbuser
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: always
    mem_limit: 2g
    cpus: '0.5'

  redis:
    image: redis:7-alpine
    container_name: redis
    command: >
      redis-server
      --maxmemory 1gb
      --maxmemory-policy allkeys-lru
      --save ""
    restart: always
    mem_limit: 1g
    cpus: '0.25'

  app:
    build: .
    container_name: flask_app
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://dbuser:${DB_PASSWORD}@postgres:5432/startup_monitor
      - REDIS_URL=redis://redis:6379
      - MAX_CPU_THREADS=2
      - MAX_CONCURRENT_REQUESTS=5
    volumes:
      - ./models:/app/models:ro
      - app_logs:/app/logs
    depends_on:
      - postgres
      - redis
    restart: always
    mem_limit: 10g
    cpus: '1.25'

volumes:
  postgres_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/data/postgres
  app_logs:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/data/logs

networks:
  default:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### 11.2 시스템 최적화
```bash
# /etc/sysctl.conf 추가
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
vm.overcommit_memory = 1

# 적용
sudo sysctl -p

# 스왑 설정
sudo swapoff -a
sudo swapon -a
```

## 🚨 STEP 12: 보안 강화

### 12.1 기본 보안 설정
```bash
# 1. SSH 포트 변경 (선택)
sudo nano /etc/ssh/sshd_config
# Port 22 → Port 2222
sudo systemctl restart sshd

# 2. 방화벽 설정
sudo ufw enable
sudo ufw allow 2222/tcp  # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 5000/tcp  # Flask (개발시)

# 3. fail2ban 설치 (무차별 공격 방어)
sudo apt install -y fail2ban
sudo systemctl enable fail2ban

# 4. 자동 보안 업데이트
sudo apt install -y unattended-upgrades
sudo dpkg-reconfigure --priority=low unattended-upgrades
```

### 12.2 환경변수 보안
```bash
# .env.production 파일 생성
cat > .env.production << EOF
# 보안 주의! 절대 Git에 커밋하지 마세요
DB_PASSWORD=$(openssl rand -base64 32)
SECRET_KEY=$(openssl rand -base64 48)
FLASK_SECRET_KEY=$(openssl rand -base64 48)

# NCP Object Storage
NCP_ACCESS_KEY=your_access_key
NCP_SECRET_KEY=your_secret_key

# 기타 설정
FLASK_ENV=production
MAX_CPU_THREADS=2
EOF

# 권한 설정
chmod 600 .env.production
```

## 📝 STEP 13: 서버 초기 설정 스크립트

### 13.1 전체 설정 자동화
```bash
#!/bin/bash
# /root/initial_setup.sh

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}🚀 AI Monitor 서버 초기 설정 시작${NC}"

# 1. 기본 패키지 업데이트
echo -e "${GREEN}📦 시스템 업데이트 중...${NC}"
apt update && apt upgrade -y

# 2. 필수 패키지 설치
echo -e "${GREEN}🔧 필수 패키지 설치 중...${NC}"
apt install -y \
    docker.io docker-compose \
    git htop vim curl wget \
    ufw fail2ban \
    postgresql-client

# 3. Docker 설정
usermod -aG docker ubuntu
systemctl enable docker

# 4. 스왑 메모리 설정
echo -e "${GREEN}💾 스왑 메모리 설정 중...${NC}"
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

# 5. 추가 스토리지 마운트
echo -e "${GREEN}💿 추가 스토리지 마운트 중...${NC}"
mkdir -p /mnt/data
mkfs.ext4 /dev/xvdb  # 추가 디스크
mount /dev/xvdb /mnt/data
echo '/dev/xvdb /mnt/data ext4 defaults 0 0' >> /etc/fstab

# 디렉토리 생성
mkdir -p /mnt/data/{postgres,logs,backup,models}

# 6. 프로젝트 클론
echo -e "${GREEN}📥 프로젝트 다운로드 중...${NC}"
cd /home/ubuntu
git clone https://github.com/your-username/startup_monitor_system.git
cd startup_monitor_system

# 7. 환경변수 설정
echo -e "${GREEN}🔐 환경변수 설정 중...${NC}"
cp .env.example .env.production
# 여기서 수동으로 .env.production 편집 필요

# 8. 모델 파일 다운로드
echo -e "${GREEN}🤖 AI 모델 다운로드 중...${NC}"
mkdir -p models
# Object Storage에서 다운로드 또는 로컬에서 복사

# 9. Docker 이미지 빌드
echo -e "${GREEN}🐳 Docker 이미지 빌드 중...${NC}"
docker-compose -f docker-compose.prod.yml build

# 10. 방화벽 설정
echo -e "${GREEN}🔥 방화벽 설정 중...${NC}"
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

echo -e "${GREEN}✅ 초기 설정 완료!${NC}"
echo -e "${GREEN}다음 명령으로 서비스를 시작하세요:${NC}"
echo "cd /home/ubuntu/startup_monitor_system"
echo "docker-compose -f docker-compose.prod.yml up -d"
```

## 🚀 STEP 14: 서버 접속 및 프로젝트 배포 상세 가이드

### 14.1 SSH 접속
```bash
# 로컬 터미널에서 실행
# 1. .pem 파일 권한 설정
chmod 400 ~/Downloads/ai-monitor-key.pem

# 2. SSH 접속
ssh -i ~/Downloads/ai-monitor-key.pem ubuntu@[공인IP]
# 예시: ssh -i ~/Downloads/ai-monitor-key.pem ubuntu@101.101.234.567
```

### 14.2 초기 서버 설정 실행
```bash
# 서버에 접속한 후, 위의 초기 설정 스크립트(13.1)를 실행
# 방법 1: 스크립트 파일 생성
nano initial_setup.sh
# 위 스크립트 내용 복사 후 저장 (Ctrl+X, Y, Enter)

# 실행 권한 부여
chmod +x initial_setup.sh

# 스크립트 실행
sudo ./initial_setup.sh
```

### 14.3 프로젝트 클론 및 설정
```bash
# 1. GitHub에서 프로젝트 클론
cd /home/ubuntu
git clone https://github.com/your-username/startup_monitor_system.git
cd startup_monitor_system

# 2. 환경변수 파일 생성
cp .env.production.example .env.production

# 편집기로 환경변수 수정
nano .env.production

# 또는 직접 생성
cat > .env.production << 'EOF'
# Flask 설정
FLASK_PORT=5001
SECRET_KEY=your-secret-key-here-change-this

# 데이터베이스 설정
USE_POSTGRESQL=true
DB_PASSWORD=your-strong-password-here
DATABASE_URL=postgresql://postgres:your-strong-password-here@postgres:5432/ai_monitor

# Supabase (선택사항, USE_POSTGRESQL=false일 때)
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-key

# Redis
REDIS_URL=redis://redis:6379/0

# AI 설정
MIN_SCORE_THRESHOLD=30
DEVICE=cpu
MAX_CPU_THREADS=4
MAX_CONCURRENT_REQUESTS=2

# 로깅
LOG_LEVEL=INFO
LOG_TO_FILE=true

# 서비스 URL
SERVICE_URL=https://startup.yourdomain.com

# CORS
CORS_ORIGINS=https://startup.yourdomain.com
EOF

# 파일 권한 설정
chmod 600 .env.production
```

### 14.4 필요 파일 생성

#### Dockerfile 생성
```bash
cat > Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 환경변수 설정
ENV FLASK_APP=run.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV FLASK_PORT=5001

# 포트 설정
EXPOSE 5001

# non-root 사용자 생성
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# 실행 명령
CMD ["python", "run.py"]
EOF
```

#### docker-compose.prod.yml 생성
```bash
cat > docker-compose.prod.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:13-alpine
    container_name: postgres
    environment:
      POSTGRES_DB: startup_monitor
      POSTGRES_USER: dbuser
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - app-network
    restart: always
    mem_limit: 2g
    cpus: '0.5'

  redis:
    image: redis:7-alpine
    container_name: redis
    command: >
      redis-server
      --maxmemory 1gb
      --maxmemory-policy allkeys-lru
      --save ""
    volumes:
      - redis_data:/data
    networks:
      - app-network
    restart: always
    mem_limit: 1g
    cpus: '0.25'

  app:
    build: .
    container_name: flask_app
    ports:
      - "5001:5001"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://dbuser:${DB_PASSWORD}@postgres:5432/startup_monitor
      - REDIS_URL=redis://redis:6379
      - MAX_CPU_THREADS=2
      - MAX_CONCURRENT_REQUESTS=5
    env_file:
      - .env.production
    volumes:
      - ./models:/app/models:ro
      - app_logs:/app/logs
    depends_on:
      - postgres
      - redis
    networks:
      - app-network
    restart: always
    mem_limit: 10g
    cpus: '1.25'

networks:
  app-network:
    driver: bridge

volumes:
  postgres_data:
  redis_data:
  app_logs:
EOF
```

#### nginx.prod.conf 생성
```bash
cat > nginx.prod.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # 로그 설정
    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;

    # 업로드 크기 제한
    client_max_body_size 100M;

    # Gzip 압축
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    upstream app {
        server app:5001;
    }

    # HTTP → HTTPS 리다이렉트
    server {
        listen 80;
        server_name startup.yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS 서버
    server {
        listen 443 ssl http2;
        server_name startup.yourdomain.com;

        # SSL 인증서
        ssl_certificate /etc/letsencrypt/live/startup.yourdomain.com/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/startup.yourdomain.com/privkey.pem;

        # SSL 설정
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;

        # 정적 파일
        location /static/ {
            alias /usr/share/nginx/html/static/;
            expires 30d;
            add_header Cache-Control "public, immutable";
        }

        # 프록시 설정
        location / {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # 타임아웃 설정
            proxy_connect_timeout 300s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }

        # 건강 체크 엔드포인트
        location /health {
            proxy_pass http://app/health;
            access_log off;
        }
    }
}
EOF
```

### 14.5 AI 모델 파일 준비
```bash
# 1. models 디렉토리 생성
mkdir -p models

# 2. 옵션 A: Object Storage에서 다운로드 (이미 업로드했다면)
aws configure  # NCP 인증키 입력
aws --endpoint-url=https://kr.object.ncloudstorage.com \
    s3 cp s3://ai-monitor-backup/models/apple_silicon_production_model.pkl \
    ./models/

# 3. 옵션 B: 로컬에서 업로드 (아직 업로드하지 않았다면)
# 로컬 터미널에서:
scp -i ~/Downloads/ai-monitor-key.pem \
    ~/Desktop/programming/startup_monitor_system/models/apple_silicon_production_model.pkl \
    ubuntu@[공인IP]:/home/ubuntu/startup_monitor_system/models/
```

### 14.6 Docker 컨테이너 실행
```bash
# 1. Docker 이미지 빌드
docker-compose -f docker-compose.prod.yml build

# 2. 컨테이너 실행 (백그라운드)
docker-compose -f docker-compose.prod.yml up -d

# 3. 실행 상태 확인
docker ps

# 4. 로그 확인
docker-compose -f docker-compose.prod.yml logs -f app

# 5. 헬스체크
curl http://localhost:5001/health
```

## 🧪 STEP 15: 서비스 테스트

### 15.1 로컬에서 포트 포워딩 테스트
```bash
# 새로운 로컬 터미널에서 SSH 터널링
ssh -i ~/Downloads/ai-monitor-key.pem -L 5001:localhost:5001 ubuntu@[공인IP]

# 브라우저에서 접속: http://localhost:5001
```

### 15.2 직접 접속 테스트
```bash
# Cloudflare 설정했다면
https://startup.yourdomain.com:5001

# 또는 공인 IP로 직접
http://[공인IP]:5001
```

### 15.3 API 엔드포인트 테스트
```bash
# 헬스체크
curl http://localhost:5001/health

# API 상태
curl http://localhost:5001/api/stats

# 프로그램 목록
curl http://localhost:5001/api/programs
```

## 🔧 STEP 16: 트러블슈팅

### 16.1 Docker 관련 문제
```bash
# 컨테이너 재시작
docker-compose -f docker-compose.prod.yml restart

# 컨테이너 완전 재시작
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d

# 개별 컨테이너 로그 확인
docker logs postgres
docker logs redis
docker logs flask_app
```

### 16.2 메모리 부족 문제
```bash
# 메모리 상태 확인
free -h

# Docker 리소스 정리
docker system prune -a

# 스왑 메모리 확인
swapon --show
```

### 16.3 포트 연결 문제
```bash
# 포트 사용 확인
sudo netstat -tlnp | grep 5001

# 방화벽 상태 확인
sudo ufw status

# NCP Security Group 확인
# NCP 콘솔에서 TCP 5001 포트가 열려있는지 확인
```

## 📊 최종 비용 계산 (30만원 크레딧)

### 구성별 월 비용
```
🎯 추천 구성 (단순화):
- High Memory-m2 (시간제): 17,600원
- 공인 IP: 4,032원  
- Object Storage: 340원
- 서버 백업: 5,760원
월 총액: 27,732원
크레딧 지속: 10.8개월

💡 절약 구성:
- High Memory-m2 (시간제): 17,600원
- 공인 IP: 4,032원
- Object Storage: 340원
월 총액: 21,972원
크레딧 지속: 13.7개월

⏰ 운영 시간:
- 평일 9-18시 (9시간)
- 주 5일, 월 20일 기준
```

## ✅ 최종 체크리스트

### 서버 생성 전
- [x] NCP 크레딧 확인 (30만원)
- [ ] GitHub 저장소 준비
- [ ] AI 모델 파일 준비 (505MB)
- [x] 도메인 준비 (선택)

### 서버 생성 후
- [x] Ubuntu 24.04 서버 생성
- [x] 추가 스토리지 연결
- [x] Security Group 설정
- [x] 공인 IP 할당

### 애플리케이션 설정
- [ ] SSH 접속 확인
- [ ] 초기 설정 스크립트 실행
- [ ] Docker/Docker Compose 설치
- [ ] 프로젝트 클론
- [ ] 환경변수 설정
- [ ] AI 모델 파일 업로드
- [ ] Docker 컨테이너 실행
- [ ] 헬스체크 확인

### 자동화 설정
- [ ] 시간제 스케줄러 설정
- [ ] 백업 스크립트 설정
- [ ] 모니터링 설정