#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌐 AI 지원사업 모니터링 시스템 - 메인 애플리케이션 (리팩토링 버전)
"""

import logging
import sys
import os
from pathlib import Path

# Flask 관련
from flask import Flask
from flask_cors import CORS

# 프로젝트 모듈
from core import Config, DatabaseManager, AIEngine
from core.database import get_database_manager
from core.ai_engine import get_ai_engine
from core.services import ProgramService, DashboardService, AIService
from core.routes import create_routes

# ============================================
# 로깅 설정
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# 외부 라이브러리 로그 레벨 조정
for lib in ['urllib3', 'requests', 'httpx', 'httpcore']:
    logging.getLogger(lib).setLevel(logging.WARNING)

# ============================================
# 애플리케이션 팩토리
# ============================================

def create_app():
    """Flask 애플리케이션 팩토리"""
    
    # 프로젝트 루트 경로 설정
    project_root = Path(__file__).parent.parent
    template_dir = project_root / 'templates'
    static_dir = project_root / 'static'
    
    # Flask 앱 생성 (템플릿 경로 지정)
    app = Flask(__name__, 
                template_folder=str(template_dir),
                static_folder=str(static_dir))
    app.config['SECRET_KEY'] = Config.SECRET_KEY or 'your-secret-key-here'
    
    # CORS 설정
    CORS(app)
    
    # 서비스 초기화
    try:
        logger.info("🚀 AI 지원사업 모니터링 시스템 초기화")
        
        # 데이터베이스 연결
        logger.info("🔗 데이터베이스 연결 중...")
        db_manager = get_database_manager()
        logger.info("✅ 데이터베이스 연결 완료")
        
        # AI 엔진 초기화
        logger.info("🤖 AI 엔진 초기화 중...")
        ai_engine = get_ai_engine()
        logger.info("✅ AI 엔진 초기화 완료")
        
        # 서비스 계층 생성
        program_service = ProgramService(db_manager)
        dashboard_service = DashboardService(db_manager, program_service)
        ai_service = AIService(ai_engine)
        
        # 라우트 등록
        routes_bp = create_routes(program_service, dashboard_service, ai_service)
        app.register_blueprint(routes_bp)
        
        logger.info("🎉 시스템 초기화 완료")
        logger.info("🌐 웹 서버: http://localhost:5001")
        
    except Exception as e:
        logger.error(f"❌ 초기화 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    return app

# ============================================
# 메인 실행부
# ============================================

def main():
    """메인 실행 함수"""
    try:
        # 환경변수 로드
        from dotenv import load_dotenv
        load_dotenv()
        
        # Flask 애플리케이션 생성
        app = create_app()
        
        # 서버 실행
        port = int(os.getenv('PORT', 5001))
        debug = os.getenv('FLASK_ENV') == 'development'
        
        logger.info(f"🌐 서버 시작: http://localhost:{port}")
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("👋 서버가 정상적으로 종료되었습니다.")
    except Exception as e:
        logger.error(f"❌ 서버 실행 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main() 