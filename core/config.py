#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 통합 설정 관리
모든 시스템 설정을 중앙화
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

class Config:
    """통합 시스템 설정"""
    
    # 프로젝트 기본 설정
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()
    
    # 디렉토리 경로
    DATA_DIR = PROJECT_ROOT / 'data'
    RESULTS_DIR = PROJECT_ROOT / 'results'
    LOGS_DIR = PROJECT_ROOT / 'logs'
    MODELS_DIR = PROJECT_ROOT / 'models'
    TEMPLATES_DIR = PROJECT_ROOT / 'templates'
    STATIC_DIR = PROJECT_ROOT / 'static'
    
    # Supabase 설정 (환경변수에서 읽기)
    SUPABASE_URL = os.getenv('SUPABASE_URL', '')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')
    SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY', '')
    # 호환성을 위한 별칭
    SUPABASE_SERVICE_ROLE_KEY = os.getenv('SUPABASE_SERVICE_KEY', '')
    
    # GPU/CPU 최적화 설정
    @classmethod
    def setup_device(cls):
        """디바이스 설정 및 최적화"""
        if torch.backends.mps.is_available():
            cls.USE_GPU = True
            cls.DEVICE = torch.device('mps')
            cls.DEVICE_NAME = "Apple Silicon GPU (MPS)"
        elif torch.cuda.is_available():
            cls.USE_GPU = True
            cls.DEVICE = torch.device('cuda')
            cls.DEVICE_NAME = "NVIDIA GPU (CUDA)"
        else:
            cls.USE_GPU = False
            cls.DEVICE = torch.device('cpu')
            cls.DEVICE_NAME = "CPU"
        
        # CPU 부하 최소화 설정
        cls.MAX_CPU_THREADS = 2 if cls.USE_GPU else 4
        cls.MAX_CONCURRENT_REQUESTS = 2 if cls.USE_GPU else 1
        cls.REQUEST_DELAY = 2.0 if cls.USE_GPU else 3.0
        
        # PyTorch 최적화
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['OMP_NUM_THREADS'] = str(cls.MAX_CPU_THREADS)
        torch.set_num_threads(cls.MAX_CPU_THREADS)
        
        return cls.DEVICE
    
    # AI 모델 설정
    BERT_MODEL_NAME = "klue/bert-base"
    SENTENCE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # Colab 훈련 모델 경로 (실제 파일에 맞춤)
    COLAB_SENTENCE_MODEL_DIR = MODELS_DIR / "colab_sentence_model"
    # 실제 존재하지 않는 파일들 - 향후 추가 예정
    COLAB_RF_MODEL_PATH = MODELS_DIR / "production_optimized_model.pkl"  # 미존재
    COLAB_GB_MODEL_PATH = MODELS_DIR / "improved_ensemble_model.pkl"     # 미존재
    COLAB_COMPLETE_MODEL_PATH = MODELS_DIR / "complete_ensemble_model.pkl"  # 미존재
    COLAB_FEATURE_EXTRACTOR_PATH = MODELS_DIR / "feature_extractor.pkl"  # 미존재
    COLAB_MODEL_INFO_PATH = MODELS_DIR / "apple_silicon_production_model_metadata.json"  # 실제 파일
    
    # AI 점수 계산 가중치
    BERT_WEIGHT = 0.6
    ENSEMBLE_WEIGHT = 0.4
    COLAB_MODEL_WEIGHT = 0.8  # Colab 모델 우선순위
    
    # 크롤링 설정
    TIMEOUT = 10
    MAX_RETRIES = 2
    USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    
    # 성능 설정
    MAX_SITES_TO_PROCESS = None  # GPU 사용 시 제한 없음
    MAX_PROGRAMS_PER_SITE = 50  # 더 많은 프로그램 수집
    MAX_RESULTS_TO_KEEP = 200
    MIN_SCORE_THRESHOLD = 30   # 강화학습 최적화된 임계값 (0.3 * 100)
    
    # 사용자 프로필 기본값
    DEFAULT_USER_PROFILE = {
        "business_type": "AI 가족 맞춤형 여행 큐레이션 서비스",
        "stage": "예비창업",
        "region": "광주",
        "keywords": [
            "창업", "지원사업", "스타트업", "벤처", "기업지원",
            "AI", "인공지능", "디지털", "ICT", "소프트웨어",
            "여행", "관광", "콘텐츠", "서비스", "큐레이션",
            "자금지원", "투자", "보조금", "융자", "펀딩"
        ],
        "funding_needs": ["자금지원", "입주프로그램", "멘토링", "사업화지원"]
    }
    
    # 피드백 학습 설정
    MIN_FEEDBACK_FOR_RETRAIN = 3  # 3개 피드백마다 재훈련
    CONFIDENCE_THRESHOLD = 0.7
    INTEREST_THRESHOLD = 0.3
    
    # 로깅 구조화 설정
    
    # === Supabase 로그 (중요한 시스템 데이터) ===
    SUPABASE_LOG_CATEGORIES = {
        'SYSTEM': 'SYSTEM',      # 시스템 시작/종료, 설정 변경
        'CRAWLING': 'CRAWLING',  # 크롤링 시작/완료, 사이트별 결과
        'AI_LEARNING': 'AI_LEARNING',  # 모델 재훈련, 정확도 변화
        'USER_ACTION': 'USER_ACTION',  # 사용자 피드백, 삭제/관심
        'ERROR': 'ERROR'         # 심각한 오류, 복구 불가능한 문제
    }
    
    # === 콘솔 로그 (개발/디버깅용) ===
    CONSOLE_LOG_LEVEL = "INFO"
    CONSOLE_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    
    # === 로그 저장 정책 (로컬 파일 없음) ===
    LOG_RETENTION_DAYS = {
        'console': 0,        # 콘솔은 즉시 소멸
        'supabase': 90,      # Supabase는 90일 보관
    }

    # 웹 대시보드 설정 (환경변수 우선)
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', '5001'))
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'  # debug 모드 비활성화
    SECRET_KEY = os.getenv('SECRET_KEY', 'ai_support_monitor_2025')
    
    # 추가 환경변수 설정들
    MAX_CONCURRENT_REQUESTS_ENV = int(os.getenv('MAX_CONCURRENT_REQUESTS', '10'))
    CRAWL_DELAY_SECONDS = float(os.getenv('CRAWL_DELAY_SECONDS', '1.0'))
    LOG_RETENTION_DAYS_ENV = int(os.getenv('LOG_RETENTION_DAYS', '90'))
    AUTO_CLEANUP_ENABLED = os.getenv('AUTO_CLEANUP_ENABLED', 'true').lower() == 'true'
    
    @classmethod
    def ensure_directories(cls):
        """필요한 디렉토리 생성 (로그 폴더 제외)"""
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, 
                         cls.TEMPLATES_DIR, cls.STATIC_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod 
    def should_log_to_supabase(cls, level: str, category: str) -> bool:
        """Supabase에 로그를 저장해야 하는지 판단"""
        # INFO 이상 레벨이고 정의된 카테고리인 경우
        return (level in ['INFO', 'WARNING', 'ERROR', 'CRITICAL'] and 
                category in cls.SUPABASE_LOG_CATEGORIES.values())
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """설정을 딕셔너리로 반환"""
        return {
            'supabase_url': cls.SUPABASE_URL,
            'device': str(cls.DEVICE) if hasattr(cls, 'DEVICE') else 'cpu',
            'max_cpu_threads': cls.MAX_CPU_THREADS if hasattr(cls, 'MAX_CPU_THREADS') else 4,
            'flask_port': cls.FLASK_PORT,
            'min_feedback_for_retrain': cls.MIN_FEEDBACK_FOR_RETRAIN
        }

# 시스템 초기화
Config.setup_device()
Config.ensure_directories() 