#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 로깅 시스템 설정
"""

import logging
import logging.handlers
import os
from datetime import datetime


def setup_logger(name: str = None) -> logging.Logger:
    """
    로거 설정 및 반환
    
    Args:
        name: 로거 이름 (기본값: root logger)
    
    Returns:
        설정된 Logger 객체
    """
    logger = logging.getLogger(name)
    
    # 이미 핸들러가 설정되어 있으면 중복 설정 방지
    if logger.handlers:
        return logger
    
    # 로그 레벨 설정 (환경변수에서 가져오거나 기본값 INFO)
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # 로그 포맷 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (선택적)
    if os.getenv('LOG_TO_FILE', 'false').lower() == 'true':
        # logs 디렉토리 생성
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 날짜별 로그 파일
        log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")
        
        # 회전 파일 핸들러 (10MB, 5개 백업)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    모듈별 로거 가져오기
    
    Args:
        name: 모듈 이름 (보통 __name__ 사용)
    
    Returns:
        Logger 객체
    """
    return setup_logger(name)


# 로그 레벨별 이모지 매핑 (선택적)
LOG_EMOJIS = {
    'DEBUG': '🐛',
    'INFO': 'ℹ️',
    'WARNING': '⚠️',
    'ERROR': '❌',
    'CRITICAL': '🚨'
}


class EmojiFormatter(logging.Formatter):
    """이모지가 포함된 포맷터 (개발 환경용)"""
    
    def format(self, record):
        # 기본 포맷 적용
        msg = super().format(record)
        
        # 개발 환경에서만 이모지 추가
        if os.getenv('FLASK_ENV') == 'development':
            emoji = LOG_EMOJIS.get(record.levelname, '')
            if emoji:
                msg = f"{emoji} {msg}"
        
        return msg


def setup_emoji_logger(name: str = None) -> logging.Logger:
    """
    이모지가 포함된 로거 설정 (개발용)
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # 이모지 포맷터
    formatter = EmojiFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger