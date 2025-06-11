#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 AI 지원사업 모니터링 시스템 - 런처 스크립트 (리팩토링 버전)
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 메인 애플리케이션 실행
if __name__ == '__main__':
    try:
        from core.app import main
        print("🚀 리팩토링된 AI 지원사업 모니터링 시스템 시작")
        main()
    except ImportError as e:
        print(f"❌ 애플리케이션 import 실패: {e}")
        print("💡 core/app.py 파일이 있는지 확인해주세요.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 시스템 시작 실패: {e}")
        sys.exit(1) 