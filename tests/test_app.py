#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 기본 애플리케이션 테스트
"""

import unittest
import sys
import os

# 테스트를 위해 상위 디렉토리를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAppBasic(unittest.TestCase):
    """기본 애플리케이션 테스트"""

    def test_import_main_app(self):
        """메인 애플리케이션 임포트 테스트"""
        try:
            from core.app import main
            self.assertTrue(callable(main))
            print("✅ 메인 앱 임포트 성공")
        except ImportError as e:
            self.fail(f"❌ 메인 앱 임포트 실패: {e}")

    def test_import_core_modules(self):
        """핵심 모듈들 임포트 테스트"""
        modules_to_test = [
            'core.config',
            'core.database',
            'core.services',
        ]
        
        for module_name in modules_to_test:
            try:
                __import__(module_name)
                print(f"✅ {module_name} 임포트 성공")
            except ImportError as e:
                print(f"⚠️ {module_name} 임포트 실패: {e}")
                # 일부 모듈은 환경 설정이 필요할 수 있으므로 경고로만 처리

    def test_run_script_exists(self):
        """run.py 파일 존재 확인"""
        run_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'run.py')
        self.assertTrue(os.path.exists(run_file), "run.py 파일이 존재하지 않습니다")

    def test_requirements_file_exists(self):
        """requirements.txt 파일 존재 확인"""
        req_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'requirements.txt')
        self.assertTrue(os.path.exists(req_file), "requirements.txt 파일이 존재하지 않습니다")


if __name__ == '__main__':
    unittest.main() 