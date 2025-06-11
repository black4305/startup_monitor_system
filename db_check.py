#!/usr/bin/env python3
"""
DB 상태 확인 스크립트 - 프로그램 데이터 확인
"""
import os
import sys
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.database import get_database_manager

def main():
    """DB 상태 확인"""
    print("🔍 DB 상태 확인 중...")
    
    try:
        # DB 매니저 생성
        db_manager = get_database_manager()
        
        print("\n=== 프로그램 데이터 확인 ===")
        
        # 1. 전체 프로그램 개수 확인
        total_count = db_manager.get_total_programs_count(active_only=False)
        active_count = db_manager.get_total_programs_count(active_only=True)
        
        print(f"📊 전체 프로그램: {total_count}개")
        print(f"📊 활성 프로그램: {active_count}개")
        
        if total_count == 0:
            print("❌ 저장된 프로그램이 없습니다!")
            print("💡 크롤링을 실행해야 합니다.")
        else:
            # 2. 샘플 프로그램 5개 조회
            print(f"\n=== 최근 프로그램 5개 샘플 ===")
            programs = db_manager.get_programs(limit=5, active_only=True)
            
            if programs:
                for i, program in enumerate(programs, 1):
                    title = program.get('title', '제목없음')[:50]
                    ai_score = program.get('ai_score', 0)
                    site_name = program.get('site_name', '사이트없음')
                    created_at = program.get('created_at', '')[:10]
                    
                    print(f"{i}. 📝 {title}...")
                    print(f"   🤖 AI점수: {ai_score}, 🌐 사이트: {site_name}, 📅 생성: {created_at}")
                    print()
            else:
                print("❌ 프로그램 데이터를 조회할 수 없습니다.")
        
        # 3. 대시보드 통계 확인
        print("=== 대시보드 통계 ===")
        stats = db_manager.get_dashboard_stats()
        for key, value in stats.items():
            print(f"📈 {key}: {value}")
            
    except Exception as e:
        print(f"❌ DB 확인 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 