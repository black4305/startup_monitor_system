"""
서비스 계층 - 비즈니스 로직 분리
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import threading
import requests
import logging

logger = logging.getLogger(__name__)

class ProgramService:
    """프로그램 관련 비즈니스 로직"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.cache = {
            'programs': [],
            'last_loaded': None,
            'cache_duration': 300  # 5분 캐시
        }
        self.link_status_cache = {}
        self.link_check_in_progress = set()
    
    def get_programs(self, limit: int = 1000, use_cache: bool = True) -> List[Dict]:
        """프로그램 목록 조회 (캐시 지원) - 전체 목록용"""
        try:
            # 캐시 확인
            if use_cache and self._is_cache_valid():
                logger.info(f"💾 캐시된 데이터 사용: {len(self.cache['programs'])}개")
                return self.cache['programs']
            
            # DB에서 새 데이터 로드 (전체 데이터)
            logger.info("📁 Supabase DB에서 전체 데이터 로드")
            programs = self.db_manager.get_programs(limit=10000, active_only=True)  # 충분히 큰 값
            
            # 데이터 정렬 (AI 점수 높은 순)
            programs.sort(key=lambda x: x.get('ai_score', 0), reverse=True)
            
            # 캐시 업데이트
            self.cache['programs'] = programs
            self.cache['last_loaded'] = datetime.now()
            
            logger.info(f"✅ 데이터 로드 완료: {len(programs)}개 (DB)")
            return programs
            
        except Exception as e:
            logger.error(f"❌ 프로그램 로드 실패: {e}")
            return self.cache.get('programs', [])
    
    def get_programs_paginated(self, page: int = 1, per_page: int = 50) -> Dict:
        """페이지네이션된 프로그램 목록 조회"""
        try:
            # 전체 개수 조회
            total_count = self.db_manager.get_total_programs_count(active_only=True)
            
            # offset 계산
            offset = (page - 1) * per_page
            
            # 페이지네이션된 데이터 조회
            programs = self.db_manager.get_programs(
                limit=per_page, 
                offset=offset, 
                active_only=True
            )
            
            # 총 페이지 수 계산
            total_pages = (total_count + per_page - 1) // per_page
            
            logger.info(f"📄 페이지 {page}/{total_pages}: {len(programs)}개 조회 (전체: {total_count}개)")
            
            return {
                'programs': programs,
                'pagination': {
                    'current_page': page,
                    'per_page': per_page,
                    'total_pages': total_pages,
                    'total_count': total_count,
                    'has_prev': page > 1,
                    'has_next': page < total_pages
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 페이지네이션된 프로그램 조회 실패: {e}")
            return {
                'programs': [],
                'pagination': {
                    'current_page': 1,
                    'per_page': per_page,
                    'total_pages': 0,
                    'total_count': 0,
                    'has_prev': False,
                    'has_next': False
                }
            }
    
    def get_total_count(self) -> int:
        """전체 프로그램 개수 조회"""
        try:
            return self.db_manager.get_total_programs_count(active_only=True)
        except Exception as e:
            logger.error(f"❌ 전체 개수 조회 실패: {e}")
            return 0
    
    def _is_cache_valid(self) -> bool:
        """캐시 유효성 검사"""
        if not self.cache['programs'] or not self.cache['last_loaded']:
            return False
        
        now = datetime.now()
        cache_age = (now - self.cache['last_loaded']).total_seconds()
        return cache_age < self.cache['cache_duration']
    
    def refresh_cache(self):
        """캐시 강제 새로고침"""
        self.cache['last_loaded'] = None
        return self.get_programs(use_cache=False)
    
    def check_links_async(self, programs: List[Dict]):
        """비동기로 링크들을 검사하여 캐시에 저장"""
        def check_batch():
            for program in programs:
                url = program.get('url', '')
                program_id = program.get('external_id', program.get('id', ''))
                
                if (url and program_id not in self.link_status_cache 
                    and program_id not in self.link_check_in_progress):
                    
                    self.link_check_in_progress.add(program_id)
                    try:
                        response = requests.head(url, timeout=5, allow_redirects=True)
                        is_valid = response.status_code < 400
                        
                        self.link_status_cache[program_id] = {
                            'valid': is_valid,
                            'checked_at': datetime.now().isoformat(),
                            'url': url
                        }
                    except:
                        self.link_status_cache[program_id] = {
                            'valid': False,
                            'checked_at': datetime.now().isoformat(),
                            'url': url
                        }
                    finally:
                        self.link_check_in_progress.discard(program_id)
        
        thread = threading.Thread(target=check_batch)
        thread.daemon = True
        thread.start()

class DashboardService:
    """대시보드 관련 비즈니스 로직"""
    
    def __init__(self, db_manager, program_service):
        self.db_manager = db_manager
        self.program_service = program_service
    
    def get_dashboard_data(self, limit: int = 10) -> Dict:
        """대시보드 데이터 조회"""
        programs = self.program_service.get_programs()
        recent_programs = programs[:limit]
        
        # 통계 정보
        stats = self.db_manager.get_dashboard_stats()
        if not stats:
            stats = {
                'total_programs': len(programs),
                'active_programs': len(programs),
                'total_feedback': 0,
                'accuracy_percentage': 100.0
            }
        
        return {
            'programs': recent_programs,
            'stats': stats
        }

class AIService:
    """AI 관련 비즈니스 로직"""
    
    def __init__(self, ai_engine):
        self.ai_engine = ai_engine
    
    def get_learning_status(self) -> Dict:
        """AI 학습 상태 조회"""
        try:
            return self.ai_engine.get_learning_status()
        except Exception as e:
            logger.error(f"❌ AI 학습 상태 조회 실패: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def retrain_model(self) -> Dict:
        """AI 모델 재학습"""
        try:
            return self.ai_engine.retrain_models()
        except Exception as e:
            logger.error(f"❌ AI 모델 재학습 실패: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def start_crawling(self, progress_callback=None) -> Dict:
        """크롤링 시작"""
        try:
            import asyncio
            # 비동기 크롤링 실행 - 모든 사이트 크롤링
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.ai_engine.crawl_websites(max_sites=None, progress_callback=progress_callback)
            )
            loop.close()
            return result
        except Exception as e:
            logger.error(f"❌ 크롤링 실행 실패: {e}")
            return {'status': 'error', 'message': str(e)} 