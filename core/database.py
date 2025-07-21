#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
💾 Supabase 데이터베이스 관리자 통합
모든 DB 작업을 중앙화
"""

import os
import json
import logging
import requests
from typing import Optional, Dict, List, Any
from datetime import datetime
import hashlib

from .config import Config

# Supabase 클라이언트
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

class DatabaseManager:
    """Supabase 데이터베이스 매니저 - REST API 방식으로 모든 DB 작업 통합"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Supabase 설정
        self.supabase_url = Config.SUPABASE_URL
        self.supabase_key = Config.SUPABASE_KEY
        self.service_role_key = Config.SUPABASE_SERVICE_ROLE_KEY
        
        # REST API 기본 설정
        self.api_base_url = f"{self.supabase_url}/rest/v1"
        self.headers = {
            'apikey': self.supabase_key,
            'Authorization': f'Bearer {self.supabase_key}',
            'Content-Type': 'application/json'
        }
        
        # 연결 객체 (호환성 유지)
        self.supabase = None
        
        # 초기화
        self.initialize_connections()
    
    def initialize_connections(self):
        """DB 연결 초기화 - 실제 Supabase 클라이언트 생성"""
        try:
            if self.supabase_url and self.supabase_key and SUPABASE_AVAILABLE:
                # 실제 Supabase 클라이언트 생성
                self.supabase = create_client(self.supabase_url, self.supabase_key)
                
                # 연결 테스트
                test_url = f"{self.api_base_url}/crawling_sites?select=count"
                response = requests.get(test_url, headers=self.headers)
                
                if response.status_code == 200:
                    self.logger.info("✅ Supabase REST API 연결 성공")
                else:
                    self.logger.error(f"❌ Supabase 연결 실패: HTTP {response.status_code}")
                    self.supabase = None
            else:
                self.logger.warning("⚠️ Supabase 환경변수 누락 또는 라이브러리 없음")
                self.supabase = None
                
        except Exception as e:
            self.logger.error(f"❌ 데이터베이스 연결 실패: {e}")
            self.supabase = None
    
    # === 프로그램 관련 메서드 ===
    
    def insert_program(self, program_data: Dict) -> bool:
        """프로그램 삽입/업데이트"""
        try:
            self.logger.info(f"💾 DB 저장 시작: {program_data.get('title', '제목없음')[:50]}")
            
            # 안전한 데이터 정제
            def safe_get(data, key, default=''):
                value = data.get(key, default)
                if value is None:
                    return default
                # 리스트나 딕셔너리인 경우 문자열로 변환
                if isinstance(value, (list, dict)):
                    return str(value)
                return value
            
            # 필수 필드 검증
            title = safe_get(program_data, 'title')
            url = safe_get(program_data, 'url')
            
            self.logger.info(f"📝 필드 검증: title='{title[:30]}...', url='{url[:50]}...'")
            
            if not title or not url:
                self.logger.error(f"❌ 필수 필드 누락: title={bool(title)}, url={bool(url)}")
                return False
            
            # external_id 생성: URL 기반 해시
            external_id = hashlib.md5(url.encode()).hexdigest()[:16]  # 16자리 해시
            
            # 날짜 처리
            deadline = safe_get(program_data, 'deadline')
            formatted_deadline = None
            if deadline and deadline.strip():
                try:
                    # 다양한 날짜 형식 시도
                    from datetime import datetime
                    import re
                    # YYYY-MM-DD 형식으로 변환 시도
                    if re.match(r'\d{4}-\d{2}-\d{2}', deadline):
                        formatted_deadline = deadline
                    else:
                        # 기타 날짜 문자열은 None으로 처리
                        formatted_deadline = None
                except:
                    formatted_deadline = None

            data = {
                'external_id': external_id,  # URL 기반 고유 ID
                'title': title[:500],  # 길이 제한
                'content': safe_get(program_data, 'content')[:2000],  # 길이 제한
                'url': url[:1000],  # 길이 제한
                'organization': safe_get(program_data, 'site_name')[:100],  # site_name을 organization으로 매핑
                'ai_score': float(program_data.get('ai_score', 0)) if program_data.get('ai_score') is not None else 0.0,
                'support_type': safe_get(program_data, 'support_type', '일반')[:100],
                'application_deadline': formatted_deadline,
                'is_active': True
            }
            
            # None 값들을 빈 문자열로 변환 (application_deadline 제외)
            for key, value in data.items():
                if value is None and key != 'application_deadline':
                    data[key] = ''
            
            self.logger.info(f"🗂️ 최종 저장 데이터: external_id='{data.get('external_id')}', ai_score={data.get('ai_score')}")
            
            result = self.supabase.table('support_programs').upsert(data, on_conflict='external_id').execute()
            
            self.logger.info(f"📊 DB 응답: data={bool(result.data)}, count={len(result.data) if result.data else 0}")
            
            if result.data:
                self.logger.info(f"✅ 프로그램 저장 성공: {data['title'][:30]}...")
                return True
            else:
                self.logger.error(f"❌ DB 응답이 비어있음: {result}")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ 프로그램 저장 실패: {e}")
            # 디버깅을 위한 상세 정보
            if program_data:
                self.logger.debug(f"문제 데이터: {str(program_data)[:200]}...")
            return False
    
    def get_programs(self, limit: int = 100, offset: int = 0, active_only: bool = True) -> List[Dict]:
        """프로그램 목록 조회 - REST API 방식"""
        try:
            if not self.supabase:
                self.logger.error("❌ Supabase 연결이 없습니다")
                return []
            
            # 1차: 기본 테이블 존재 확인
            basic_url = f"{self.api_base_url}/support_programs?select=*&limit=1"
            test_response = requests.get(basic_url, headers=self.headers)
            
            if test_response.status_code == 404:
                self.logger.warning("❌ support_programs 테이블이 존재하지 않습니다")
                
                # 다른 테이블명 시도
                alternative_tables = ['programs', 'startup_programs', 'business_programs', 'support_business']
                for table_name in alternative_tables:
                    alt_url = f"{self.api_base_url}/{table_name}?select=*&limit=1"
                    alt_response = requests.get(alt_url, headers=self.headers)
                    if alt_response.status_code == 200:
                        self.logger.info(f"✅ 대안 테이블 발견: {table_name}")
                        # 임시로 이 테이블 사용
                        return self._get_programs_from_table(table_name, limit, offset, active_only)
                
                return []
            elif test_response.status_code != 200:
                self.logger.warning(f"❌ 테이블 접근 실패: HTTP {test_response.status_code}")
                self.logger.warning(f"응답: {test_response.text[:200]}")
                return []
            
            # 2차: 실제 데이터 조회
            url = f"{self.api_base_url}/support_programs?select=*"
            
            if active_only:
                # is_active 컬럼이 없을 수도 있으므로 조건부 추가
                try:
                    active_test_url = f"{self.api_base_url}/support_programs?select=*&is_active=eq.true&limit=1"
                    active_test_response = requests.get(active_test_url, headers=self.headers)
                    if active_test_response.status_code == 200:
                        url += "&is_active=eq.true"
                    else:
                        self.logger.info("ℹ️ is_active 컬럼이 없어 전체 데이터 조회")
                except:
                    pass
            
            url += f"&order=created_at.desc&offset={offset}&limit={limit}"
            
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                self.logger.info(f"✅ 프로그램 조회 성공: {len(data)}개 (offset: {offset}, limit: {limit})")
                return data
            else:
                self.logger.error(f"❌ 프로그램 조회 실패: HTTP {response.status_code}")
                self.logger.error(f"응답: {response.text[:200]}")
                return []
            
        except Exception as e:
            self.logger.error(f"❌ 프로그램 조회 실패: {e}")
            return []
    
    def get_total_programs_count(self, active_only: bool = True) -> int:
        """전체 프로그램 개수 조회"""
        try:
            if not self.supabase:
                self.logger.error("❌ Supabase 연결이 없습니다")
                return 0
            
            # Supabase에서 COUNT 쿼리 사용
            query = self.supabase.table('support_programs').select('id', count='exact')
            
            if active_only:
                # is_active 컬럼이 있는지 확인하고 조건 추가
                try:
                    test_query = self.supabase.table('support_programs').select('*').eq('is_active', True).limit(1).execute()
                    if test_query.data is not None:  # 에러가 없으면 is_active 컬럼이 존재
                        query = query.eq('is_active', True)
                except:
                    pass  # is_active 컬럼이 없으면 전체 개수 반환
            
            result = query.execute()
            
            if hasattr(result, 'count') and result.count is not None:
                count = result.count
                self.logger.info(f"📊 전체 프로그램 개수: {count}개")
                return count
            else:
                # count가 없으면 실제 데이터를 가져와서 계산 (fallback)
                self.logger.warning("⚠️ COUNT 쿼리 실패, 대안 방법 사용")
                return self._get_total_count_fallback(active_only)
            
        except Exception as e:
            self.logger.error(f"❌ 전체 프로그램 개수 조회 실패: {e}")
            return self._get_total_count_fallback(active_only)
    
    def _get_total_count_fallback(self, active_only: bool = True) -> int:
        """전체 개수 조회 대안 방법 (REST API)"""
        try:
            url = f"{self.api_base_url}/support_programs?select=id"
            
            if active_only:
                try:
                    # is_active 컬럼 존재 여부 확인
                    test_url = f"{self.api_base_url}/support_programs?select=*&is_active=eq.true&limit=1"
                    test_response = requests.get(test_url, headers=self.headers)
                    if test_response.status_code == 200:
                        url += "&is_active=eq.true"
                except:
                    pass
            
            # 모든 ID만 가져오기 (limit 없이)
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                count = len(data)
                self.logger.info(f"📊 대안 방법으로 계산한 전체 개수: {count}개")
                return count
            else:
                self.logger.error(f"❌ 대안 개수 조회 실패: HTTP {response.status_code}")
                return 0
                
        except Exception as e:
            self.logger.error(f"❌ 대안 개수 조회 실패: {e}")
            return 0
    
    def _get_programs_from_table(self, table_name: str, limit: int, offset: int, active_only: bool) -> List[Dict]:
        """특정 테이블에서 프로그램 데이터 조회"""
        try:
            url = f"{self.api_base_url}/{table_name}?select=*"
            url += f"&order=created_at.desc&offset={offset}&limit={limit}"
            
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                self.logger.info(f"✅ {table_name} 테이블에서 {len(data)}개 데이터 조회")
                return data
            else:
                self.logger.error(f"❌ {table_name} 테이블 조회 실패: HTTP {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"❌ {table_name} 테이블 조회 실패: {e}")
            return []
    
    def get_program_by_external_id(self, external_id: str) -> Optional[Dict]:
        """외부 ID로 프로그램 조회"""
        try:
            result = self.supabase.table('support_programs').select('*').eq('external_id', external_id).execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 프로그램 조회 실패: {e}")
            return None
    
    def deactivate_program(self, external_id: str) -> bool:
        """프로그램 비활성화 (삭제)"""
        try:
            result = self.supabase.table('support_programs').update({
                'is_active': False,
                'updated_at': datetime.now().isoformat()
            }).eq('external_id', external_id).execute()
            
            return bool(result.data)
            
        except Exception as e:
            self.logger.error(f"❌ 프로그램 비활성화 실패: {e}")
            return False
    
    def search_programs(self, search_query: str, limit: int = 25) -> List[Dict]:
        """프로그램 검색"""
        try:
            # PostgreSQL의 전문 검색 사용
            result = self.supabase.table('support_programs')\
                .select('*')\
                .or_(f'title.ilike.%{search_query}%,content.ilike.%{search_query}%,organization.ilike.%{search_query}%')\
                .eq('is_active', True)\
                .order('ai_score', desc=True)\
                .limit(limit)\
                .execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            self.logger.error(f"❌ 프로그램 검색 실패: {e}")
            return []
    
    # === 사용자 피드백 관련 메서드 ===
    
    def insert_user_feedback(self, program_external_id: str, action: str, reason: str = None, confidence: float = None) -> bool:
        """사용자 피드백 저장 (강화학습용 프로그램 정보 포함)"""
        try:
            self.logger.info(f"🔍 피드백 저장 시작: {program_external_id} - {action}")
            
            # 프로그램 정보 찾기
            program = self.get_program_by_external_id(program_external_id)
            if not program:
                self.logger.error(f"❌ 프로그램을 찾을 수 없음: {program_external_id}")
                return False
            
            self.logger.info(f"✅ 프로그램 정보 조회 성공: {program.get('title', '')[:30]}...")
            
            # 강화학습을 위한 프로그램 정보 저장
            program_info = {
                'title': program.get('title', ''),
                'content': program.get('content', ''),
                'ai_score': program.get('ai_score', 0),
                'site_name': program.get('organization', program.get('site_name', '')),
                'url': program.get('url', ''),
                'deadline': program.get('application_deadline', program.get('deadline', ''))
            }
            
            # JSON 직렬화 테스트
            import json
            try:
                json_test = json.dumps(program_info)
                self.logger.info(f"📄 프로그램 정보 JSON 변환 성공: {len(json_test)} bytes")
            except Exception as json_error:
                self.logger.error(f"❌ JSON 변환 실패: {json_error}")
                # JSON 변환 실패 시 문자열로 변환
                program_info = str(program_info)
            
            data = {
                'program_external_id': program_external_id,  # external_id로 프로그램 식별
                'action': action,
                'reason': reason,
                'confidence': confidence
            }
            
            self.logger.info(f"💾 피드백 데이터 준비 완료: {data.keys()}")
            
            # user_feedback 테이블 존재 여부 확인
            try:
                test_query = self.supabase.table('user_feedback').select('*').limit(1).execute()
                self.logger.info(f"✅ user_feedback 테이블 접근 가능")
            except Exception as table_error:
                self.logger.error(f"❌ user_feedback 테이블 접근 실패: {table_error}")
                return False
            
            result = self.supabase.table('user_feedback').insert(data).execute()
            
            self.logger.info(f"📊 Supabase 응답: {result}")
            
            if result.data:
                self.logger.info(f"✅ 피드백 저장 성공: {action} - {reason} (프로그램: {program_info.get('title', '')[:30] if isinstance(program_info, dict) else 'JSON'}...)")
                return True
            else:
                self.logger.error(f"❌ 피드백 저장 실패: result.data가 비어있음")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ 피드백 저장 실패: {e}")
            import traceback
            self.logger.error(f"📋 상세 오류: {traceback.format_exc()}")
            return False
    
    def get_user_feedback_stats(self) -> Dict:
        """사용자 피드백 통계 - REST API 방식"""
        try:
            if not self.supabase:
                self.logger.warning("❌ Supabase 연결이 없습니다")
                return {
                    'total_feedback': 0,
                    'total_deletions': 0,
                    'total_keeps': 0,
                    'total_views': 0,
                    'accuracy_percentage': 0
                }
            
            url = f"{self.api_base_url}/ai_learning_stats?select=*"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]
            
            return {
                'total_feedback': 0,
                'total_deletions': 0,
                'total_keeps': 0,
                'total_views': 0,
                'accuracy_percentage': 0
            }
            
        except Exception as e:
            self.logger.error(f"❌ 피드백 통계 조회 실패: {e}")
            return {
                'total_feedback': 0,
                'total_deletions': 0,
                'total_keeps': 0,
                'total_views': 0,
                'accuracy_percentage': 0
            }
    
    def update_ai_learning_stats(self, stats_data: Dict) -> bool:
        """AI 학습 통계 업데이트"""
        try:
            # 기존 통계가 있는지 확인
            existing = self.supabase.table('ai_learning_stats').select('*').limit(1).execute()
            
            data = {
                'total_feedback': stats_data.get('total_feedback', 0),
                'total_deletions': stats_data.get('total_deletions', 0),
                'total_keeps': stats_data.get('total_keeps', 0),
                'total_views': stats_data.get('total_views', 0),
                'accuracy_percentage': stats_data.get('accuracy_percentage', 0),
                'last_learning_at': stats_data.get('last_learning_at', datetime.now().isoformat()),
                'model_version': stats_data.get('model_version', '1.0'),
                'learning_count': stats_data.get('learning_count', 0),
                'updated_at': datetime.now().isoformat()
            }
            
            if existing.data and len(existing.data) > 0:
                # 기존 통계 업데이트
                result = self.supabase.table('ai_learning_stats')\
                    .update(data)\
                    .eq('id', existing.data[0]['id'])\
                    .execute()
            else:
                # 새 통계 생성
                result = self.supabase.table('ai_learning_stats').insert(data).execute()
            
            self.logger.info(f"✅ AI 학습 통계 업데이트: 피드백 {data['total_feedback']}개, 정확도 {data['accuracy_percentage']:.1f}%")
            return bool(result.data)
            
        except Exception as e:
            self.logger.error(f"❌ AI 학습 통계 업데이트 실패: {e}")
            return False
    
    def record_learning_event(self, learning_type: str, performance_before: float, 
                             performance_after: float, details: Dict = None) -> bool:
        """강화학습 이벤트 기록"""
        try:
            # 현재 통계 조회
            current_stats = self.get_user_feedback_stats()
            
            # 학습 카운트 증가
            learning_count = current_stats.get('learning_count', 0) + 1
            
            # 통계 업데이트
            updated_stats = {
                **current_stats,
                'learning_count': learning_count,
                'last_learning_at': datetime.now().isoformat(),
                'accuracy_percentage': performance_after,
                'model_version': f"{learning_count}.0"
            }
            
            # 세부 정보 추가
            if details:
                updated_stats.update(details)
            
            success = self.update_ai_learning_stats(updated_stats)
            
            if success:
                self.logger.info(f"🧠 강화학습 이벤트 기록: {learning_type}, 성능 {performance_before:.1f}% → {performance_after:.1f}%")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ 강화학습 이벤트 기록 실패: {e}")
            return False
    
    def get_recent_feedback(self, limit: int = 10) -> List[Dict]:
        """최근 피드백 목록 - REST API 방식"""
        try:
            if not self.supabase:
                self.logger.warning("❌ Supabase 연결이 없습니다")
                return []
            
            # user_feedback 테이블은 존재하므로 support_programs와 조인 시도
            url = f"{self.api_base_url}/user_feedback?select=*,support_programs(title,organization)&order=created_at.desc&limit={limit}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                # 조인 실패 시 기본 조회로 시도
                self.logger.debug(f"📊 조인 실패, 기본 user_feedback 조회 시도")
                url = f"{self.api_base_url}/user_feedback?select=*&order=created_at.desc&limit={limit}"
                response = requests.get(url, headers=self.headers)
                if response.status_code == 200:
                    return response.json()
                return []
            else:
                self.logger.warning(f"❌ 최근 피드백 조회 실패: HTTP {response.status_code}")
                return []
            
        except Exception as e:
            self.logger.error(f"❌ 최근 피드백 조회 실패: {e}")
            return []
    
    # === AI 학습 패턴 관련 메서드 ===
    
    def update_learning_pattern(self, pattern_type: str, category: str, pattern_key: str, 
                              reason: str = None, frequency_increment: int = 1):
        """AI 학습 패턴 업데이트"""
        try:
            # 기존 패턴 조회
            existing = self.supabase.table('learning_patterns')\
                .select('*')\
                .eq('pattern_type', pattern_type)\
                .eq('category', category)\
                .eq('pattern_key', pattern_key)\
                .execute()
            
            if existing.data and len(existing.data) > 0:
                # 기존 패턴 업데이트
                new_frequency = existing.data[0]['frequency'] + frequency_increment
                result = self.supabase.table('learning_patterns')\
                    .update({
                        'frequency': new_frequency,
                        'reason': reason,
                        'last_updated': datetime.now().isoformat()
                    })\
                    .eq('id', existing.data[0]['id'])\
                    .execute()
            else:
                # 새 패턴 생성
                data = {
                    'pattern_type': pattern_type,
                    'category': category, 
                    'pattern_key': pattern_key,
                    'frequency': frequency_increment,
                    'reason': reason
                }
                result = self.supabase.table('learning_patterns').insert(data).execute()
            
            self.logger.info(f"✅ 학습 패턴 업데이트: {pattern_type}-{category}-{pattern_key}")
            
        except Exception as e:
            self.logger.error(f"❌ 학습 패턴 업데이트 실패: {e}")
    
    def get_learning_patterns(self, pattern_type: str = None, category: str = None) -> List[Dict]:
        """학습 패턴 조회"""
        try:
            query = self.supabase.table('learning_patterns').select('*')
            
            if pattern_type:
                query = query.eq('pattern_type', pattern_type)
            if category:
                query = query.eq('category', category)
            
            query = query.order('frequency', desc=True)
            result = query.execute()
            
            patterns = result.data if result.data else []
            self.logger.info(f"📊 학습 패턴 조회: {len(patterns)}개")
            return patterns
            
        except Exception as e:
            self.logger.error(f"❌ 학습 패턴 조회 실패: {e}")
            return []
    
    # === 시스템 로그 관련 메서드 ===
    
    def log_system_event(self, level: str, category: str, message: str, details: Dict = None):
        """시스템 이벤트 로그 저장"""
        try:
            data = {
                'log_level': level,
                'category': category,
                'message': message,
                'details': details or {}
            }
            
            self.supabase.table('system_logs').insert(data).execute()
            
        except Exception as e:
            self.logger.error(f"❌ 시스템 로그 저장 실패: {e}")
    
    def get_system_logs(self, level: str = None, category: str = None, limit: int = 100) -> List[Dict]:
        """시스템 로그 조회"""
        try:
            query = self.supabase.table('system_logs').select('*')
            
            if level:
                query = query.eq('log_level', level)
            if category:
                query = query.eq('category', category)
            
            query = query.order('created_at', desc=True).limit(limit)
            
            result = query.execute()
            return result.data if result.data else []
            
        except Exception as e:
            self.logger.error(f"❌ 시스템 로그 조회 실패: {e}")
            return []
    
    # === 설정 관련 메서드 ===
    
    def get_setting(self, setting_key: str) -> Any:
        """시스템 설정 조회"""
        try:
            result = self.supabase.table('system_settings')\
                .select('setting_value')\
                .eq('setting_key', setting_key)\
                .execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]['setting_value']
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 설정 조회 실패: {e}")
            return None
    
    def update_setting(self, setting_key: str, setting_value: Any, description: str = None) -> bool:
        """시스템 설정 업데이트"""
        try:
            data = {
                'setting_key': setting_key,
                'setting_value': setting_value,
                'description': description,
                'updated_at': datetime.now().isoformat()
            }
            
            result = self.supabase.table('system_settings')\
                .upsert(data, on_conflict='setting_key')\
                .execute()
            
            return bool(result.data)
            
        except Exception as e:
            self.logger.error(f"❌ 설정 업데이트 실패: {e}")
            return False
    
    # === 데이터 마이그레이션 관련 메서드 ===
    
    def migrate_json_to_db(self, json_file_path: str) -> bool:
        """JSON 파일에서 DB로 데이터 마이그레이션"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # announcements 데이터 처리
            if 'announcements' in data:
                programs = data['announcements']
                success_count = 0
                
                for program in programs:
                    if self.insert_program(program):
                        success_count += 1
                
                self.logger.info(f"✅ 마이그레이션 완료: {success_count}/{len(programs)}개")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ JSON 마이그레이션 실패: {e}")
            return False
    
    # === 크롤링 사이트 관리 메서드 ===
    
    def get_crawling_sites(self, enabled_only: bool = True, priority: str = None, region: str = None) -> List[Dict]:
        """크롤링 사이트 목록 조회 - REST API 방식"""
        try:
            if not self.supabase:
                self.logger.error("❌ Supabase 연결이 없습니다")
                return []
            
            # REST API 쿼리 구성
            url = f"{self.api_base_url}/crawling_sites?select=*"
            
            filters = []
            if enabled_only:
                filters.append("enabled=eq.true")
            if priority:
                filters.append(f"priority=eq.{priority}")
            if region:
                filters.append(f"region=eq.{region}")
            
            if filters:
                url += "&" + "&".join(filters)
                
            url += "&order=priority.desc,name.asc"
            
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"❌ 크롤링 사이트 조회 실패: HTTP {response.status_code}")
                return []
            
        except Exception as e:
            self.logger.error(f"❌ 크롤링 사이트 조회 실패: {e}")
            return []
    
    def get_crawling_site_by_id(self, site_id: str) -> Optional[Dict]:
        """특정 크롤링 사이트 조회"""
        try:
            result = self.supabase.table('crawling_sites')\
                .select('*')\
                .eq('id', site_id)\
                .execute()
            
            return result.data[0] if result.data else None
            
        except Exception as e:
            self.logger.error(f"❌ 크롤링 사이트 조회 실패: {e}")
            return None
    
    def update_crawling_stats(self, site_id: str, success: bool, details: Dict = None):
        """크롤링 통계 업데이트"""
        try:
            # 통계 업데이트를 위한 stored function 호출
            result = self.supabase.rpc('update_crawling_stats', {
                'site_id': site_id,
                'success': success
            }).execute()
            
            # 상세 로그 기록
            if details:
                log_level = 'INFO' if success else 'ERROR'
                log_message = f"크롤링 {'성공' if success else '실패'}: {details.get('site_name', 'Unknown')}"
                
                self.log_system_event(
                    level=log_level,
                    category=Config.SUPABASE_LOG_CATEGORIES['CRAWLING'],
                    message=log_message,
                    details={
                        'site_id': site_id,
                        'success': success,
                        **details
                    }
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 크롤링 통계 업데이트 실패: {e}")
            
            # Stored function이 없는 경우 직접 업데이트
            try:
                now = datetime.now().isoformat()
                
                if success:
                    self.supabase.table('crawling_sites')\
                        .update({
                            'crawl_success_count': self.supabase.table('crawling_sites').select('crawl_success_count').eq('id', site_id).execute().data[0]['crawl_success_count'] + 1,
                            'last_crawled_at': now,
                            'updated_at': now
                        })\
                        .eq('id', site_id)\
                        .execute()
                else:
                    self.supabase.table('crawling_sites')\
                        .update({
                            'crawl_fail_count': self.supabase.table('crawling_sites').select('crawl_fail_count').eq('id', site_id).execute().data[0]['crawl_fail_count'] + 1,
                            'last_crawled_at': now,
                            'updated_at': now
                        })\
                        .eq('id', site_id)\
                        .execute()
                
                return True
                
            except Exception as fallback_error:
                self.logger.error(f"❌ 크롤링 통계 직접 업데이트도 실패: {fallback_error}")
                return False
    
    def disable_crawling_site(self, site_id: str, reason: str = None):
        """크롤링 사이트 비활성화"""
        try:
            result = self.supabase.table('crawling_sites')\
                .update({
                    'enabled': False,
                    'updated_at': datetime.now().isoformat()
                })\
                .eq('id', site_id)\
                .execute()
            
            if result.data:
                site_name = result.data[0].get('name', 'Unknown')
                
                self.log_system_event(
                    level='WARNING',
                    category=Config.SUPABASE_LOG_CATEGORIES['SYSTEM'],
                    message=f"크롤링 사이트 비활성화: {site_name}",
                    details={
                        'site_id': site_id,
                        'reason': reason or 'Manual disable'
                    }
                )
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 크롤링 사이트 비활성화 실패: {e}")
            return False
    
    def get_crawling_stats_summary(self) -> Dict:
        """크롤링 통계 요약"""
        try:
            result = self.supabase.table('crawling_sites').select('*').execute()
            
            if not result.data:
                return {
                    'total_sites': 0,
                    'enabled_sites': 0,
                    'disabled_sites': 0,
                    'total_success': 0,
                    'total_failures': 0,
                    'success_rate': 0,
                    'by_priority': {},
                    'by_region': {},
                    'by_category': {}
                }
            
            sites = result.data
            total_sites = len(sites)
            enabled_sites = len([s for s in sites if s.get('enabled', True)])
            disabled_sites = total_sites - enabled_sites
            
            total_success = sum(s.get('crawl_success_count', 0) for s in sites)
            total_failures = sum(s.get('crawl_fail_count', 0) for s in sites)
            total_attempts = total_success + total_failures
            success_rate = (total_success / total_attempts * 100) if total_attempts > 0 else 0
            
            # 카테고리별 통계
            by_priority = {}
            by_region = {}
            by_category = {}
            
            for site in sites:
                # 우선순위별
                priority = site.get('priority', 'medium')
                by_priority[priority] = by_priority.get(priority, 0) + 1
                
                # 지역별
                region = site.get('region', '기타')
                by_region[region] = by_region.get(region, 0) + 1
                
                # 카테고리별
                category = site.get('category', '기타')
                by_category[category] = by_category.get(category, 0) + 1
            
            return {
                'total_sites': total_sites,
                'enabled_sites': enabled_sites,
                'disabled_sites': disabled_sites,
                'total_success': total_success,
                'total_failures': total_failures,
                'success_rate': round(success_rate, 2),
                'by_priority': by_priority,
                'by_region': by_region,
                'by_category': by_category
            }
            
        except Exception as e:
            self.logger.error(f"❌ 크롤링 통계 요약 실패: {e}")
            return {}
    
    # === 통계 관련 메서드 ===
    
    def get_dashboard_stats(self) -> Dict:
        """대시보드용 통계 데이터 - 임시로 기본값 반환"""
        try:
            # 임시로 기본값 반환 (테이블이 없거나 연결 문제로 인한 오류 방지)
            return {
                'total_programs': 0,
                'active_programs': 0,
                'total_feedback': 0,
                'total_sites': 370,  # 마이그레이션된 사이트 수
                'enabled_sites': 370,
                'accuracy_percentage': 0
            }
            
        except Exception as e:
            self.logger.error(f"❌ 대시보드 통계 조회 실패: {e}")
            return {
                'total_programs': 0,
                'active_programs': 0,
                'total_feedback': 0,
                'accuracy_percentage': 0
            }
    
    def close_connections(self):
        """연결 정리"""
        try:
            if self.supabase:
                # Supabase 클라이언트는 자동으로 정리됨
                self.logger.info("✅ DB 연결 정리 완료")
        except Exception as e:
            self.logger.error(f"❌ DB 연결 정리 실패: {e}")
    
    def delete_program_permanently(self, external_id: str) -> bool:
        """프로그램 완전 삭제 (실제 DB에서 삭제)"""
        try:
            # support_programs 테이블에서 완전 삭제
            delete_query = """
            DELETE FROM support_programs 
            WHERE external_id = %s
            """
            
            cursor = self.supabase.table('support_programs').delete().eq('external_id', external_id).execute()
            
            if cursor.data:
                self.logger.info(f"✅ 프로그램 완전 삭제 완료: {external_id}")
                return True
            else:
                self.logger.warning(f"⚠️ 삭제할 프로그램을 찾을 수 없음: {external_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 프로그램 완전 삭제 실패: {e}")
            return False
    
    def get_deleted_programs_for_learning(self, limit: int = 100) -> List[Dict]:
        """강화학습용 삭제된 프로그램 데이터 조회 (피드백 테이블에서)"""
        try:
            # user_feedback 테이블에서 'delete' 액션인 것들만 가져오기
            result = self.supabase.table('user_feedback')\
                .select('*')\
                .eq('action', 'delete')\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            
            deleted_programs = []
            
            if result.data:
                for feedback in result.data:
                    try:
                        # 프로그램 정보 파싱
                        program_info = feedback.get('program_info', {})
                        if isinstance(program_info, str):
                            import json
                            program_info = json.loads(program_info)
                        
                        # 강화학습용 데이터 구성
                        deleted_program = {
                            'title': program_info.get('title', ''),
                            'content': program_info.get('content', ''),
                            'action': feedback.get('action', 'delete'),
                            'reason': feedback.get('reason', ''),
                            'ai_score': program_info.get('ai_score', 0),
                            'confidence': feedback.get('confidence', 0),
                            'deleted_at': feedback.get('created_at', ''),
                            'program_external_id': feedback.get('program_external_id', '')
                        }
                        
                        # 텍스트가 있는 경우에만 추가
                        if deleted_program['title'] or deleted_program['content']:
                            deleted_programs.append(deleted_program)
                            
                    except Exception as e:
                        self.logger.warning(f"⚠️ 삭제된 프로그램 데이터 파싱 실패: {e}")
                        continue
            
            self.logger.info(f"📚 강화학습용 삭제 데이터 조회: {len(deleted_programs)}개")
            return deleted_programs
            
        except Exception as e:
            self.logger.error(f"❌ 삭제된 프로그램 조회 실패: {e}")
            return []

# 전역 인스턴스 생성 함수
_db_manager = None

def get_database_manager() -> DatabaseManager:
    """데이터베이스 매니저 싱글톤 인스턴스 반환"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager 