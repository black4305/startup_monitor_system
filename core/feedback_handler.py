"""
피드백 처리 모듈 - 사용자 피드백 수집 및 AI 학습 (강화학습 통합)
"""

import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime
import json
import asyncio
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class FeedbackHandler:
    """사용자 피드백 처리 및 학습 관리 (강화학습 통합)"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.feedback_patterns = {}
        self.deletion_patterns = {}
        
        # 강화학습 연동
        self.rl_optimizer = None
        self.feedback_queue = []
        self.min_feedback_for_rl = 5  # 5개 피드백마다 강화학습 실행
        
        # 강화학습 모듈 로드 시도
        self._initialize_reinforcement_learning()
        
    def _initialize_reinforcement_learning(self):
        """강화학습 모듈 초기화"""
        try:
            from .reinforcement_learning_optimizer import ReinforcementLearningOptimizer
            
            # AI 엔진 참조 (나중에 설정)
            self.rl_optimizer = None
            logger.info("🤖 강화학습 모듈 준비 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 강화학습 모듈 로드 실패: {e}")
            self.rl_optimizer = None
    
    def set_ai_engine(self, ai_engine):
        """AI 엔진 참조 설정 (강화학습용)"""
        try:
            if self.rl_optimizer is None:
                from .reinforcement_learning_optimizer import ReinforcementLearningOptimizer
                self.rl_optimizer = ReinforcementLearningOptimizer(ai_engine=ai_engine)
                logger.info("✅ 강화학습 최적화기 초기화 완료")
        except Exception as e:
            logger.error(f"❌ AI 엔진 설정 실패: {e}")
    
    def record_user_feedback(self, program_data: Dict, action: str, reason: str = "") -> bool:
        """사용자 피드백 기록 및 실시간 학습"""
        try:
            program_id = program_data.get('external_id', program_data.get('id', ''))
            
            # DB에 피드백 저장
            success = self.db_manager.insert_user_feedback(
                program_external_id=program_id,
                action=action,
                reason=reason,
                confidence=program_data.get('ai_score', 0) / 100.0
            )
            
            if success:
                # 패턴 업데이트
                self.update_patterns_from_feedback(program_data, action, reason)
                
                # 통계 업데이트
                self.update_feedback_stats(action, reason)
                
                # 강화학습 큐에 추가
                self._add_to_rl_queue(program_data, action, reason)
                
                # 삭제 액션인 경우 유사한 프로그램 자동 삭제 (백그라운드)
                if action == 'delete':
                    import threading
                    delete_thread = threading.Thread(
                        target=self.auto_delete_similar_programs,
                        args=(program_data, reason)
                    )
                    delete_thread.daemon = True
                    delete_thread.start()
                
                # 재훈련 필요성 체크
                self.check_retrain_needed()
                
                logger.info(f"✅ 피드백 기록: {action} - {program_data.get('title', '')[:50]}")
                return True
            else:
                logger.error(f"❌ 피드백 저장 실패")
                return False
                
        except Exception as e:
            logger.error(f"❌ 피드백 처리 실패: {e}")
            return False
    
    def _add_to_rl_queue(self, program_data: Dict, action: str, reason: str):
        """강화학습 큐에 피드백 추가"""
        try:
            # 피드백 데이터 구성
            feedback_item = {
                'program_data': program_data,
                'action': action,
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
                'title': program_data.get('title', ''),
                'content': program_data.get('content', ''),
                'ai_score': program_data.get('ai_score', 0)
            }
            
            self.feedback_queue.append(feedback_item)
            
            # 충분한 피드백이 쌓이면 강화학습 실행
            if len(self.feedback_queue) >= self.min_feedback_for_rl:
                self._trigger_reinforcement_learning()
                
            logger.info(f"🧠 강화학습 큐 추가: {len(self.feedback_queue)}개 대기 중")
            
        except Exception as e:
            logger.error(f"❌ 강화학습 큐 추가 실패: {e}")
    
    def _trigger_reinforcement_learning(self):
        """강화학습 트리거 (백그라운드에서 실행)"""
        if not self.rl_optimizer or len(self.feedback_queue) < self.min_feedback_for_rl:
            return
        
        def run_rl_optimization():
            try:
                logger.info(f"🚀 강화학습 시작: {len(self.feedback_queue)}개 피드백 처리")
                
                # 실제 딥러닝 모델 재훈련 실행
                result = self.rl_optimizer.retrain_deep_learning_model(self.feedback_queue)
                
                if result['status'] == 'success':
                    logger.info("✅ 딥러닝 모델 재훈련 성공!")
                    logger.info(f"   성능 향상: {result.get('improvement', 0):+.3f}")
                    logger.info(f"   훈련된 에포크: {result.get('epochs_trained', 0)}")
                    
                    # 성공 시 큐 비우기
                    self.feedback_queue.clear()
                    
                    # DB에 학습 결과 로깅
                    self.db_manager.log_system_event(
                        level='INFO',
                        category='AI_LEARNING',
                        message='딥러닝 모델 재훈련 완료',
                        details=result
                    )
                    
                else:
                    logger.warning(f"⚠️ 딥러닝 모델 재훈련 실패: {result.get('message', '알 수 없는 오류')}")
                    
                    # 실패한 경우 메타파라미터 최적화라도 시도
                    if result['status'] != 'insufficient_data':
                        logger.info("🔄 메타파라미터 최적화로 폴백...")
                        fallback_result = self.rl_optimizer.optimize_from_feedback(self.feedback_queue)
                        
                        if fallback_result.get('status') == 'success':
                            logger.info("✅ 메타파라미터 최적화 성공")
                            self.feedback_queue.clear()
                        
            except Exception as e:
                logger.error(f"❌ 강화학습 실행 실패: {e}")
        
        # 백그라운드 스레드에서 실행
        thread = threading.Thread(target=run_rl_optimization, daemon=True)
        thread.start()
    
    def force_reinforcement_learning(self) -> Dict[str, Any]:
        """강제 강화학습 실행 (수동 트리거)"""
        try:
            logger.info("🔧 수동 강화학습 트리거")
            
            if not self.rl_optimizer:
                return {
                    'status': 'no_optimizer',
                    'message': '강화학습 모듈이 없습니다.',
                    'timestamp': datetime.now().isoformat()
                }
            
            if len(self.feedback_queue) == 0:
                # 큐가 비어있으면 최근 피드백을 가져와서 시도
                recent_feedback = self.db_manager.get_recent_feedback(limit=10)
                if not recent_feedback:
                    return {
                        'status': 'no_feedback',
                        'message': '학습할 피드백 데이터가 없습니다.',
                        'timestamp': datetime.now().isoformat()
                    }
                feedback_data = recent_feedback
            else:
                feedback_data = self.feedback_queue
            
            logger.info(f"📚 {len(feedback_data)}개 피드백으로 딥러닝 모델 재훈련 시작...")
            
            # 실제 딥러닝 모델 재훈련 실행
            result = self.rl_optimizer.retrain_deep_learning_model(feedback_data)
            
            if result['status'] == 'success':
                logger.info("🎉 수동 딥러닝 재훈련 성공!")
                
                # 성공 시 큐 비우기
                self.feedback_queue.clear()
                
                # 상세한 결과 포함
                return {
                    'status': 'success',
                    'message': '딥러닝 모델 재훈련이 완료되었습니다!',
                    'result': result,
                    'improvement': result.get('improvement', 0),
                    'training_samples': result.get('training_samples', 0),
                    'epochs_trained': result.get('epochs_trained', 0),
                    'timestamp': datetime.now().isoformat()
                }
                
            elif result['status'] == 'insufficient_data':
                # 데이터 부족 시 메타파라미터 최적화라도 시도
                logger.info("🔄 데이터 부족으로 메타파라미터 최적화 시도...")
                fallback_result = self.rl_optimizer.optimize_from_feedback(feedback_data)
                
                return {
                    'status': 'fallback_success',
                    'message': '데이터가 부족하여 메타파라미터 최적화를 수행했습니다.',
                    'result': fallback_result,
                    'data_count': result.get('data_count', 0),
                    'timestamp': datetime.now().isoformat()
                }
                
            else:
                return {
                    'status': 'failed',
                    'message': result.get('message', '재훈련 실패'),
                    'error_details': result,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ 수동 강화학습 실패: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_rl_status(self) -> Dict[str, Any]:
        """강화학습 상태 조회"""
        try:
            status = {
                'rl_available': self.rl_optimizer is not None,
                'feedback_queue_size': len(self.feedback_queue),
                'min_feedback_threshold': self.min_feedback_for_rl,
                'ready_for_rl': len(self.feedback_queue) >= self.min_feedback_for_rl
            }
            
            if self.rl_optimizer:
                status.update(self.rl_optimizer.get_rl_optimization_status())
            
            return status
            
        except Exception as e:
            logger.error(f"❌ 강화학습 상태 조회 실패: {e}")
            return {'rl_available': False, 'error': str(e)}
    
    def update_patterns_from_feedback(self, program_data: Dict, action: str, reason: str):
        """피드백을 바탕으로 패턴 업데이트"""
        try:
            title = program_data.get('title', '')
            content = program_data.get('content', '')
            site_name = program_data.get('site_name', '')
            
            # 키워드 추출
            keywords = self.extract_keywords_from_text(f"{title} {content}")
            
            # 패턴별 업데이트
            pattern_updates = [
                ('keyword', action, keyword, reason) for keyword in keywords
            ]
            
            if site_name:
                pattern_updates.append(('site', action, site_name, reason))
            
            # 실제 DB에 패턴 저장
            for pattern_type, category, pattern_key, reason in pattern_updates:
                self.db_manager.update_learning_pattern(
                    pattern_type=pattern_type,
                    category=category,
                    pattern_key=pattern_key,
                    reason=reason,
                    frequency_increment=1
                )
            
            logger.info(f"✅ 학습 패턴 업데이트 완료: {len(pattern_updates)}개 패턴 저장")
            
        except Exception as e:
            logger.error(f"❌ 패턴 업데이트 실패: {e}")
    
    def extract_keywords_from_text(self, text: str) -> List[str]:
        """텍스트에서 키워드 추출"""
        # 간단한 키워드 추출 (실제로는 더 정교한 NLP 기법 사용 가능)
        keywords = []
        
        # 지원사업 관련 키워드들
        important_keywords = [
            '창업', '지원', '사업', '기업', '투자', '융자', '보조금',
            'R&D', '연구', '개발', '기술', '혁신', '벤처', '스타트업',
            '중소기업', '소상공인', '육성', '활성화', '촉진'
        ]
        
        text_lower = text.lower()
        for keyword in important_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
        
        return keywords
    
    def update_feedback_stats(self, action: str, reason: str):
        """피드백 통계 업데이트"""
        try:
            # 현재 통계 가져오기
            current_stats = self.db_manager.get_user_feedback_stats()
            
            # 통계 정보를 시스템 로그에 기록
            self.db_manager.log_system_event(
                level='INFO',
                category='USER_ACTION',
                message=f'사용자 피드백: {action}',
                details={
                    'action': action,
                    'reason': reason,
                    'total_feedback': current_stats.get('total_feedback', 0) + 1,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"❌ 피드백 통계 업데이트 실패: {e}")
    
    def check_retrain_needed(self):
        """재훈련 필요성 체크"""
        try:
            from .config import Config
            
            # 최근 피드백 수 확인
            recent_feedback = self.db_manager.get_recent_feedback(limit=100)
            
            if len(recent_feedback) >= Config.MIN_FEEDBACK_FOR_RETRAIN:
                # 재훈련 알림 생성
                self.create_retrain_notification(len(recent_feedback))
                
                logger.info(f"🔄 재훈련 권장: {len(recent_feedback)}개 피드백 누적")
                
        except Exception as e:
            logger.error(f"❌ 재훈련 체크 실패: {e}")
    
    def create_retrain_notification(self, feedback_count: int):
        """재훈련 알림 생성"""
        try:
            self.db_manager.log_system_event(
                level='INFO',
                category='AI_LEARNING',
                message='AI 모델 재훈련 권장',
                details={
                    'feedback_count': feedback_count,
                    'recommendation': 'retrain_models',
                    'priority': 'high' if feedback_count > 10 else 'medium',
                    'timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"❌ 재훈련 알림 생성 실패: {e}")
    
    def prepare_training_data(self) -> List[Tuple[str, int]]:
        """피드백 데이터를 학습용 데이터로 변환"""
        try:
            # 피드백 데이터 가져오기
            feedback_data = self.db_manager.get_recent_feedback(limit=1000)
            
            training_data = []
            for feedback in feedback_data:
                action = feedback.get('action', '')
                program_info = feedback.get('program_info', {})
                
                if isinstance(program_info, str):
                    try:
                        program_info = json.loads(program_info)
                    except:
                        continue
                
                title = program_info.get('title', '')
                content = program_info.get('content', '')
                text = f"{title} {content}".strip()
                
                if text and action in ['keep', 'delete']:
                    label = 1 if action == 'keep' else 0
                    training_data.append((text, label))
            
            logger.info(f"✅ 학습 데이터 준비 완료: {len(training_data)}개")
            return training_data
            
        except Exception as e:
            logger.error(f"❌ 학습 데이터 준비 실패: {e}")
            return []
    
    def auto_delete_similar_programs(self, deleted_program: Dict, reason: str):
        """삭제된 프로그램과 유사한 프로그램들을 자동으로 찾아서 삭제"""
        try:
            logger.info(f"🔍 유사 프로그램 자동 삭제 시작: {deleted_program.get('title', '')[:50]}")
            
            # 자동 삭제 상태 초기화
            self.auto_delete_status = {
                'is_running': True,
                'total_programs': 0,
                'processed': 0,
                'deleted': 0,
                'current_program': '',
                'trigger_program': deleted_program.get('title', ''),
                'started_at': datetime.now().isoformat()
            }
            
            # 삭제된 프로그램의 특징 추출
            deleted_keywords = self.extract_keywords_from_text(
                f"{deleted_program.get('title', '')} {deleted_program.get('content', '')}"
            )
            deleted_site = deleted_program.get('site_name', '')
            
            # 활성화된 모든 프로그램 가져오기
            active_programs = self.db_manager.get_all_programs(
                include_inactive=False,
                limit=1000
            )
            
            self.auto_delete_status['total_programs'] = len(active_programs)
            
            auto_deleted_count = 0
            auto_deleted_programs = []
            
            for i, program in enumerate(active_programs):
                # 진행 상황 업데이트
                self.auto_delete_status['processed'] = i + 1
                self.auto_delete_status['current_program'] = program.get('title', '')[:50]
                
                # 이미 삭제된 프로그램은 건너뛰기
                if program.get('external_id') == deleted_program.get('external_id'):
                    continue
                
                # 유사도 계산
                similarity_score = self.calculate_similarity(
                    program, deleted_program, deleted_keywords, deleted_site
                )
                
                # 유사도가 높으면 자동 삭제 (70% 이상)
                if similarity_score >= 0.7:
                    logger.info(f"  → 유사 프로그램 발견 (유사도: {similarity_score:.1%}): {program.get('title', '')[:50]}")
                    
                    # 1. 자동 삭제 피드백 저장
                    feedback_success = self.db_manager.insert_user_feedback(
                        program_external_id=program.get('external_id'),
                        action='delete',
                        reason=f"자동 삭제 - '{deleted_program.get('title', '')[:30]}'와 유사 (유사도: {similarity_score:.1%})",
                        confidence=similarity_score
                    )
                    
                    # 2. 프로그램 비활성화
                    if self.db_manager.deactivate_program(program.get('external_id')):
                        auto_deleted_count += 1
                        self.auto_delete_status['deleted'] = auto_deleted_count
                        auto_deleted_programs.append({
                            'title': program.get('title', ''),
                            'similarity': similarity_score,
                            'external_id': program.get('external_id')
                        })
                    
            # 자동 삭제 완료
            self.auto_delete_status['is_running'] = False
            self.auto_delete_status['completed_at'] = datetime.now().isoformat()
            
            if auto_deleted_count > 0:
                logger.info(f"✅ 총 {auto_deleted_count}개 프로그램 자동 삭제 완료")
                
                # 시스템 로그에 기록
                self.db_manager.log_system_event(
                    level='INFO',
                    category='AUTO_DELETE',
                    message=f'유사 프로그램 자동 삭제',
                    details={
                        'trigger_program': deleted_program.get('title', ''),
                        'auto_deleted_count': auto_deleted_count,
                        'auto_deleted_programs': auto_deleted_programs[:10],  # 최대 10개만 로그에 기록
                        'timestamp': datetime.now().isoformat()
                    }
                )
            
        except Exception as e:
            logger.error(f"❌ 유사 프로그램 자동 삭제 실패: {e}")
    
    def calculate_similarity(self, program1: Dict, program2: Dict, keywords2: List[str], site2: str) -> float:
        """두 프로그램의 유사도 계산 (0~1)"""
        try:
            # 1. 키워드 유사도 (40%)
            keywords1 = self.extract_keywords_from_text(
                f"{program1.get('title', '')} {program1.get('content', '')}"
            )
            
            keyword_overlap = len(set(keywords1) & set(keywords2))
            keyword_total = len(set(keywords1) | set(keywords2))
            keyword_similarity = keyword_overlap / keyword_total if keyword_total > 0 else 0
            
            # 2. 제목 유사도 (30%)
            title1 = program1.get('title', '').lower()
            title2 = program2.get('title', '').lower()
            
            # 간단한 문자열 유사도 (실제로는 더 정교한 알고리즘 사용 가능)
            title_words1 = set(title1.split())
            title_words2 = set(title2.split())
            title_overlap = len(title_words1 & title_words2)
            title_total = len(title_words1 | title_words2)
            title_similarity = title_overlap / title_total if title_total > 0 else 0
            
            # 3. 사이트 일치도 (20%)
            site1 = program1.get('site_name', '')
            site_similarity = 1.0 if site1 == site2 else 0.0
            
            # 4. AI 점수 유사도 (10%)
            score1 = program1.get('ai_score', 50)
            score2 = program2.get('ai_score', 50)
            score_diff = abs(score1 - score2)
            score_similarity = 1.0 - (score_diff / 100.0)  # 점수 차이가 작을수록 유사
            
            # 총 유사도 계산
            total_similarity = (
                keyword_similarity * 0.4 +
                title_similarity * 0.3 +
                site_similarity * 0.2 +
                score_similarity * 0.1
            )
            
            return total_similarity
            
        except Exception as e:
            logger.error(f"❌ 유사도 계산 실패: {e}")
            return 0.0
    
    def get_auto_delete_status(self) -> Dict[str, Any]:
        """자동 삭제 진행 상황 반환"""
        if hasattr(self, 'auto_delete_status'):
            return self.auto_delete_status
        else:
            return {'is_running': False}
    
    def get_learning_patterns(self) -> Dict[str, Any]:
        """학습된 패턴 정보 반환"""
        try:
            patterns = self.db_manager.get_learning_patterns()
            
            # 패턴을 유형별로 그룹화
            grouped_patterns = {
                'keywords': [],
                'sites': [],
                'actions': []
            }
            
            for pattern in patterns:
                pattern_type = pattern.get('pattern_type', '')
                if pattern_type == 'keyword':
                    grouped_patterns['keywords'].append(pattern)
                elif pattern_type == 'site':
                    grouped_patterns['sites'].append(pattern)
                else:
                    grouped_patterns['actions'].append(pattern)
            
            return grouped_patterns
            
        except Exception as e:
            logger.error(f"❌ 학습 패턴 조회 실패: {e}")
            return {'keywords': [], 'sites': [], 'actions': []}
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """피드백 요약 정보 (강화학습 정보 포함)"""
        try:
            # 통계 정보
            stats = self.db_manager.get_user_feedback_stats()
            
            # 최근 피드백
            recent_feedback = self.db_manager.get_recent_feedback(limit=10)
            
            # 학습 패턴
            patterns = self.get_learning_patterns()
            
            # 강화학습 상태
            rl_status = self.get_rl_status()
            
            return {
                'statistics': stats,
                'recent_feedback': recent_feedback,
                'learning_patterns': patterns,
                'reinforcement_learning': rl_status,
                'summary': {
                    'total_feedback': stats.get('total_feedback', 0),
                    'accuracy_trend': stats.get('accuracy_percentage', 0),
                    'pattern_count': sum(len(p) for p in patterns.values()),
                    'rl_queue_size': len(self.feedback_queue),
                    'last_updated': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 피드백 요약 생성 실패: {e}")
            return {
                'statistics': {},
                'recent_feedback': [],
                'learning_patterns': {'keywords': [], 'sites': [], 'actions': []},
                'reinforcement_learning': {'rl_available': False},
                'summary': {}
            } 