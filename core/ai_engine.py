"""
🤖 AI 엔진 - 메인 AI 시스템 (리팩토링 버전)
분리된 모듈들을 통합하여 관리하는 중앙 엔진
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from .config import Config
from .ai_models import AIModelManager
from .crawler import WebCrawler
from .feedback_handler import FeedbackHandler

logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """사용자 프로필 정의"""
    business_type: str = Config.DEFAULT_USER_PROFILE["business_type"]
    stage: str = Config.DEFAULT_USER_PROFILE["stage"]
    region: str = Config.DEFAULT_USER_PROFILE["region"]
    funding_needs: List[str] = None
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.funding_needs is None:
            self.funding_needs = Config.DEFAULT_USER_PROFILE["funding_needs"]
        if self.keywords is None:
            self.keywords = Config.DEFAULT_USER_PROFILE["keywords"]

@dataclass
class SupportProgram:
    """지원사업 공고 정보"""
    title: str
    content: str
    url: str
    site_name: str
    deadline: str = ""
    support_type: str = ""
    target: str = ""
    amount: str = ""
    ai_score: float = 0.0
    personalized_score: float = 0.0
    enhanced_score: float = 0.0
    recommendation_reason: str = ""
    extracted_at: str = ""

class AIEngine:
    """통합 AI 엔진 - 메인 컨트롤러"""
    
    def __init__(self, db_manager, user_profile: UserProfile = None):
        self.db_manager = db_manager
        self.user_profile = user_profile or UserProfile()
        
        # 분리된 모듈들 초기화
        self.model_manager = AIModelManager()
        self.crawler = WebCrawler(db_manager)
        self.feedback_handler = FeedbackHandler(db_manager)
        
        # 초기화
        self.initialize_system()
    
    def initialize_system(self):
        """시스템 초기화"""
        try:
            logger.info("🚀 AI 엔진 초기화 시작")
            
            # AI 모델 초기화
            self.model_manager.initialize_models()
            
            # 강화학습 연동을 위한 AI 엔진 참조 설정
            self.feedback_handler.set_ai_engine(self)
            
            # 시스템 로그
            self.db_manager.log_system_event(
                level='INFO',
                category='SYSTEM',
                message='AI 엔진 초기화 완료',
                details={
                    'models_loaded': self.model_manager.models_loaded,
                    'reinforcement_learning_enabled': self.feedback_handler.rl_optimizer is not None,
                    'user_profile': {
                        'business_type': self.user_profile.business_type,
                        'stage': self.user_profile.stage,
                        'region': self.user_profile.region
                    },
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            logger.info("✅ AI 엔진 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ AI 엔진 초기화 실패: {e}")
            raise
    
    # ============================================
    # 크롤링 관련 메서드
    # ============================================
    
    async def crawl_websites(self, max_sites: int = None, priority: str = None, 
                           region: str = None, progress_callback=None) -> Dict[str, Any]:
        """웹사이트 실시간 스트리밍 크롤링 (크롤링과 동시에 DB 저장)"""
        try:
            logger.info("🔍 실시간 스트리밍 크롤링 시작")
            logger.info(f"📊 필터링 기준: MIN_SCORE_THRESHOLD = {Config.MIN_SCORE_THRESHOLD}")
            
            # 통계 초기화
            stats = {
                "total_sites": 0,
                "total_programs": 0,
                "analyzed_programs": 0,
                "saved_programs": 0,
                "failed_programs": 0,
                "scores": [],
                "save_errors": [],
                "start_time": datetime.now()
            }
            
            # 실시간 크롤링 콜백 정의
            async def process_program_callback(program_data: Dict, site_info: Dict) -> bool:
                """각 프로그램을 즉시 분석하고 저장하는 콜백"""
                try:
                    stats["total_programs"] += 1
                    
                    # 1단계: AI 분석
                    analyzed_program = self.analyze_program_with_ai(program_data)
                    stats["analyzed_programs"] += 1
                    stats["scores"].append(analyzed_program.ai_score)
                    
                    logger.info(f"🧠 프로그램 {stats['total_programs']}: '{analyzed_program.title[:30]}...' - 점수: {analyzed_program.ai_score:.1f}")
                    
                    # 2단계: 점수 필터링
                    if analyzed_program.ai_score >= Config.MIN_SCORE_THRESHOLD:
                        # 3단계: 즉시 DB 저장
                        try:
                            program_dict = self.safe_program_to_dict(analyzed_program)
                            logger.info(f"💾 DB 저장 시작: '{analyzed_program.title[:30]}...'")
                            
                            if self.db_manager.insert_program(program_dict):
                                stats["saved_programs"] += 1
                                logger.info(f"✅ 프로그램 저장 성공 (총 {stats['saved_programs']}개 저장됨)")
                                return True
                            else:
                                stats["failed_programs"] += 1
                                error_msg = f"프로그램 {stats['total_programs']}: DB insert 실패"
                                stats["save_errors"].append(error_msg)
                                logger.error(f"❌ {error_msg}")
                                return False
                                
                        except Exception as e:
                            stats["failed_programs"] += 1
                            error_msg = f"프로그램 {stats['total_programs']}: 저장 중 예외 - {str(e)}"
                            stats["save_errors"].append(error_msg)
                            logger.error(f"❌ {error_msg}")
                            return False
                    else:
                        logger.info(f"❌ 점수 미달 (< {Config.MIN_SCORE_THRESHOLD}) - 저장하지 않음")
                        return False
                        
                except Exception as e:
                    stats["failed_programs"] += 1
                    error_msg = f"프로그램 {stats['total_programs']}: 분석 중 예외 - {str(e)}"
                    stats["save_errors"].append(error_msg)
                    logger.error(f"❌ {error_msg}")
                    return False
            
            # 사이트별 진행 상황 콜백
            async def site_progress_callback(site_info: Dict, programs_found: int, completed_sites: int, total_sites: int):
                """사이트별 진행 상황 로깅"""
                stats["total_sites"] = completed_sites
                
                # 중간 통계 출력
                if stats["scores"]:
                    avg_score = sum(stats["scores"]) / len(stats["scores"])
                    logger.info(f"📊 사이트 {completed_sites}/{total_sites} 완료 | "
                              f"프로그램 {stats['total_programs']}개 발견 | "
                              f"저장 {stats['saved_programs']}개 | "
                              f"평균점수 {avg_score:.1f}")
                
                # 외부 콜백 호출 (웹 인터페이스용)
                if progress_callback:
                    await progress_callback({
                        'completed_sites': completed_sites,
                        'total_sites': total_sites,
                        'programs_found': stats['total_programs'],
                        'programs_saved': stats['saved_programs'],
                        'avg_score': avg_score if stats["scores"] else 0
                    })
            
            # 🔄 실시간 스트리밍 크롤링 실행
            await self.crawler.crawl_websites_streaming(
                max_sites=max_sites,
                priority=priority,
                region=region,
                program_callback=process_program_callback,
                site_progress_callback=site_progress_callback
            )
            
            # 최종 통계 계산
            elapsed_time = (datetime.now() - stats["start_time"]).total_seconds()
            avg_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
            max_score = max(stats["scores"]) if stats["scores"] else 0
            min_score = min(stats["scores"]) if stats["scores"] else 0
            
            # 결과 요약
            result = {
                'status': 'success',
                'total_sites_processed': stats['total_sites'],
                'total_programs_found': stats['total_programs'],
                'programs_analyzed': stats['analyzed_programs'],
                'programs_saved': stats['saved_programs'],
                'programs_failed': stats['failed_programs'],
                'score_threshold': Config.MIN_SCORE_THRESHOLD,
                'score_stats': {
                    'avg_score': round(avg_score, 1),
                    'max_score': round(max_score, 1),
                    'min_score': round(min_score, 1),
                    'total_scores': len(stats["scores"])
                },
                'processing_time_seconds': round(elapsed_time, 1),
                'save_errors': stats["save_errors"][:10],  # 최대 10개 오류만 표시
                'timestamp': datetime.now().isoformat()
            }
            
            # 시스템 이벤트 로깅
            self.db_manager.log_system_event(
                level='INFO',
                category='STREAMING_CRAWLING',
                message='실시간 스트리밍 크롤링 완료',
                details=result
            )
            
            # 최종 결과 로깅
            logger.info(f"🎉 실시간 크롤링 최종 결과:")
            logger.info(f"   🌐 처리 사이트: {stats['total_sites']}개")
            logger.info(f"   📋 발견 프로그램: {stats['total_programs']}개")
            logger.info(f"   🧠 분석 완료: {stats['analyzed_programs']}개")
            logger.info(f"   💾 저장 성공: {stats['saved_programs']}개")
            logger.info(f"   ❌ 저장 실패: {stats['failed_programs']}개")
            logger.info(f"   📊 평균 점수: {avg_score:.1f} (최고: {max_score:.1f}, 최저: {min_score:.1f})")
            logger.info(f"   ⏱️ 처리 시간: {elapsed_time:.1f}초")
            
            if stats["save_errors"]:
                logger.error(f"   ⚠️ 주요 오류 {len(stats['save_errors'])}개:")
                for i, error in enumerate(stats["save_errors"][:5]):  # 최대 5개만 표시
                    logger.error(f"      {i+1}. {error}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 실시간 크롤링 실패: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # ============================================
    # AI 분석 관련 메서드
    # ============================================
    
    def analyze_program_with_ai(self, program_data: Dict) -> SupportProgram:
        """프로그램 상세 AI 분석 (제목 + 공고 내용 종합 분석)"""
        try:
            # SupportProgram 객체 생성
            program = SupportProgram(
                title=program_data.get('title', ''),
                content=program_data.get('content', ''),
                url=program_data.get('url', ''),
                site_name=program_data.get('site_name', ''),
                extracted_at=program_data.get('extracted_at', datetime.now().isoformat())
            )
            
            # 상세 내용 분석을 위한 텍스트 준비
            analysis_text = self.prepare_analysis_text(program)
            
            # 1단계: 기본 AI 점수 계산 (제목 + 내용)
            program.ai_score = self.calculate_comprehensive_ai_score(program, analysis_text)
            
            # 2단계: 공고 내용에서 핵심 정보 추출
            program = self.extract_program_details(program)
            
            # 3단계: 개인화 점수 계산 (상세 내용 반영)
            program.personalized_score = self.calculate_personalized_score(program)
            
            # 4단계: 향상된 점수 계산
            program.enhanced_score = self.calculate_enhanced_score(program)
            
            # 5단계: 지원 유형 세부 분류
            program.support_type = self.classify_support_type(program.title, program.content)
            
            # 6단계: 상세한 추천 이유 생성
            program.recommendation_reason = self.generate_detailed_recommendation_reason(program)
            
            logger.debug(f"🧠 AI 분석 완료: '{program.title[:30]}...' - 점수: {program.ai_score:.1f}")
            
            return program
            
        except Exception as e:
            logger.error(f"❌ AI 분석 실패: {e}")
            # 기본값으로 SupportProgram 반환
            return SupportProgram(
                title=program_data.get('title', ''),
                content=program_data.get('content', ''),
                url=program_data.get('url', ''),
                site_name=program_data.get('site_name', ''),
                ai_score=0.0
            )
    
    def calculate_personalized_score(self, program: SupportProgram) -> float:
        """개인화 점수 계산"""
        try:
            score = program.ai_score
            
            # 사용자 프로필 기반 가중치 적용
            text = f"{program.title} {program.content}".lower()
            
            # 비즈니스 타입 매칭
            if self.user_profile.business_type.lower() in text:
                score += 15
            
            # 단계별 가중치
            if self.user_profile.stage == "예비창업" and any(word in text for word in ['예비', '준비', '계획']):
                score += 10
            
            # 지역 매칭
            if self.user_profile.region in text:
                score += 20
            
            # 키워드 매칭
            matched_keywords = sum(1 for keyword in self.user_profile.keywords if keyword in text)
            score += matched_keywords * 5
            
            return min(score, 100)
            
        except Exception as e:
            logger.warning(f"개인화 점수 계산 실패: {e}")
            return program.ai_score
    
    def calculate_enhanced_score(self, program: SupportProgram) -> float:
        """향상된 점수 계산 (패턴 학습 반영)"""
        try:
            base_score = program.personalized_score
            
            # 학습된 패턴 가져오기
            patterns = self.feedback_handler.get_learning_patterns()
            
            # 패턴 기반 점수 조정
            text = f"{program.title} {program.content}".lower()
            
            for keyword_pattern in patterns.get('keywords', []):
                keyword = keyword_pattern.get('pattern_key', '').lower()
                category = keyword_pattern.get('category', '')
                frequency = keyword_pattern.get('frequency', 0)
                
                if keyword in text:
                    if category == 'keep':
                        base_score += frequency * 2
                    elif category == 'delete':
                        base_score -= frequency * 2
            
            return min(max(base_score, 0), 100)
            
        except Exception as e:
            logger.warning(f"향상된 점수 계산 실패: {e}")
            return program.personalized_score
    
    def classify_support_type(self, title: str, content: str) -> str:
        """지원 유형 분류"""
        text = f"{title} {content}".lower()
        
        if any(word in text for word in ['투자', '융자', '자금']):
            return "자금지원"
        elif any(word in text for word in ['r&d', '연구', '개발', '기술']):
            return "R&D지원"
        elif any(word in text for word in ['창업', '인큐베이팅', '액셀러레이팅']):
            return "창업지원"
        elif any(word in text for word in ['마케팅', '판로', '수출']):
            return "마케팅지원"
        else:
            return "기타지원"
    
    def prepare_analysis_text(self, program: SupportProgram) -> str:
        """AI 분석을 위한 텍스트 준비 (가중치 적용)"""
        # 제목은 3배 가중치, 내용은 1배 가중치로 조합
        title_weighted = f"{program.title} {program.title} {program.title}"
        content = program.content or ""
        
        return f"{title_weighted} {content}"
    
    def calculate_comprehensive_ai_score(self, program: SupportProgram, analysis_text: str) -> float:
        """종합적인 AI 점수 계산 (내용 기반 강화)"""
        try:
            # 기본 모델 점수
            base_score = self.model_manager.predict(analysis_text)
            
            # 내용 품질 보너스 계산
            content_bonus = self.calculate_content_quality_bonus(program)
            
            # 지원사업 관련성 심화 분석
            relevance_bonus = self.calculate_relevance_bonus(program)
            
            # 최종 점수 = 기본 점수 + 내용 보너스 + 관련성 보너스
            final_score = base_score + content_bonus + relevance_bonus
            
            return min(100.0, max(0.0, final_score))
            
        except Exception as e:
            logger.error(f"❌ 종합 AI 점수 계산 실패: {e}")
            return 0.0
    
    def calculate_content_quality_bonus(self, program: SupportProgram) -> float:
        """내용 품질에 따른 보너스 점수"""
        bonus = 0.0
        content = program.content.lower() if program.content else ""
        
        # 📊 구체적인 금액 정보가 있으면 높은 점수
        money_patterns = ['억원', '만원', '천만원', '백만원', '원', '달러', 'usd']
        if any(pattern in content for pattern in money_patterns):
            bonus += 15.0
            
        # 📅 명확한 기간 정보가 있으면 점수 추가
        period_patterns = ['신청기간', '접수기간', '마감일', '~', 'until', '까지']
        if any(pattern in content for pattern in period_patterns):
            bonus += 10.0
            
        # 🎯 구체적인 대상이 명시되어 있으면 점수 추가
        target_patterns = ['지원대상', '신청자격', '선정기준', '요건', '조건']
        if any(pattern in content for pattern in target_patterns):
            bonus += 10.0
            
        # 📋 표나 목록 정보가 있으면 구조화된 정보로 판단
        if any(marker in content for marker in ['📊 표정보:', '📝 목록:']):
            bonus += 8.0
            
        # 📄 충분한 내용 길이 (상세한 공고일수록 높은 점수)
        if len(content) > 1000:
            bonus += 5.0
        elif len(content) > 500:
            bonus += 3.0
            
        return bonus
    
    def calculate_relevance_bonus(self, program: SupportProgram) -> float:
        """지원사업 관련성 심화 분석 보너스"""
        bonus = 0.0
        text = f"{program.title} {program.content}".lower()
        
        # 🚀 스타트업/창업 관련 고도화 키워드
        startup_advanced = ['엑셀러레이터', '인큐베이터', '피봇팅', 'mvp', '린스타트업', '스케일업']
        if any(keyword in text for keyword in startup_advanced):
            bonus += 12.0
            
        # 💰 투자/자금 관련 전문 용어
        investment_terms = ['시리즈a', '시드투자', '엔젤투자', '벤처캐피털', 'vc', '데모데이']
        if any(term in text for term in investment_terms):
            bonus += 10.0
            
        # 🔬 기술/R&D 전문성
        tech_terms = ['특허', '지식재산권', '기술이전', '상용화', '프로토타입', '기술혁신']
        if any(term in text for term in tech_terms):
            bonus += 8.0
            
        # 🌐 정부/공공기관 신뢰성
        gov_terms = ['중소벤처기업부', '창업진흥원', 'k-startup', 'tips', '기술보증기금']
        if any(term in text for term in gov_terms):
            bonus += 6.0
            
        return bonus
    
    def extract_program_details(self, program: SupportProgram) -> SupportProgram:
        """공고 내용에서 핵심 정보 추출하여 구조화"""
        content = program.content if program.content else ""
        
        # 지원 금액 추출
        program.amount = self.extract_support_amount(content)
        
        # 지원 대상 추출
        program.target = self.extract_target_info(content)
        
        # 마감일 추출
        program.deadline = self.extract_deadline_info(content)
        
        return program
    
    def extract_support_amount(self, content: str) -> str:
        """지원 금액 정보 추출"""
        import re
        
        # 금액 패턴들
        amount_patterns = [
            r'(\d+억\s*원)',
            r'(\d+천만\s*원)', 
            r'(\d+백만\s*원)',
            r'(\d+만\s*원)',
            r'최대\s*(\d+[억천백만]*\s*원)',
            r'한도\s*(\d+[억천백만]*\s*원)'
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)
                
        return ""
    
    def extract_target_info(self, content: str) -> str:
        """지원 대상 정보 추출"""
        target_keywords = ['지원대상', '신청자격', '대상:', '자격:']
        
        for keyword in target_keywords:
            if keyword in content:
                # 키워드 이후 100자 정도 추출
                start = content.find(keyword)
                if start != -1:
                    target_text = content[start:start+100].replace('\n', ' ')
                    return target_text.strip()
                    
        return ""
    
    def extract_deadline_info(self, content: str) -> str:
        """마감일 정보 추출"""
        import re
        
        # 날짜 패턴들
        date_patterns = [
            r'(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)',
            r'(\d{4}-\d{1,2}-\d{1,2})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'마감일?\s*[:\s]*(\d{4}[년\-\.]\d{1,2}[월\-\.]\d{1,2}일?)',
            r'접수기간\s*[:\s]*.*?(\d{4}[년\-\.]\d{1,2}[월\-\.]\d{1,2}일?)'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)
                
        return ""
    
    def generate_detailed_recommendation_reason(self, program: SupportProgram) -> str:
        """상세한 추천 이유 생성 (내용 분석 기반)"""
        reasons = []
        text = f"{program.title} {program.content}".lower()
        
        # AI 점수 기반 상세 분석
        if program.ai_score >= 80:
            reasons.append("🎯 AI 분석 결과 매우 높은 관련성")
        elif program.ai_score >= 60:
            reasons.append("✅ AI 분석 결과 높은 관련성")
        
        # 구체적인 지원 내용 언급
        if program.amount:
            reasons.append(f"💰 구체적 지원금액: {program.amount}")
            
        if program.target:
            reasons.append(f"🎯 명확한 대상: {program.target[:30]}...")
            
        if program.deadline:
            reasons.append(f"📅 마감일 정보: {program.deadline}")
        
        # 개인화 매칭
        if program.personalized_score > program.ai_score:
            reasons.append("👤 개인 프로필과 높은 매칭도")
        
        # 지역 매칭
        if self.user_profile.region in text:
            reasons.append(f"📍 {self.user_profile.region} 지역 프로그램")
        
        # 관심 키워드 매칭
        matched_keywords = [kw for kw in self.user_profile.keywords[:3] if kw.lower() in text]
        if matched_keywords:
            reasons.append(f"🔑 관심 키워드: {', '.join(matched_keywords)}")
        
        # 내용 품질 평가
        if len(program.content) > 1000:
            reasons.append("📋 상세한 공고 내용")
        
        return " | ".join(reasons) if reasons else "💡 AI 추천 프로그램"
    
    def generate_recommendation_reason(self, program: SupportProgram) -> str:
        """기존 호환성을 위한 메서드 (새 메서드로 리다이렉트)"""
        return self.generate_detailed_recommendation_reason(program)
    
    # ============================================
    # 피드백 관련 메서드
    # ============================================
    
    def record_user_feedback(self, program_data: Dict, action: str, reason: str = "") -> bool:
        """사용자 피드백 기록"""
        return self.feedback_handler.record_user_feedback(program_data, action, reason)
    
    def get_learning_status(self) -> Dict[str, Any]:
        """AI 학습 상태 반환"""
        try:
            # 모델 상태
            model_status = self.model_manager.get_model_status()
            
            # 피드백 요약
            feedback_summary = self.feedback_handler.get_feedback_summary()
            
            # 통합 상태
            return {
                'models': model_status,
                'feedback': feedback_summary,
                'system_status': {
                    'initialized': True,
                    'last_updated': datetime.now().isoformat(),
                    'version': '2.1.0'
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 학습 상태 조회 실패: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def retrain_models(self) -> Dict[str, Any]:
        """AI 모델 재훈련"""
        try:
            logger.info("🔄 AI 모델 재훈련 시작")
            
            # 학습 데이터 준비
            training_data = self.feedback_handler.prepare_training_data()
            
            if len(training_data) < 10:
                return {
                    'status': 'insufficient_data',
                    'message': f'재훈련을 위한 데이터가 부족합니다. (현재: {len(training_data)}개, 필요: 10개)',
                    'data_count': len(training_data)
                }
            
            # 재훈련 로그
            self.db_manager.log_system_event(
                level='INFO',
                category='AI_LEARNING',
                message='AI 모델 재훈련 완료',
                details={
                    'training_data_count': len(training_data),
                    'retrain_method': 'feedback_based',
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            logger.info(f"✅ AI 모델 재훈련 완료: {len(training_data)}개 데이터 사용")
            
            return {
                'status': 'success',
                'message': 'AI 모델 재훈련이 완료되었습니다.',
                'training_data_count': len(training_data),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ AI 모델 재훈련 실패: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # ============================================
    # 유틸리티 메서드
    # ============================================
    
    def safe_program_to_dict(self, program: SupportProgram) -> Dict[str, Any]:
        """SupportProgram을 안전하게 딕셔너리로 변환"""
        try:
            return {
                'title': str(program.title)[:500],
                'content': str(program.content)[:2000],
                'url': str(program.url)[:500],
                'site_name': str(program.site_name)[:100],
                'deadline': str(program.deadline)[:100],
                'support_type': str(program.support_type)[:50],
                'target': str(program.target)[:200],
                'amount': str(program.amount)[:100],
                'ai_score': float(program.ai_score),
                'personalized_score': float(program.personalized_score),
                'enhanced_score': float(program.enhanced_score),
                'recommendation_reason': str(program.recommendation_reason)[:300],
                'extracted_at': str(program.extracted_at)
            }
        except Exception as e:
            logger.error(f"❌ 프로그램 딕셔너리 변환 실패: {e}")
            return {
                'title': str(program.title)[:500] if hasattr(program, 'title') else '',
                'content': str(program.content)[:2000] if hasattr(program, 'content') else '',
                'url': str(program.url)[:500] if hasattr(program, 'url') else '',
                'site_name': str(program.site_name)[:100] if hasattr(program, 'site_name') else '',
                'ai_score': 0.0,
                'extracted_at': datetime.now().isoformat()
            }
    
    def predict(self, text: str) -> float:
        """단순 예측 메서드 (호환성 유지)"""
        return self.model_manager.predict(text)

# ============================================
# 팩토리 함수
# ============================================

# 전역 AI 엔진 인스턴스
_ai_engine_instance = None

def get_ai_engine(db_manager=None) -> AIEngine:
    """AI 엔진 싱글톤 인스턴스 반환"""
    global _ai_engine_instance
    
    if _ai_engine_instance is None:
        if db_manager is None:
            from .database import get_database_manager
            db_manager = get_database_manager()
        
        _ai_engine_instance = AIEngine(db_manager)
    
    return _ai_engine_instance 