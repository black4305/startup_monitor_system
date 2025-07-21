"""
웹 라우팅 계층 - Flask 라우트 분리
"""

from flask import Blueprint, render_template, request, jsonify, send_from_directory
import logging
import threading
import traceback
from datetime import datetime
from .api_utils import (
    success_response, error_response, paginated_response,
    not_found_response, validation_error_response, internal_server_error_response
)

logger = logging.getLogger(__name__)

# 크롤링 진행 상태 전역 변수
search_progress = {
    'in_progress': False,
    'progress_percentage': 0,
    'completed_sites': 0,
    'total_sites': 0,
    'current_site': '',
    'message': '검색이 시작되지 않았습니다.'
}


def create_routes(program_service, dashboard_service, ai_service):
    """라우트 생성 함수"""
    
    bp = Blueprint('main', __name__)
    
    @bp.route('/')
    def dashboard():
        """메인 대시보드"""
        try:
            data = dashboard_service.get_dashboard_data(limit=10)
            
            # 링크 상태 비동기 체크 시작
            program_service.check_links_async(data['programs'])
            
            return render_template('clean_dashboard.html', 
                                 programs=data['programs'], 
                                 stats=data['stats'])
        except Exception as e:
            logger.error(f"❌ 대시보드 로드 실패: {e}")
            logger.error(traceback.format_exc())
            return render_template('error.html', error=str(e)), 500
    
    @bp.route('/programs')
    def programs_list():
        """프로그램 목록 페이지"""
        try:
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 50))
            
            # 페이지네이션된 데이터 조회
            result = program_service.get_programs_paginated(page=page, per_page=per_page)
            programs = result['programs']
            pagination = result['pagination']
            
            # 링크 상태 비동기 체크
            program_service.check_links_async(programs)
            
            logger.info(f"📄 프로그램 목록 페이지 {page}: {len(programs)}개/{pagination['total_count']}개 표시")
            
            return render_template('clean_programs.html',
                                 programs=programs,
                                 current_page=pagination['current_page'],
                                 total_pages=pagination['total_pages'],
                                 per_page=pagination['per_page'],
                                 total_programs=pagination['total_count'])
                                 
        except Exception as e:
            logger.error(f"❌ 프로그램 목록 로드 실패: {e}")
            return render_template('error.html', error=str(e)), 500
    
    @bp.route('/api/data_status')
    def data_status():
        """데이터 상태 API"""
        try:
            programs = program_service.get_programs()
            return success_response(data={
                'total_programs': len(programs),
                'last_updated': datetime.now().isoformat(),
                'cache_valid': program_service._is_cache_valid()
            })
        except Exception as e:
            return internal_server_error_response(message=str(e))
    
    @bp.route('/api/refresh_data', methods=['POST'])
    def refresh_data():
        """데이터 새로고침 API"""
        try:
            programs = program_service.refresh_cache()
            return success_response(
                data={'total_programs': len(programs)},
                message=f'데이터 새로고침 완료: {len(programs)}개'
            )
        except Exception as e:
            return internal_server_error_response(message=str(e))
    
    @bp.route('/api/learning_status')
    def learning_status():
        """AI 학습 상태 API"""
        try:
            status = ai_service.get_learning_status()
            return success_response(data=status)
        except Exception as e:
            return internal_server_error_response(message=str(e))
    
    @bp.route('/api/retrain_ai', methods=['POST'])
    def retrain_ai():
        """AI 재학습 API"""
        try:
            def run_retrain():
                try:
                    result = ai_service.retrain_model()
                    logger.info(f"✅ AI 재학습 완료: {result}")
                except Exception as e:
                    logger.error(f"❌ AI 재학습 실패: {e}")
            
            # 백그라운드에서 실행
            thread = threading.Thread(target=run_retrain)
            thread.daemon = True
            thread.start()
            
            return success_response(
                data={'status': 'started'},
                message='AI 재학습이 백그라운드에서 시작되었습니다.'
            )
        except Exception as e:
            return internal_server_error_response(message=str(e))
    
    @bp.route('/api/start_search', methods=['POST'])
    def start_search():
        """지원사업 크롤링 및 검색 시작 API"""
        try:
            # 전역 진행 상태 변수 초기화
            global search_progress
            search_progress.update({
                'in_progress': False,
                'progress_percentage': 0,
                'completed_sites': 0,
                'total_sites': 0,
                'current_site': '',
                'message': '검색 준비 중...'
            })
            
            def update_progress(completed, total, current_site="", message=""):
                """진행 상태 업데이트 콜백"""
                global search_progress
                search_progress['completed_sites'] = completed
                search_progress['total_sites'] = total
                search_progress['current_site'] = current_site
                search_progress['message'] = message
                if total > 0:
                    search_progress['progress_percentage'] = (completed / total) * 100
            
            def run_crawling():
                global search_progress
                try:
                    search_progress['in_progress'] = True
                    search_progress['message'] = '크롤링 시작...'
                    
                    logger.info("🔍 지원사업 크롤링 시작...")
                    # 진행 상태 콜백과 함께 크롤링 시작
                    result = ai_service.start_crawling(progress_callback=update_progress)
                    logger.info(f"✅ 크롤링 완료: {result}")
                    
                    # 완료 상태 업데이트
                    search_progress['progress_percentage'] = 100
                    search_progress['message'] = '검색 완료!'
                    search_progress['in_progress'] = False
                    
                except Exception as e:
                    logger.error(f"❌ 크롤링 실패: {e}")
                    search_progress['message'] = f'크롤링 실패: {str(e)}'
                    search_progress['in_progress'] = False
            
            # 백그라운드에서 실행
            thread = threading.Thread(target=run_crawling)
            thread.daemon = True
            thread.start()
            
            return success_response(
                message='지원사업 검색이 백그라운드에서 시작되었습니다.'
            )
        except Exception as e:
            logger.error(f"크롤링 시작 실패: {e}")
            return internal_server_error_response(message=str(e))
    
    @bp.route('/api/search_progress')
    def search_progress():
        """크롤링 진행 상태 API"""
        try:
            # 전역 진행 상태 변수가 없다면 초기화
            if 'search_progress' not in globals():
                global search_progress
                search_progress = {
                    'in_progress': False,
                    'progress_percentage': 0,
                    'completed_sites': 0,
                    'total_sites': 0,
                    'current_site': '',
                    'message': '검색이 시작되지 않았습니다.'
                }
            
            return success_response(data={'progress': search_progress})
        except Exception as e:
            logger.error(f"진행 상태 확인 실패: {e}")
            return internal_server_error_response(message=str(e))
    
    @bp.route('/api/delete/<program_id>', methods=['POST'])
    def delete_program(program_id):
        """개별 프로그램 삭제 API - 실제 삭제 + 강화학습 데이터 저장"""
        try:
            from core.database import get_database_manager
            db_manager = get_database_manager()
            
            # 삭제 전 프로그램 정보 조회 (강화학습용 데이터 백업)
            program = db_manager.get_program_by_external_id(program_id)
            
            if not program:
                return not_found_response('프로그램')
            
            # 삭제 이유 가져오기
            reason = request.json.get('reason', '') if request.json else ''
            
            # 1단계: 강화학습용 피드백 데이터 저장 (삭제 전에 먼저!)
            feedback_success = db_manager.insert_user_feedback(
                program_external_id=program_id,
                action='delete',
                reason=reason,
                confidence=program.get('ai_score', 0) / 100.0
            )
            
            if feedback_success:
                logger.info(f"✅ 강화학습 데이터 저장 완료: {program.get('title', '')[:30]}...")
                
                # AI 엔진에 피드백 전달 (패턴 학습)
                ai_service.ai_engine.record_user_feedback(program, 'delete', reason)
            else:
                logger.warning("⚠️ 강화학습 데이터 저장 실패")
            
            # 2단계: 논리적 삭제 (is_active = false)
            deletion_success = db_manager.deactivate_program(program_id)
            
            if deletion_success:
                # 캐시 무효화
                program_service.refresh_cache()
                
                logger.info(f"🗑️ 프로그램 완전 삭제: {program.get('title', '')[:30]}...")
                
                return success_response(
                    data={
                        'program_title': program.get('title', ''),
                        'feedback_saved': feedback_success
                    },
                    message='프로그램이 삭제되고 AI 학습용 데이터가 저장되었습니다.'
                )
            else:
                return internal_server_error_response(message='프로그램 삭제 실패')
                
        except Exception as e:
            logger.error(f"❌ 프로그램 삭제 실패: {e}")
            return internal_server_error_response(message=str(e))
    
    @bp.route('/api/bulk-delete', methods=['POST'])
    def bulk_delete():
        """일괄 삭제 API - 실제 삭제 + 강화학습 데이터 저장"""
        try:
            data = request.get_json()
            program_ids = data.get('program_ids', [])
            reason = data.get('reason', '일괄삭제')
            
            if not program_ids:
                return validation_error_response({'program_ids': '삭제할 프로그램이 선택되지 않았습니다.'})
            
            from core.database import get_database_manager
            db_manager = get_database_manager()
            
            deleted_count = 0
            failed_count = 0
            feedback_saved_count = 0
            
            for program_id in program_ids:
                try:
                    # 프로그램 정보 조회
                    program = db_manager.get_program_by_external_id(program_id)
                    
                    if program:
                        # 1단계: 강화학습용 피드백 데이터 저장
                        feedback_success = db_manager.insert_user_feedback(
                            program_external_id=program_id,
                            action='delete',
                            reason=reason,
                            confidence=program.get('ai_score', 0) / 100.0
                        )
                        
                        if feedback_success:
                            feedback_saved_count += 1
                            # AI 엔진에 피드백 전달
                            ai_service.ai_engine.record_user_feedback(program, 'delete', reason)
                        
                        # 2단계: 논리적 삭제
                        if db_manager.deactivate_program(program_id):
                            deleted_count += 1
                        else:
                            failed_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"개별 삭제 실패 {program_id}: {e}")
                    failed_count += 1
            
            # 캐시 새로고침
            program_service.refresh_cache()
            
            logger.info(f"🗑️ 일괄 삭제 완료: {deleted_count}개 삭제, {feedback_saved_count}개 학습데이터 저장, {failed_count}개 실패")
            
            return success_response(
                data={
                    'deleted_count': deleted_count,
                    'feedback_saved_count': feedback_saved_count,
                    'failed_count': failed_count
                },
                message=f'{deleted_count}개 프로그램이 삭제되고 {feedback_saved_count}개 학습 데이터가 저장되었습니다.'
            )
            
        except Exception as e:
            logger.error(f"❌ 일괄 삭제 실패: {e}")
            return internal_server_error_response(message=str(e))
    
    @bp.route('/api/feedback/<program_id>', methods=['POST'])
    def record_feedback(program_id):
        """사용자 피드백 기록 API (강화학습용)"""
        try:
            data = request.get_json()
            action = data.get('action')  # 'like', 'dislike', 'keep', 'delete'
            reason = data.get('reason', '')
            confidence = data.get('confidence', 0.8)
            
            if not action:
                return validation_error_response({'action': '액션이 필요합니다.'})
            
            from core.database import get_database_manager
            db_manager = get_database_manager()
            
            # 피드백 기록
            success = db_manager.insert_user_feedback(program_id, action, reason, confidence)
            
            if success:
                logger.info(f"✅ 피드백 기록: {program_id} - {action}")
                
                # AI 엔진에 피드백 전달 (패턴 학습)
                program = db_manager.get_program_by_external_id(program_id)
                if program:
                    ai_service.ai_engine.record_user_feedback(program, action, reason)
                
                return success_response(message='피드백이 기록되었습니다.')
            else:
                return internal_server_error_response(message='피드백 기록 실패')
                
        except Exception as e:
            logger.error(f"❌ 피드백 기록 실패: {e}")
            return internal_server_error_response(message=str(e))
    
    @bp.route('/api/stats')
    def dashboard_stats():
        """대시보드 통계 API"""
        try:
            dashboard_data = dashboard_service.get_dashboard_data(limit=0)
            stats = dashboard_data.get('stats', {})
            return success_response(data=stats)
            
        except Exception as e:
            logger.error(f"❌ 통계 조회 실패: {e}")
            return internal_server_error_response(message=str(e))
    
    @bp.route('/api/feedback_stats')
    def feedback_stats():
        """피드백 통계 API"""
        try:
            from core.database import get_database_manager
            db_manager = get_database_manager()
            
            stats = db_manager.get_user_feedback_stats()
            recent_feedback = db_manager.get_recent_feedback(limit=20)
            
            return success_response(data={
                'stats': stats,
                'recent_feedback': recent_feedback
            })
            
        except Exception as e:
            logger.error(f"❌ 피드백 통계 조회 실패: {e}")
            return internal_server_error_response(message=str(e))
    
    @bp.route('/api/reinforcement_learning/status')
    def rl_status():
        """강화학습 상태 조회 API"""
        try:
            status = ai_service.ai_engine.feedback_handler.get_rl_status()
            return success_response(data={'reinforcement_learning': status})
            
        except Exception as e:
            logger.error(f"❌ 강화학습 상태 조회 실패: {e}")
            return internal_server_error_response(message=str(e))
    
    @bp.route('/api/auto_delete_status')
    def auto_delete_status():
        """자동 삭제 진행 상황 API"""
        try:
            status = ai_service.ai_engine.feedback_handler.get_auto_delete_status()
            
            return success_response(data={
                'status': status,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ 자동 삭제 상태 조회 실패: {e}")
            return internal_server_error_response(message=str(e))
    
    @bp.route('/api/reinforcement_learning/optimize', methods=['POST'])
    def force_rl_optimization():
        """강화학습 강제 실행"""
        try:
            def run_optimization():
                try:
                    logger.info("🚀 수동 강화학습 최적화 시작")
                    result = ai_service.ai_engine.feedback_handler.force_reinforcement_learning()
                    logger.info(f"✅ 강화학습 결과: {result.get('status', 'unknown')}")
                    
                    if result.get('status') == 'success':
                        logger.info(f"🎉 딥러닝 모델 성능 향상: {result.get('improvement', 0):+.3f}")
                        logger.info(f"📚 훈련 샘플: {result.get('training_samples', 0)}개")
                        logger.info(f"🔄 에포크: {result.get('epochs_trained', 0)}")
                    
                except Exception as e:
                    logger.error(f"❌ 강화학습 최적화 실행 중 오류: {e}")
            
            # 백그라운드에서 실행
            import threading
            thread = threading.Thread(target=run_optimization, daemon=True)
            thread.start()
            
            return success_response(
                data={
                    'status': 'started',
                    'timestamp': datetime.now().isoformat()
                },
                message='딥러닝 모델 재훈련이 백그라운드에서 시작되었습니다. 진행 상황은 로그를 확인하세요.'
            )
            
        except Exception as e:
            logger.error(f"❌ 강화학습 트리거 실패: {e}")
            return internal_server_error_response(message=str(e))
    
    @bp.route('/api/ai_learning/detailed_status')
    def detailed_learning_status():
        """AI 학습 상세 상태 API"""
        try:
            # 피드백 요약
            feedback_summary = ai_service.ai_engine.feedback_handler.get_feedback_summary()
            
            # 모델 성능 상태
            model_status = ai_service.ai_engine.model_manager.get_model_status()
            
            # 통합 응답
            return success_response(data={
                'ai_learning_status': {
                    'feedback_system': feedback_summary,
                    'model_performance': model_status,
                    'system_health': {
                        'initialized': True,
                        'last_updated': datetime.now().isoformat()
                    }
                }
            })
            
        except Exception as e:
            logger.error(f"❌ AI 학습 상세 상태 조회 실패: {e}")
            return internal_server_error_response(message=str(e))
    
    @bp.route('/api/debug_sites')
    def debug_sites():
        """크롤링 사이트 디버그 정보 - DB에서 가져온 사이트 목록 확인"""
        try:
            from core.database import get_database_manager
            db_manager = get_database_manager()
            
            # 크롤링 사이트 정보 가져오기
            sites = db_manager.get_crawling_sites(enabled_only=False)
            enabled_sites = db_manager.get_crawling_sites(enabled_only=True)
            
            return success_response(data={
                'total_sites': len(sites),
                'enabled_sites_count': len(enabled_sites),
                'sites_detail': [
                    {
                        'id': s.get('id'),
                        'name': s.get('name'),
                        'url': s.get('url'),
                        'enabled': s.get('enabled'),
                        'priority': s.get('priority'),
                        'category': s.get('category'),
                        'region': s.get('region')
                    } for s in sites
                ],
                'enabled_sites_detail': [
                    {
                        'id': s.get('id'), 
                        'name': s.get('name'),
                        'url': s.get('url'),
                        'priority': s.get('priority')
                    } for s in enabled_sites
                ]
            })
        except Exception as e:
            logger.error(f"사이트 디버그 실패: {e}")
            return internal_server_error_response(message=str(e))
    
    @bp.route('/static/<path:filename>')
    def static_files(filename):
        """정적 파일 서빙"""
        return send_from_directory('static', filename)
    
    @bp.errorhandler(404)
    def not_found_error(error):
        return render_template('error.html', error="페이지를 찾을 수 없습니다."), 404
    
    @bp.errorhandler(500)
    def internal_error(error):
        return render_template('error.html', error="내부 서버 오류가 발생했습니다."), 500
    
    return bp 