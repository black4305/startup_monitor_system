"""
🏗️ 핵심 시스템 모듈 (리팩토링 버전)
AI 지원사업 모니터링 시스템의 핵심 기능들
"""

import logging

# 핵심 모듈들
from .database import DatabaseManager
from .ai_engine import AIEngine, UserProfile, SupportProgram
from .config import Config

# 분리된 모듈들
from .ai_models import AIModelManager
from .crawler import WebCrawler
from .feedback_handler import FeedbackHandler

# 선택적 import 모듈들
logger = logging.getLogger(__name__)

# 딥러닝 모듈
try:
    from .deep_learning_engine import DeepLearningEngine
    DEEP_LEARNING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"딥러닝 모듈 import 실패: {e}")
    DeepLearningEngine = None
    DEEP_LEARNING_AVAILABLE = False

# 강화학습 모듈 (선택적 import)
try:
    from .reinforcement_learning_optimizer import ReinforcementLearningOptimizer
    RL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"강화학습 모듈 import 실패: {e}")
    logger.info("강화학습 기능을 사용하려면 다음을 설치하세요: pip install gymnasium stable-baselines3 optuna")
    ReinforcementLearningOptimizer = None
    RL_AVAILABLE = False

# 기본 exports
__all__ = [
    'DatabaseManager', 
    'AIEngine', 
    'Config',
    'UserProfile',
    'SupportProgram',
    'AIModelManager',
    'WebCrawler', 
    'FeedbackHandler'
]

# 선택적 모듈들 추가
if DEEP_LEARNING_AVAILABLE:
    __all__.append('DeepLearningEngine')

if RL_AVAILABLE:
    __all__.append('ReinforcementLearningOptimizer')

__version__ = '2.1.0'

# 모듈 상태 정보
def get_module_status():
    """모듈 로딩 상태 반환"""
    return {
        'version': __version__,
        'core_modules': ['DatabaseManager', 'AIEngine', 'Config'],
        'separated_modules': ['AIModelManager', 'WebCrawler', 'FeedbackHandler'],
        'optional_modules': {
            'deep_learning': DEEP_LEARNING_AVAILABLE,
            'reinforcement_learning': RL_AVAILABLE
        },
        'total_modules': len(__all__)
    } 