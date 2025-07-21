#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 스타트업 모니터링 시스템 - 강화학습 최적화
====================================================

코랩에서 훈련된 모델을 로드하여 강화학습으로 성능 최적화

주요 최적화 목표:
1. 임계값 동적 조정
2. 모델 가중치 최적화  
3. 능동학습 전략
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# 로깅 설정
from .logger import get_logger

# 강화학습 라이브러리 (수정된 import)
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False

try:
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# 머신러닝 라이브러리
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# 로컬 모듈 import
from .config import Config
from .database import get_database_manager


if GYM_AVAILABLE:
    BaseEnv = gym.Env
else:
    # gym이 없는 경우 더미 클래스
    class BaseEnv:
        def __init__(self): pass
        def reset(self, **kwargs): return np.array([0.0]), {}
        def step(self, action): return np.array([0.0]), 0.0, True, False, {}

class StartupClassifierEnv(BaseEnv):
    """스타트업 분류기 강화학습 환경"""
    
    def __init__(self, model, test_texts, test_labels, mode='threshold'):
        super().__init__()
        
        self.model = model
        self.test_texts = test_texts
        self.test_labels = test_labels
        self.mode = mode  # 'threshold' or 'weights'
        
        # 현재 에피소드 상태
        self.current_step = 0
        self.max_steps = 100
        
        if GYM_AVAILABLE:
            if mode == 'threshold':
                # 임계값 최적화 모드 (0.1 ~ 0.9)
                self.action_space = spaces.Box(low=0.1, high=0.9, shape=(1,), dtype=np.float32)
                self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
                
            elif mode == 'weights':
                # 모델 가중치 최적화 모드 (7개 모델)
                num_models = len(self.model.model_weights) if hasattr(self.model, 'model_weights') else 3
                self.action_space = spaces.Box(low=0.0, high=1.0, shape=(num_models,), dtype=np.float32)
                self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        else:
            # gym이 없는 경우 더미 설정
            self.action_space = None
            self.observation_space = None
        
        # 성과 기록
        self.best_score = 0
        self.performance_history = []
        
    def reset(self, seed=None, options=None):
        """환경 초기화 (Gymnasium 호환)"""
        super().reset(seed=seed)
        self.current_step = 0
        
        if self.mode == 'threshold':
            # 현재 임계값, 정확도, 정밀도, 재현율, F1
            obs = np.array([0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            # 현재 가중치들 + 성과 지표들
            weights = list(getattr(self.model, 'model_weights', {0: 1.0, 1: 1.0, 2: 1.0}).values())
            metrics = [0.0, 0.0, 0.0, 0.0]  # accuracy, precision, recall, f1
            obs = np.array(weights + metrics, dtype=np.float32)
        
        # Gymnasium 호환을 위해 (observation, info) 튜플 반환
        info = {
            'step': self.current_step,
            'mode': self.mode,
            'best_score': self.best_score
        }
        return obs, info
    
    def step(self, action):
        """한 스텝 실행"""
        self.current_step += 1
        
        # 액션 적용
        if self.mode == 'threshold':
            new_threshold = float(action[0])
            if hasattr(self.model, 'threshold'):
                self.model.threshold = new_threshold
            
        elif self.mode == 'weights':
            # 가중치 정규화
            normalized_weights = action / np.sum(action)
            if hasattr(self.model, 'model_weights'):
                model_names = list(self.model.model_weights.keys())
                for i, name in enumerate(model_names):
                    if i < len(normalized_weights):
                        self.model.model_weights[name] = float(normalized_weights[i])
        
        # 성과 평가
        try:
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(self.test_texts)
            else:
                # 기본 예측 (모델이 predict 메서드가 없는 경우)
                predictions = [1] * len(self.test_texts)
            
            accuracy = accuracy_score(self.test_labels, predictions)
            precision = precision_score(self.test_labels, predictions, zero_division=0)
            recall = recall_score(self.test_labels, predictions, zero_division=0)
            f1 = f1_score(self.test_labels, predictions, zero_division=0)
            
            # 복합 보상 함수
            reward = self._calculate_reward(accuracy, precision, recall, f1)
            
            # 상태 업데이트
            if self.mode == 'threshold':
                threshold = getattr(self.model, 'threshold', 0.5)
                next_state = np.array([threshold, accuracy, precision, recall, f1], dtype=np.float32)
            else:
                weights = list(getattr(self.model, 'model_weights', {0: 1.0, 1: 1.0, 2: 1.0}).values())
                next_state = np.array(weights + [accuracy, precision, recall, f1], dtype=np.float32)
            
            # 에피소드 종료 조건
            done = self.current_step >= self.max_steps
            truncated = False
            
            # 성과 기록
            self.performance_history.append({
                'step': self.current_step,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'reward': reward
            })
            
            if accuracy > self.best_score:
                self.best_score = accuracy
            
        except Exception as e:
            self.logger.error(f"⚠️ 평가 중 오류: {e}")
            reward = -1.0
            next_state, _ = self.reset()
            done = True
            truncated = False
        
        return next_state, reward, done, truncated, {}
    
    def _calculate_reward(self, accuracy, precision, recall, f1):
        """보상 함수 계산"""
        # 지원사업 탐지를 위해 재현율에 높은 가중치
        base_reward = 0.4 * accuracy + 0.2 * precision + 0.3 * recall + 0.1 * f1
        
        # 보너스: 높은 성과에 대한 추가 보상
        if accuracy > 0.9:
            base_reward += 0.1
        if recall > 0.85:  # 지원사업 놓치지 않기
            base_reward += 0.1
        if precision > 0.8:  # 오탐지 최소화
            base_reward += 0.05
            
        return base_reward


if GYM_AVAILABLE:
    BaseWrapper = gym.Wrapper
else:
    # gym이 없는 경우 더미 클래스
    class BaseWrapper:
        def __init__(self, env): self.env = env
        def reset(self, **kwargs): return np.array([0.0]), {}
        def step(self, action): return np.array([0.0]), 0.0, True, False, {}

class StableBaselinesEnvWrapper(BaseWrapper):
    """Stable-baselines3와 Gymnasium 호환성을 위한 래퍼"""
    
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self, **kwargs):
        """reset에서 observation만 반환 (stable-baselines3용)"""
        obs, _ = self.env.reset(**kwargs)
        return obs
    
    def step(self, action):
        """step은 그대로 유지"""
        return self.env.step(action)


class ReinforcementLearningOptimizer:
    """강화학습 기반 모델 최적화"""
    
    def __init__(self, model_path: str = None, test_data_path: str = None, ai_engine=None):
        """초기화"""
        self.logger = get_logger(__name__)
        self.logger.info("🚀 강화학습 최적화 시스템 초기화!")
        self.logger.info("="*50)
        
        # 의존성 체크
        if not GYM_AVAILABLE:
            raise ImportError("gymnasium 또는 gym이 설치되지 않았습니다: pip install gymnasium")
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3가 설치되지 않았습니다: pip install stable-baselines3")
        if not OPTUNA_AVAILABLE:
            self.logger.warning("⚠️ optuna가 설치되지 않아 하이퍼파라미터 최적화를 사용할 수 없습니다")
        
        # 모델 및 데이터 로드
        if ai_engine:
            # AI 엔진으로부터 모델 사용
            self.model = ai_engine
            self.test_texts, self.test_labels = self._get_test_data_from_ai_engine(ai_engine)
            self.logger.info("✅ AI 엔진으로부터 모델 및 데이터 로드 완료")
        else:
            # 파일로부터 로드
            if model_path and os.path.exists(model_path):
                self.logger.info("📦 모델 로딩 중...")
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                
                # 메타데이터 파일 확인 및 로드
                metadata_path = model_path.replace('.pkl', '_metadata.json')
                if os.path.exists(metadata_path):
                    self.logger.info("📋 모델 메타데이터 로딩 중...")
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # 메타데이터 적용
                    if hasattr(self.model, 'threshold'):
                        self.model.threshold = metadata.get('threshold', 0.3)
                        self.logger.info(f"✅ 임계값 설정: {self.model.threshold}")
                    
                    if hasattr(self.model, 'model_weights') and 'model_weights' in metadata:
                        self.model.model_weights = metadata['model_weights']
                        self.logger.info(f"✅ 모델 가중치 설정: {self.model.model_weights}")
                    
                    # 성능 정보 출력
                    if 'performance' in metadata:
                        perf = metadata['performance']
                        self.logger.info(f"📊 모델 성능 - 정확도: {perf.get('accuracy', 0):.1%}, F1: {perf.get('f1', 0):.3f}")
            else:
                self.logger.warning("⚠️ 모델 파일을 찾을 수 없습니다. 기본 모델을 사용합니다.")
                self.model = self._create_dummy_model()
            
            if test_data_path and os.path.exists(test_data_path):
                self.logger.info("📊 테스트 데이터 로딩 중...")
                with open(test_data_path, 'r', encoding='utf-8') as f:
                    test_data = json.load(f)
                self.test_texts = test_data['texts']
                self.test_labels = test_data['labels']
            else:
                self.logger.warning("⚠️ 테스트 데이터를 찾을 수 없습니다. 기본 데이터를 사용합니다.")
                self.test_texts, self.test_labels = self._create_dummy_test_data()
        
        # 추가 테스트 케이스 (문제 케이스들)
        additional_cases = [
            ('벤처투자 펀딩 프로그램 참가자 모집', 1),
            ('2024 K-스타트업 센터 입주기업 모집', 1),
            ('신기술 스타트업 육성사업 공고', 1),
            ('일반 상품 광고 안내', 0),
            ('카페 신메뉴 출시 이벤트', 0),
            ('부동산 투자 상담 서비스', 0),
        ]
        
        for text, label in additional_cases:
            self.test_texts.append(text)
            self.test_labels.append(label)
        
        self.logger.info(f"✅ 총 {len(self.test_texts)}개 테스트 케이스 로드 완료")
        
        # 초기 성과 측정
        self.baseline_performance = self._evaluate_model()
        self.logger.info(f"🎯 베이스라인 성과: {self.baseline_performance['accuracy']:.3f}")
        
        # 데이터베이스 연결
        self.db = get_database_manager()
        
    def _get_test_data_from_ai_engine(self, ai_engine) -> Tuple[List[str], List[int]]:
        """AI 엔진으로부터 실제 사용자 피드백 데이터 수집"""
        try:
            self.logger.info("📊 실제 사용자 피드백 데이터 수집 중...")
            
            # 데이터베이스에서 실제 피드백 데이터 가져오기
            db = get_database_manager()
            feedback_data = db.get_recent_feedback(limit=100)
            
            texts = []
            labels = []
            
            if feedback_data:
                self.logger.info(f"✅ {len(feedback_data)}개의 실제 피드백 발견")
                
                for feedback in feedback_data:
                    try:
                        # 프로그램 데이터 가져오기
                        program = db.get_program_by_external_id(feedback.get('program_external_id', ''))
                        if not program:
                            continue
                            
                        # 텍스트 구성
                        title = program.get('title', '')
                        content = program.get('content', '')
                        text = f"{title} {content}".strip()
                        
                        if not text:
                            continue
                        
                        # 피드백 액션을 라벨로 변환
                        action = feedback.get('action', '')
                        if action in ['keep', 'like', 'interest']:
                            label = 1  # 긍정
                        elif action in ['delete', 'dislike', 'not_interested']:
                            label = 0  # 부정
                        else:
                            continue
                        
                        texts.append(text)
                        labels.append(label)
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️ 피드백 데이터 처리 실패: {e}")
                        continue
                        
                self.logger.info(f"📈 처리된 학습 데이터: {len(texts)}개 (긍정: {sum(labels)}, 부정: {len(labels)-sum(labels)})")
                
            # 실제 피드백이 부족한 경우 최소한의 기본 케이스 추가
            if len(texts) < 10:
                self.logger.warning("⚠️ 실제 피드백 데이터가 부족합니다. 기본 케이스 추가...")
                additional_cases = [
                    ('스타트업 펀딩 지원사업 공모', 1),
                    ('창업기업 육성 프로그램 모집', 1),
                    ('벤처투자 매칭 서비스', 1),
                    ('일반 마케팅 광고', 0),
                    ('부동산 투자 안내', 0),
                    ('온라인 쇼핑몰 오픈', 0),
                ]
                
                for text, label in additional_cases:
                    texts.append(text)
                    labels.append(label)
                    
                self.logger.info(f"📝 기본 케이스 {len(additional_cases)}개 추가")
                
            self.logger.info(f"🎯 최종 학습 데이터: {len(texts)}개")
            return texts, labels
            
        except Exception as e:
            self.logger.error(f"❌ 실제 피드백 데이터 수집 실패: {e}")
            self.logger.info("🔄 기본 테스트 케이스로 폴백...")
            
            # 폴백 케이스
            fallback_cases = [
                ('스타트업 펀딩 지원사업 공모', 1),
                ('창업기업 육성 프로그램 모집', 1),
                ('벤처투자 매칭 서비스', 1),
                ('일반 마케팅 광고', 0),
                ('부동산 투자 안내', 0),
                ('온라인 쇼핑몰 오픈', 0),
            ]
            
            texts = [case[0] for case in fallback_cases]
            labels = [case[1] for case in fallback_cases]
            
            return texts, labels
    
    def _create_dummy_model(self):
        """실제 딥러닝 모델 로드 시도, 실패시 더미 모델"""
        try:
            # 실제 딥러닝 모델 로드 시도
            from .deep_learning_engine import get_deep_learning_engine
            self.logger.info("🧠 딥러닝 모델 로드 시도...")
            
            deep_engine = get_deep_learning_engine()
            if deep_engine and hasattr(deep_engine, 'predict'):
                self.logger.info("✅ 딥러닝 모델 로드 성공!")
                
                # 딥러닝 모델을 강화학습에 맞게 래핑
                class DeepLearningWrapper:
                    def __init__(self, engine):
                        self.engine = engine
                        self.threshold = 0.5
                        self.model_weights = {'deep_learning': 1.0}
                    
                    def predict(self, texts):
                        if isinstance(texts, str):
                            texts = [texts]
                        
                        predictions = []
                        for text in texts:
                            try:
                                score = self.engine.calculate_score(text)
                                predictions.append(1 if score > 50 else 0)
                            except Exception as e:
                                self.logger.warning(f"⚠️ 딥러닝 예측 오류: {e}")
                                predictions.append(0)
                        return predictions
                
                return DeepLearningWrapper(deep_engine)
            else:
                self.logger.warning("⚠️ 딥러닝 엔진이 없거나 예측 메서드가 없음")
                
        except Exception as e:
            self.logger.warning(f"⚠️ 딥러닝 모델 로드 오류: {e}")
            self.logger.info("💡 더미 모델로 폴백...")
            
        # 더미 모델 생성
        class DummyModel:
            def __init__(self):
                self.threshold = 0.5
                self.model_weights = {'dummy': 1.0}
            
            def predict(self, texts):
                if isinstance(texts, str):
                    texts = [texts]
                    
                predictions = []
                for text in texts:
                    if any(keyword in text.lower() for keyword in ['스타트업', '창업', '펀딩', '투자', '지원사업']):
                        predictions.append(1)
                    else:
                        predictions.append(0)
                return predictions
        
        self.logger.info("🔄 더미 모델 생성 완료")
        return DummyModel()
    
    def _create_dummy_test_data(self) -> Tuple[List[str], List[int]]:
        """더미 테스트 데이터 생성"""
        texts = [
            '스타트업 지원사업 공고',
            '창업기업 펀딩 모집',
            '일반 광고 내용',
            '부동산 투자 안내'
        ]
        labels = [1, 1, 0, 0]
        return texts, labels
    
    def _evaluate_model(self) -> Dict[str, float]:
        """모델 성과 평가"""
        try:
            predictions = []
            
            # 각 텍스트를 개별적으로 예측
            for text in self.test_texts:
                try:
                    # 텍스트를 문자열로 변환
                    text_str = str(text) if not isinstance(text, str) else text
                    pred = self.model.predict(text_str)
                    
                    # 예측값이 점수인 경우 0/1로 변환 (50점 이상이면 1)
                    if isinstance(pred, (int, float)):
                        predictions.append(1 if pred > 50 else 0)
                    else:
                        predictions.append(int(pred) if pred else 0)
                except Exception as e:
                    self.logger.warning(f"⚠️ 개별 텍스트 예측 실패: {e}")
                    predictions.append(0)
            
            # 길이 맞추기
            if len(predictions) != len(self.test_labels):
                self.logger.warning(f"⚠️ 예측/라벨 길이 불일치: {len(predictions)} vs {len(self.test_labels)}")
                predictions = predictions[:len(self.test_labels)]
                if len(predictions) < len(self.test_labels):
                    predictions.extend([0] * (len(self.test_labels) - len(predictions)))
            
            accuracy = accuracy_score(self.test_labels, predictions)
            precision = precision_score(self.test_labels, predictions, zero_division=0)
            recall = recall_score(self.test_labels, predictions, zero_division=0)
            f1 = f1_score(self.test_labels, predictions, zero_division=0)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        except Exception as e:
            self.logger.error(f"⚠️ 평가 실패: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def optimize_threshold_rl(self, total_timesteps: int = 10000):
        """강화학습으로 임계값 최적화"""
        self.logger.info("\n🎯 강화학습 임계값 최적화 시작!")
        self.logger.info("-" * 40)
        
        # 환경 생성
        env = StartupClassifierEnv(self.model, self.test_texts, self.test_labels, mode='threshold')
        # Stable-baselines3 호환성을 위한 래퍼 적용
        env = StableBaselinesEnvWrapper(env)
        
        # PPO 에이전트 훈련
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
        
        self.logger.info("🧠 PPO 에이전트 훈련 중...")
        model.learn(total_timesteps=total_timesteps)
        
        # 최적 임계값 찾기
        obs = env.reset()
        best_threshold = 0.5
        best_score = 0
        
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            
            current_performance = self._evaluate_model()
            if current_performance['accuracy'] > best_score:
                best_score = current_performance['accuracy']
                best_threshold = self.model.threshold
        
        # 최적 값 적용
        self.model.threshold = best_threshold
        optimized_performance = self._evaluate_model()
        
        self.logger.info(f"\n✅ 임계값 최적화 완료!")
        self.logger.info(f"   최적 임계값: {best_threshold:.3f}")
        self.logger.info(f"   개선 전: {self.baseline_performance['accuracy']:.3f}")
        self.logger.info(f"   개선 후: {optimized_performance['accuracy']:.3f}")
        self.logger.info(f"   향상도: +{optimized_performance['accuracy'] - self.baseline_performance['accuracy']:.3f}")
        
        return best_threshold, optimized_performance
    
    def optimize_weights_rl(self, total_timesteps: int = 10000):
        """강화학습으로 모델 가중치 최적화"""
        self.logger.info("\n⚖️ 강화학습 가중치 최적화 시작!")
        self.logger.info("-" * 40)
        
        # 환경 생성
        env = StartupClassifierEnv(self.model, self.test_texts, self.test_labels, mode='weights')
        # Stable-baselines3 호환성을 위한 래퍼 적용
        env = StableBaselinesEnvWrapper(env)
        
        # A2C 에이전트 훈련 (연속적 액션에 적합)
        model = A2C("MlpPolicy", env, verbose=1, learning_rate=0.001)
        
        self.logger.info("🧠 A2C 에이전트 훈련 중...")
        model.learn(total_timesteps=total_timesteps)
        
        # 최적 가중치 찾기
        obs = env.reset()
        best_weights = self.model.model_weights.copy()
        best_score = 0
        
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            
            current_performance = self._evaluate_model()
            if current_performance['accuracy'] > best_score:
                best_score = current_performance['accuracy']
                best_weights = self.model.model_weights.copy()
        
        # 최적 가중치 적용
        self.model.model_weights = best_weights
        optimized_performance = self._evaluate_model()
        
        self.logger.info(f"\n✅ 가중치 최적화 완료!")
        self.logger.info(f"   최적 가중치: {best_weights}")
        self.logger.info(f"   개선 전: {self.baseline_performance['accuracy']:.3f}")
        self.logger.info(f"   개선 후: {optimized_performance['accuracy']:.3f}")
        self.logger.info(f"   향상도: +{optimized_performance['accuracy'] - self.baseline_performance['accuracy']:.3f}")
        
        return best_weights, optimized_performance
    
    def hyperparameter_search_optuna(self, n_trials: int = 100):
        """Optuna를 사용한 하이퍼파라미터 최적화"""
        self.logger.info("\n🔍 Optuna 하이퍼파라미터 최적화!")
        self.logger.info("-" * 40)
        
        def objective(trial):
            # 임계값과 가중치 동시 최적화
            threshold = trial.suggest_float('threshold', 0.2, 0.8)
            
            # 각 모델별 가중치
            weights = {}
            model_names = list(self.model.model_weights.keys())
            for name in model_names:
                weights[name] = trial.suggest_float(f'weight_{name}', 0.1, 1.0)
            
            # 임시 적용
            original_threshold = self.model.threshold
            original_weights = self.model.model_weights.copy()
            
            try:
                self.model.threshold = threshold
                self.model.model_weights = weights
                
                performance = self._evaluate_model()
                
                # 복합 점수 (정확도 + 재현율 중시)
                score = 0.6 * performance['accuracy'] + 0.4 * performance['recall']
                
            except Exception as e:
                score = 0.0
            finally:
                # 원래 값 복원
                self.model.threshold = original_threshold
                self.model.model_weights = original_weights
            
            return score
        
        # Optuna 스터디 실행
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # 최적 파라미터 적용
        best_params = study.best_params
        self.model.threshold = best_params['threshold']
        
        for name in self.model.model_weights.keys():
            self.model.model_weights[name] = best_params[f'weight_{name}']
        
        final_performance = self._evaluate_model()
        
        self.logger.info(f"\n✅ Optuna 최적화 완료!")
        self.logger.info(f"   최적 임계값: {best_params['threshold']:.3f}")
        self.logger.info(f"   최적 점수: {study.best_value:.3f}")
        self.logger.info(f"   최종 정확도: {final_performance['accuracy']:.3f}")
        
        return best_params, final_performance
    
    def detailed_analysis(self):
        """상세 분석 및 문제 케이스 재검토"""
        self.logger.info("\n🔍 상세 분석 시작!")
        self.logger.info("="*50)
        
        # 현재 성과
        current_performance = self._evaluate_model()
        predictions = self.model.predict(self.test_texts)
        
        self.logger.info("📊 전체 성과:")
        for metric, value in current_performance.items():
            self.logger.info(f"   {metric}: {value:.3f}")
        
        # 문제 케이스 분석
        problem_cases = []
        for i, (text, true_label, pred) in enumerate(zip(self.test_texts, self.test_labels, predictions)):
            if true_label != pred:
                problem_cases.append({
                    'text': text,
                    'true_label': '지원사업' if true_label == 1 else '일반',
                    'predicted': '지원사업' if pred == 1 else '일반',
                    'type': 'FN' if true_label == 1 and pred == 0 else 'FP'
                })
        
        self.logger.info(f"\n❌ 문제 케이스 {len(problem_cases)}개:")
        for i, case in enumerate(problem_cases):
            self.logger.info(f"   {i+1}. [{case['type']}] {case['text'][:50]}...")
            self.logger.info(f"      실제: {case['true_label']} | 예측: {case['predicted']}")
        
        return current_performance, problem_cases
    
    def save_optimized_model(self, filename: str = 'rl_optimized_model.pkl'):
        """최적화된 모델 저장"""
        self.logger.info(f"\n💾 최적화된 모델 저장: {filename}")
        
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)
        
        # 최적화 결과 메타데이터
        metadata = {
            'optimization_method': 'reinforcement_learning',
            'baseline_accuracy': self.baseline_performance['accuracy'],
            'optimized_accuracy': self._evaluate_model()['accuracy'],
            'threshold': getattr(self.model, 'threshold', 0.5),
            'model_weights': getattr(self.model, 'model_weights', {}),
            'test_cases': len(self.test_texts)
        }
        
        with open(f'rl_optimization_results.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        self.logger.info("✅ 저장 완료!")

    def optimize_from_feedback(self, feedback_data: List[Dict]) -> Dict[str, Any]:
        """피드백 데이터를 기반으로 실시간 최적화"""
        try:
            self.logger.info(f"\n🎯 피드백 기반 강화학습 시작: {len(feedback_data)}개 데이터")
            self.logger.info("-" * 50)
            
            # 피드백 데이터를 학습 데이터로 변환
            texts, labels = self._convert_feedback_to_training_data(feedback_data)
            
            if len(texts) < 3:
                return {
                    'status': 'insufficient_data',
                    'message': '학습을 위한 데이터가 부족합니다.',
                    'data_count': len(texts)
                }
            
            # 기존 테스트 데이터에 피드백 데이터 추가
            combined_texts = self.test_texts + texts
            combined_labels = self.test_labels + labels
            
            # 성능 평가 전 현재 상태 저장
            original_performance = self._evaluate_model()
            
            # 임계값 최적화 (작은 스케일)
            self.logger.info("🎯 피드백 기반 임계값 최적화...")
            optimized_threshold, threshold_performance = self._optimize_threshold_from_feedback(
                combined_texts, combined_labels
            )
            
            # 가중치 최적화 (작은 스케일)
            self.logger.info("⚖️ 피드백 기반 가중치 최적화...")
            optimized_weights, weights_performance = self._optimize_weights_from_feedback(
                combined_texts, combined_labels
            )
            
            # 최적화 결과 저장
            optimization_result = {
                'status': 'success',
                'original_performance': original_performance,
                'threshold_optimization': {
                    'optimal_threshold': optimized_threshold,
                    'performance': threshold_performance
                },
                'weights_optimization': {
                    'optimal_weights': optimized_weights,
                    'performance': weights_performance
                },
                'improvement': {
                    'accuracy_gain': weights_performance['accuracy'] - original_performance['accuracy'],
                    'f1_gain': weights_performance['f1'] - original_performance['f1']
                },
                'feedback_processed': len(feedback_data),
                'training_samples': len(texts),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ 피드백 기반 최적화 완료!")
            self.logger.info(f"   정확도 향상: {optimization_result['improvement']['accuracy_gain']:.3f}")
            self.logger.info(f"   F1 스코어 향상: {optimization_result['improvement']['f1_gain']:.3f}")
            
            # 성능이 향상된 경우에만 적용
            if optimization_result['improvement']['accuracy_gain'] > 0:
                self.model.threshold = optimized_threshold
                if hasattr(self.model, 'model_weights'):
                    self.model.model_weights = optimized_weights
                self.logger.info("🎉 최적화된 설정 적용 완료!")
            else:
                self.logger.warning("⚠️ 성능 향상이 없어 기존 설정 유지")
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"❌ 피드백 기반 최적화 실패: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _convert_feedback_to_training_data(self, feedback_data: List[Dict]) -> Tuple[List[str], List[int]]:
        """피드백 데이터를 학습용 데이터로 변환"""
        texts = []
        labels = []
        
        for feedback in feedback_data:
            try:
                title = feedback.get('title', '')
                content = feedback.get('content', '')
                action = feedback.get('action', '')
                
                # 텍스트 구성
                text = f"{title} {content}".strip()
                if not text:
                    continue
                
                # 라벨 변환 (keep=1, delete=0)
                if action in ['keep', 'like', 'interest']:
                    label = 1
                elif action in ['delete', 'dislike', 'uninterest']:
                    label = 0
                else:
                    continue  # 알 수 없는 액션은 건너뛰기
                
                texts.append(text)
                labels.append(label)
                
            except Exception as e:
                self.logger.warning(f"⚠️ 피드백 데이터 변환 실패: {e}")
                continue
        
        self.logger.info(f"📊 피드백 데이터 변환 완료: {len(texts)}개 (Keep: {sum(labels)}, Delete: {len(labels)-sum(labels)})")
        return texts, labels
    
    def _optimize_threshold_from_feedback(self, texts: List[str], labels: List[int]) -> Tuple[float, Dict[str, float]]:
        """피드백 데이터 기반 임계값 최적화 (간소화 버전)"""
        try:
            best_threshold = self.model.threshold if hasattr(self.model, 'threshold') else 0.5
            best_performance = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
            
            # 임계값 범위 탐색 (0.3 ~ 0.8)
            for threshold in np.arange(0.3, 0.81, 0.05):
                try:
                    # 임계값 임시 적용
                    if hasattr(self.model, 'threshold'):
                        self.model.threshold = threshold
                    
                    # 성능 평가
                    predictions = self.model.predict(texts)
                    
                    performance = {
                        'accuracy': accuracy_score(labels, predictions),
                        'precision': precision_score(labels, predictions, zero_division=0),
                        'recall': recall_score(labels, predictions, zero_division=0),
                        'f1': f1_score(labels, predictions, zero_division=0)
                    }
                    
                    # 최적 임계값 업데이트
                    if performance['f1'] > best_performance['f1']:
                        best_threshold = threshold
                        best_performance = performance
                        
                except Exception as e:
                    continue
            
            return best_threshold, best_performance
            
        except Exception as e:
            self.logger.warning(f"⚠️ 임계값 최적화 실패: {e}")
            return 0.5, {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    
    def _optimize_weights_from_feedback(self, texts: List[str], labels: List[int]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """피드백 데이터 기반 가중치 최적화 (간소화 버전)"""
        try:
            if not hasattr(self.model, 'model_weights'):
                # 가중치가 없는 모델의 경우 기본값 반환
                default_weights = {'model1': 1.0, 'model2': 1.0, 'model3': 1.0}
                performance = self._evaluate_model_on_data(texts, labels)
                return default_weights, performance
            
            best_weights = self.model.model_weights.copy()
            best_performance = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
            
            # 가중치 조합 탐색 (간소화)
            weight_combinations = [
                {'model1': 1.0, 'model2': 0.8, 'model3': 0.6},
                {'model1': 0.8, 'model2': 1.0, 'model3': 0.8},
                {'model1': 0.6, 'model2': 0.8, 'model3': 1.0},
                {'model1': 1.2, 'model2': 1.0, 'model3': 0.8},
                {'model1': 0.8, 'model2': 1.2, 'model3': 1.0}
            ]
            
            for weights in weight_combinations:
                try:
                    # 가중치 임시 적용
                    self.model.model_weights = weights
                    
                    # 성능 평가
                    performance = self._evaluate_model_on_data(texts, labels)
                    
                    # 최적 가중치 업데이트
                    if performance['f1'] > best_performance['f1']:
                        best_weights = weights.copy()
                        best_performance = performance
                        
                except Exception as e:
                    continue
            
            return best_weights, best_performance
            
        except Exception as e:
            self.logger.warning(f"⚠️ 가중치 최적화 실패: {e}")
            default_weights = {'model1': 1.0, 'model2': 1.0, 'model3': 1.0}
            performance = self._evaluate_model_on_data(texts, labels)
            return default_weights, performance
    
    def _evaluate_model_on_data(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """특정 데이터에 대한 모델 성능 평가"""
        try:
            predictions = self.model.predict(texts)
            
            return {
                'accuracy': accuracy_score(labels, predictions),
                'precision': precision_score(labels, predictions, zero_division=0),
                'recall': recall_score(labels, predictions, zero_division=0),
                'f1': f1_score(labels, predictions, zero_division=0)
            }
        except Exception as e:
            self.logger.warning(f"⚠️ 성능 평가 실패: {e}")
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    
    def get_rl_optimization_status(self) -> Dict[str, Any]:
        """강화학습 최적화 상태 조회"""
        try:
            current_performance = self._evaluate_model()
            
            status = {
                'optimizer_initialized': True,
                'baseline_performance': self.baseline_performance,
                'current_performance': current_performance,
                'improvement': {
                    'accuracy_gain': current_performance['accuracy'] - self.baseline_performance['accuracy'],
                    'f1_gain': current_performance['f1'] - self.baseline_performance['f1']
                },
                'model_config': {
                    'threshold': getattr(self.model, 'threshold', 0.5),
                    'weights': getattr(self.model, 'model_weights', {}),
                },
                'test_data_size': len(self.test_texts),
                'last_updated': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            return {
                'optimizer_initialized': False,
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }

    def retrain_deep_learning_model(self, feedback_data: List[Dict]) -> Dict[str, Any]:
        """실제 딥러닝 모델 재훈련 - pkl 모델이 진짜로 똑똑해짐!"""
        try:
            self.logger.info("\n🧠 실제 딥러닝 모델 재훈련 시작!")
            self.logger.info("="*50)
            
            # 피드백 데이터를 학습 데이터로 변환
            texts, labels = self._convert_feedback_to_training_data(feedback_data)
            
            if len(texts) < 5:
                return {
                    'status': 'insufficient_data',
                    'message': f'재훈련을 위한 데이터가 부족합니다. (현재: {len(texts)}개, 필요: 5개)',
                    'data_count': len(texts)
                }
            
            self.logger.info(f"📚 학습 데이터: {len(texts)}개 (긍정: {sum(labels)}, 부정: {len(labels)-sum(labels)})")
            
            # AIEngine에서 딥러닝 모델 찾기
            deep_model = None
            feature_extractor = None
            
            # AIEngine의 model_manager에서 딥러닝 모델 찾기
            if hasattr(self.model, 'model_manager') and hasattr(self.model.model_manager, 'deep_learning_model'):
                deep_learning_engine = self.model.model_manager.deep_learning_model
                if deep_learning_engine and hasattr(deep_learning_engine, 'model') and hasattr(deep_learning_engine, 'feature_extractor'):
                    deep_model = deep_learning_engine.model
                    feature_extractor = deep_learning_engine.feature_extractor
                    self.logger.info("✅ AIEngine에서 딥러닝 모델 발견")
            
            # 직접 딥러닝 엔진에서 찾기 (백업)
            if deep_model is None:
                try:
                    from .deep_learning_engine import get_deep_learning_engine
                    deep_learning_engine = get_deep_learning_engine()
                    if deep_learning_engine and hasattr(deep_learning_engine, 'model') and hasattr(deep_learning_engine, 'feature_extractor'):
                        deep_model = deep_learning_engine.model
                        feature_extractor = deep_learning_engine.feature_extractor
                        self.logger.info("✅ 직접 딥러닝 엔진에서 모델 발견")
                except Exception as e:
                    self.logger.warning(f"⚠️ 직접 딥러닝 엔진 로드 실패: {e}")
            
            if deep_model is None or feature_extractor is None:
                self.logger.warning("⚠️ 딥러닝 모델 또는 특성 추출기를 찾을 수 없음")
                return {'status': 'no_model', 'message': '딥러닝 모델을 찾을 수 없습니다.'}
            
            self.logger.info("✅ 딥러닝 모델 발견 - 실제 재훈련 시작")
            
            # 성능 평가 전
            before_performance = self._evaluate_model()
            self.logger.info(f"🎯 재훈련 전 성능: 정확도 {before_performance['accuracy']:.3f}")
            
            # 실제 모델 재훈련 실행
            retrain_result = self._perform_actual_retraining(
                deep_model, feature_extractor, texts, labels
            )
            
            if retrain_result['success']:
                # 성능 평가 후
                after_performance = self._evaluate_model()
                self.logger.info(f"🎉 재훈련 후 성능: 정확도 {after_performance['accuracy']:.3f}")
                
                improvement = after_performance['accuracy'] - before_performance['accuracy']
                self.logger.info(f"📈 성능 향상: {improvement:+.3f}")
                
                # 향상된 모델 저장
                self._save_improved_model()
                
                # 데이터베이스에 학습 이벤트 기록
                try:
                    if hasattr(self, 'ai_engine') and self.ai_engine and hasattr(self.ai_engine, 'db_manager'):
                        self.ai_engine.db_manager.record_learning_event(
                            learning_type='deep_learning_retrain',
                            performance_before=before_performance,
                            performance_after=after_performance,
                            details={
                                'training_samples': len(texts),
                                'epochs_trained': retrain_result.get('epochs', 0),
                                'final_loss': retrain_result.get('final_loss', 0),
                                'improvement': improvement
                            }
                        )
                        self.logger.info(f"📊 DB에 학습 통계 업데이트 완료")
                except Exception as e:
                    self.logger.warning(f"⚠️ DB 통계 업데이트 실패: {e}")
                
                return {
                    'status': 'success',
                    'message': '딥러닝 모델 재훈련 완료!',
                    'before_performance': before_performance,
                    'after_performance': after_performance,
                    'improvement': improvement,
                    'training_samples': len(texts),
                    'epochs_trained': retrain_result.get('epochs', 0),
                    'final_loss': retrain_result.get('final_loss', 0),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'training_failed',
                    'message': retrain_result.get('error', '재훈련 중 오류 발생'),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"❌ 딥러닝 모델 재훈련 실패: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _perform_actual_retraining(self, model, feature_extractor, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """실제 딥러닝 모델 가중치 업데이트"""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            
            self.logger.info("🔥 PyTorch 재훈련 시작...")
            
            # 디바이스 설정 - Config에서 올바른 디바이스 가져오기
            from .config import Config
            device = Config.DEVICE  # MPS, CUDA, 또는 CPU 자동 선택
            
            self.logger.info(f"🖥️ 학습 디바이스: {device} ({Config.DEVICE_NAME})")
            
            # 특성 추출
            self.logger.info("🔧 텍스트 특성 추출 중...")
            features = feature_extractor.extract_features(texts, is_training=True)
            
            # PyTorch 텐서로 변환
            X_tensor = torch.FloatTensor(features).to(device)
            y_tensor = torch.FloatTensor(labels).to(device)
            
            self.logger.info(f"📊 학습 데이터 형태: {X_tensor.shape}")
            
            # 데이터 로더 생성
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=min(16, len(texts)), shuffle=True)
            
            # 모델을 학습 모드로 전환
            model.train()
            
            # 옵티마이저 설정 (낮은 학습률로 파인튜닝)
            optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
            criterion = nn.BCEWithLogitsLoss()
            
            # 학습 실행
            num_epochs = min(10, max(3, len(texts) // 2))  # 데이터 양에 따라 조정
            self.logger.info(f"📚 {num_epochs} 에포크 동안 파인튜닝...")
            
            total_loss = 0
            for epoch in range(num_epochs):
                epoch_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    
                    # 순전파
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    
                    # 역전파
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_epoch_loss = epoch_loss / len(dataloader)
                total_loss += avg_epoch_loss
                self.logger.info(f"   에포크 {epoch+1}/{num_epochs}: Loss = {avg_epoch_loss:.4f}")
            
            final_loss = total_loss / num_epochs
            
            # 모델을 평가 모드로 전환
            model.eval()
            
            self.logger.info(f"✅ 재훈련 완료! 평균 Loss: {final_loss:.4f}")
            
            return {
                'success': True,
                'epochs': num_epochs,
                'final_loss': final_loss,
                'samples_trained': len(texts)
            }
            
        except Exception as e:
            self.logger.error(f"❌ 실제 재훈련 실패: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _save_improved_model(self):
        """향상된 모델 저장"""
        try:
            import pickle
            from datetime import datetime
            
            # 타임스탬프가 포함된 백업 파일명
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"models/rl_improved_model_{timestamp}.pkl"
            
            self.logger.info(f"💾 향상된 모델 저장 중: {backup_path}")
            
            # AIEngine의 model_manager에서 딥러닝 엔진 찾기
            deep_learning_engine = None
            if hasattr(self.model, 'model_manager') and hasattr(self.model.model_manager, 'deep_learning_model'):
                deep_learning_engine = self.model.model_manager.deep_learning_model
            
            # 직접 딥러닝 엔진에서 찾기 (백업)
            if deep_learning_engine is None:
                try:
                    from .deep_learning_engine import get_deep_learning_engine
                    deep_learning_engine = get_deep_learning_engine()
                except Exception as e:
                    self.logger.warning(f"⚠️ 딥러닝 엔진 찾기 실패: {e}")
                    deep_learning_engine = self.model  # AIEngine 자체를 저장
            
            # 모델 저장
            with open(backup_path, 'wb') as f:
                pickle.dump(deep_learning_engine, f)
            
            # 메타데이터 저장
            metadata = {
                'improvement_type': 'reinforcement_learning_feedback',
                'improved_at': datetime.now().isoformat(),
                'baseline_accuracy': self.baseline_performance.get('accuracy', 0),
                'current_accuracy': self._evaluate_model().get('accuracy', 0),
                'training_samples': len(self.test_texts),
                'model_path': backup_path
            }
            
            metadata_path = f"models/rl_improvement_log_{timestamp}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ 모델 저장 완료: {backup_path}")
            self.logger.info(f"📋 메타데이터 저장: {metadata_path}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 모델 저장 실패: {e}")


def main():
    """메인 실행 함수"""
    logger = get_logger(__name__)
    logger.info("🚀 스타트업 모니터링 시스템 - 강화학습 최적화")
    logger.info("="*60)
    
    # 파일 경로 설정 (코랩에서 다운로드한 파일들)
    model_path = 'improved_ensemble_model.pkl'  # 또는 다른 모델
    test_data_path = 'test_data.json'
    
    # 파일 존재 확인
    if not os.path.exists(model_path):
        logger.error(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        logger.info("💡 코랩에서 14-15단계를 실행하여 파일을 다운로드하세요!")
        return
    
    if not os.path.exists(test_data_path):
        logger.error(f"❌ 테스트 데이터를 찾을 수 없습니다: {test_data_path}")
        return
    
    # 최적화 시스템 초기화
    optimizer = ReinforcementLearningOptimizer(model_path, test_data_path)
    
    # 1. 베이스라인 분석
    logger.info("\n" + "="*60)
    optimizer.detailed_analysis()
    
    # 2. Optuna 최적화 (빠르고 효과적)
    logger.info("\n" + "="*60)
    best_params, performance = optimizer.hyperparameter_search_optuna(n_trials=50)
    
    # 3. 강화학습 임계값 최적화
    logger.info("\n" + "="*60)
    threshold, threshold_performance = optimizer.optimize_threshold_rl(total_timesteps=5000)
    
    # 4. 강화학습 가중치 최적화  
    logger.info("\n" + "="*60)
    weights, weights_performance = optimizer.optimize_weights_rl(total_timesteps=5000)
    
    # 5. 최종 분석
    logger.info("\n" + "="*60)
    final_performance, final_problems = optimizer.detailed_analysis()
    
    # 6. 최적화된 모델 저장
    optimizer.save_optimized_model()
    
    logger.info(f"\n🎉 강화학습 최적화 완료!")
    logger.info(f"   최종 정확도: {final_performance['accuracy']:.3f}")
    logger.info(f"   남은 문제 케이스: {len(final_problems)}개")


if __name__ == "__main__":
    main() 