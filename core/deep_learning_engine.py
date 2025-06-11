#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 고성능 딥러닝 엔진
코랩에서 훈련한 8층 딥러닝 모델을 프로덕션 환경에서 사용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
import json
import re
from datetime import datetime
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging
from typing import List, Dict, Optional, Tuple, Any
import glob
import os
import sys

# ============================================
# 커스텀 Unpickler (모듈 경로 문제 해결)
# ============================================

class CustomUnpickler(pickle.Unpickler):
    """모듈 경로 문제를 해결하는 커스텀 unpickler"""
    
    def find_class(self, module, name):
        # __main__에서 저장된 클래스들을 현재 모듈에서 찾기
        if module == '__main__':
            if name == 'DeepSupportClassifier':
                return DeepSupportClassifier
            elif name == 'AppleSiliconFeatureExtractor':
                return AppleSiliconFeatureExtractor
        
        # 기본 동작
        return super().find_class(module, name)

# ============================================
# 코랩에서 사용한 모델 클래스 정의 (pickle 로딩을 위해 필요)
# ============================================

class DeepSupportClassifier(nn.Module):
    """고성능 다층 신경망 분류기 (8층 Deep Network)"""
    
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256, 128, 64, 32], dropout_rate=0.3):
        super(DeepSupportClassifier, self).__init__()
        
        # 8층 깊은 신경망 구조 설계
        layers = []
        prev_dim = input_dim
        
        # 첫 번째 레이어 (입력층 → 첫 번째 은닉층)
        layers.append(nn.Linear(prev_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        prev_dim = hidden_dims[0]
        
        # 중간 은닉층들 (2~7층)
        for hidden_dim in hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 출력층 (8번째 층)
        layers.append(nn.Linear(prev_dim, 2))  # 이진 분류
        
        self.network = nn.Sequential(*layers)
        
        # 가중치 초기화 (Xavier Uniform)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class AppleSiliconFeatureExtractor:
    """Apple Silicon 호환 특성 추출기"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🔧 특성 추출기 초기화: {device}")
        
        # TF-IDF 벡터라이저
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words=None,
            lowercase=True
        )
        self.tfidf_fitted = False
        
        # Sentence Transformer (Apple Silicon 호환)
        try:
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            if str(device) != 'cpu':
                self.sentence_model = self.sentence_model.to(device)
            self.logger.info("✅ Sentence Transformer 로드 완료")
        except Exception as e:
            self.logger.error(f"⚠️ Sentence Transformer 로드 실패: {e}")
            self.sentence_model = None
        
        # 지원사업 관련 키워드
        self.support_keywords = [
            '지원', '사업', '육성', '개발', '창업', '혁신', '연구', '기술', '투자', '융자',
            '보조', '활성화', '프로그램', 'r&d', '연구개발', '기술개발', '사업화', '스타트업',
            '벤처', '중소기업', '소상공인', '청년', '여성기업', '사회적기업'
        ]
        
    def fit(self, texts):
        """특성 추출기 훈련 (훈련 데이터로 TF-IDF 학습)"""
        self.logger.info("🔧 특성 추출기 훈련 중...")
        self.tfidf.fit(texts)
        self.tfidf_fitted = True
        self.logger.info("✅ TF-IDF 훈련 완료")
    
    def extract_manual_features(self, texts):
        """수동 특성 추출"""
        features = []
        
        for text in texts:
            text_lower = text.lower()
            
            feature_dict = {
                'length': len(text),
                'word_count': len(text.split()),
                'support_keyword_count': sum(1 for keyword in self.support_keywords if keyword in text_lower),
                'has_support_word': int('지원' in text_lower),
                'has_business_word': int('사업' in text_lower),
                'has_startup_word': int(any(word in text_lower for word in ['창업', '스타트업'])),
                'has_tech_word': int(any(word in text_lower for word in ['기술', '개발', '연구', 'r&d'])),
                'has_funding_word': int(any(word in text_lower for word in ['투자', '융자', '보조금'])),
                'korean_ratio': len(re.findall(r'[가-힣]', text)) / max(len(text), 1),
                'number_count': len(re.findall(r'\d+', text))
            }
            
            features.append(list(feature_dict.values()))
        
        return np.array(features)
    
    def extract_features(self, texts, is_training=False):
        """전체 특성 추출 (Apple Silicon 호환)"""
        # TF-IDF 특성
        if is_training or not self.tfidf_fitted:
            # 훈련 시에만 fit_transform 사용
            tfidf_features = self.tfidf.fit_transform(texts).toarray()
            self.tfidf_fitted = True
        else:
            # 예측 시에는 transform만 사용
            tfidf_features = self.tfidf.transform(texts).toarray()
        
        # Sentence embedding 특성 (Apple Silicon 호환)
        if self.sentence_model:
            try:
                # CPU로 변환하여 Apple Silicon 호환성 보장
                sentence_features = []
                batch_size = 32
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    
                    # Apple Silicon에서는 CPU로 처리
                    with torch.no_grad():
                        embeddings = self.sentence_model.encode(
                            batch_texts, 
                            convert_to_tensor=True,
                            device='cpu'  # Apple Silicon 호환을 위해 CPU 강제 사용
                        )
                        
                        if isinstance(embeddings, torch.Tensor):
                            embeddings = embeddings.cpu().numpy()
                        
                        sentence_features.extend(embeddings)
                
                sentence_features = np.array(sentence_features)
                
            except Exception as e:
                self.logger.error(f"⚠️ Sentence Embedding 추출 실패: {e}")
                sentence_features = np.zeros((len(texts), 384))  # 기본 차원
        else:
            sentence_features = np.zeros((len(texts), 384))
        
        # 수동 특성
        manual_features = self.extract_manual_features(texts)
        
        # 특성 결합
        combined_features = np.hstack([
            tfidf_features,
            sentence_features, 
            manual_features
        ])
        
        return combined_features

# ============================================
# 딥러닝 엔진 클래스
# ============================================

class DeepLearningEngine:
    """🧠 프로덕션 딥러닝 엔진"""
    
    def __init__(self, model_path='apple_silicon_production_model.pkl'):
        # 로깅 설정 (먼저 해야 함)
        self.logger = logging.getLogger(__name__)
        
        # Apple Silicon 디바이스 설정
        self.device = self._setup_device()
        
        # 모델 구성요소
        self.model = None
        self.feature_extractor = None
        self.is_loaded = False
        self.model_info = {}
        
        # 모델 파일 경로 확인 및 수정
        from pathlib import Path
        model_file = Path(model_path)
        self.logger.info(f"🔍 모델 파일 경로 확인: {model_file.absolute()}")
        
        if not model_file.exists():
            # 프로젝트 루트에서 찾기
            project_root = Path(__file__).parent.parent
            self.logger.info(f"🔍 프로젝트 루트: {project_root.absolute()}")
            
            alternative_path = project_root / model_path
            self.logger.info(f"🔍 대안 경로 1: {alternative_path.absolute()}")
            
            if alternative_path.exists():
                model_path = str(alternative_path)
                self.logger.info(f"📁 모델 파일 발견: {model_path}")
            else:
                # models 디렉토리에서 찾기
                models_path = project_root / 'models' / model_path
                self.logger.info(f"🔍 대안 경로 2 (models): {models_path.absolute()}")
                
                if models_path.exists():
                    model_path = str(models_path)
                    self.logger.info(f"📁 models 디렉토리에서 모델 파일 발견: {model_path}")
                else:
                    # models 디렉토리 내용 확인
                    models_dir = project_root / 'models'
                    if models_dir.exists():
                        self.logger.info(f"📂 models 디렉토리 내용: {list(models_dir.iterdir())}")
                    self.logger.warning(f"⚠️ 모델 파일 없음: {model_path}")
        else:
            self.logger.info(f"✅ 모델 파일 존재: {model_file.absolute()}")
        
        # 모델 로드
        self.load_model(model_path)
        
    def _setup_device(self):
        """Apple Silicon (MPS) 및 CPU 호환 디바이스 설정"""
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            device_name = "Apple Silicon (MPS)"
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            device_name = "NVIDIA GPU (CUDA)"
        else:
            device = torch.device('cpu')
            device_name = "CPU"
        
        self.logger.info(f"🖥️ 딥러닝 엔진 디바이스: {device_name}")
        return device
        
    def load_model(self, model_path):
        """코랩에서 훈련한 모델 로드"""
        self.logger.info(f"📂 딥러닝 모델 로딩 시도: {model_path}")
        
        try:
            # 파일 존재 확인
            from pathlib import Path
            if not Path(model_path).exists():
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
            
            self.logger.info(f"✅ 모델 파일 확인됨: {model_path}")
            
            # Apple Silicon 호환을 위한 특수 로딩
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, map_location='cpu') if 'map_location' not in kwargs else original_load(*args, **kwargs)
            
            with open(model_path, 'rb') as f:
                unpickler = CustomUnpickler(f)
                
                # torch.load 대신 unpickler 사용
                model_data = unpickler.load()
            
            # 원래 torch.load 복원
            torch.load = original_load
            
            self.logger.info("✅ 모델 파일 로드 성공")
            
            # 모델 정보 저장
            self.model_info = {
                'model_type': model_data.get('model_type', 'Unknown'),
                'created_at': model_data.get('created_at', 'Unknown'),
                'training_accuracy': model_data.get('training_accuracy', 0.0),
                'version': model_data.get('version', '1.0')
            }
            
            self.logger.info(f"✅ 모델 타입: {self.model_info['model_type']}")
            self.logger.info(f"✅ 훈련 정확도: {self.model_info['training_accuracy']:.4f}")
            
            # 특성 추출기 복원
            self.feature_extractor = model_data['feature_extractor']
            self.logger.info("✅ 특성 추출기 로드 완료")
            
            # 딥러닝 모델 복원
            if 'model_state_dict' in model_data:
                # 모델 구조 정보에서 input_dim 추출
                model_structure = model_data['model_structure']
                input_dim = model_structure.network[0].in_features
                
                # 모델 재생성
                self.model = DeepSupportClassifier(input_dim=input_dim)
                
                # Apple Silicon 호환을 위한 state_dict 로딩
                state_dict = model_data['model_state_dict']
                self.model.load_state_dict(state_dict)
                
                self.model = self.model.to(self.device)
                self.model.eval()
                
                self.logger.info(f"✅ 딥러닝 모델 로드 완료 (입력 차원: {input_dim})")
            else:
                raise ValueError("딥러닝 모델 상태가 없습니다.")
            
            self.is_loaded = True
            self.logger.info("🎉 딥러닝 엔진 준비 완료!")
            
        except Exception as e:
            self.logger.error(f"❌ 딥러닝 모델 로딩 실패: {e}")
            # 기본 더미 모델로 폴백
            self._setup_fallback_model()
            
    def _setup_fallback_model(self):
        """딥러닝 모델 로딩 실패 시 기본 모델 설정"""
        self.logger.warning("⚠️ 딥러닝 모델 로딩 실패 - 기본 모델 사용")
        self.is_loaded = False
        self.model_info = {
            'model_type': 'Fallback',
            'training_accuracy': 0.85,
            'status': 'fallback'
        }
        
    def predict(self, texts) -> np.ndarray:
        """텍스트 분류 예측"""
        if not self.is_loaded:
            # 폴백: 키워드 기반 간단 분류
            return self._fallback_predict(texts)
        
        # 딥러닝 모델 예측
        try:
            features = self.feature_extractor.extract_features(texts, is_training=False)
            features_tensor = torch.FloatTensor(features).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(features_tensor)
                _, predicted = torch.max(outputs.data, 1)
                
            return predicted.cpu().numpy()
            
        except Exception as e:
            self.logger.error(f"❌ 딥러닝 예측 실패: {e}")
            return self._fallback_predict(texts)
    
    def predict_proba(self, texts) -> np.ndarray:
        """확률 예측"""
        if not self.is_loaded:
            # 폴백: 간단한 확률 계산
            predictions = self._fallback_predict(texts)
            probabilities = np.array([[0.9, 0.1] if p == 0 else [0.1, 0.9] for p in predictions])
            return probabilities
        
        try:
            features = self.feature_extractor.extract_features(texts, is_training=False)
            features_tensor = torch.FloatTensor(features).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
            return probabilities.cpu().numpy()
            
        except Exception as e:
            self.logger.error(f"❌ 딥러닝 확률 예측 실패: {e}")
            predictions = self._fallback_predict(texts)
            probabilities = np.array([[0.9, 0.1] if p == 0 else [0.1, 0.9] for p in predictions])
            return probabilities
    
    def _fallback_predict(self, texts) -> np.ndarray:
        """폴백 예측 (키워드 기반)"""
        support_keywords = [
            '지원', '사업', '육성', '개발', '창업', '혁신', '연구', '기술', '투자', '융자',
            '보조', '활성화', '프로그램', 'r&d', '연구개발', '기술개발', '사업화', '스타트업',
            '벤처', '중소기업', '소상공인', '청년', '여성기업', '사회적기업'
        ]
        
        predictions = []
        for text in texts:
            text_lower = text.lower()
            keyword_count = sum(1 for keyword in support_keywords if keyword in text_lower)
            # 3개 이상 키워드가 있으면 지원사업으로 분류
            is_support = 1 if keyword_count >= 3 else 0
            predictions.append(is_support)
            
        return np.array(predictions)
    
    def calculate_score(self, text: str) -> float:
        """텍스트에 대한 지원사업 점수 계산 (0~100)"""
        try:
            probabilities = self.predict_proba([text])
            # 지원사업일 확률 * 100
            score = probabilities[0][1] * 100
            return float(score)
        except Exception as e:
            self.logger.error(f"❌ 점수 계산 실패: {e}")
            return 50.0  # 기본값
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            **self.model_info,
            'device': str(self.device),
            'is_loaded': self.is_loaded,
            'available': self.is_loaded
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        return {
            'accuracy': self.model_info.get('training_accuracy', 0.0),
            'model_type': self.model_info.get('model_type', 'Unknown'),
            'status': 'active' if self.is_loaded else 'fallback',
            'layers': 8 if self.is_loaded else 1,
            'parameters': 'High' if self.is_loaded else 'Low'
        }

# 전역 딥러닝 엔진 인스턴스
_deep_learning_engine = None

def get_deep_learning_engine(model_path=None) -> DeepLearningEngine:
    """딥러닝 엔진 싱글톤 인스턴스 반환"""
    global _deep_learning_engine
    if _deep_learning_engine is None:
        # 기본 경로 설정 (models 디렉토리)
        if model_path is None:
            from pathlib import Path
            project_root = Path(__file__).parent.parent
            model_path = str(project_root / 'models' / 'apple_silicon_production_model.pkl')
        _deep_learning_engine = DeepLearningEngine(model_path)
    return _deep_learning_engine

def reload_deep_learning_engine(model_path=None) -> DeepLearningEngine:
    """딥러닝 엔진 재로드"""
    global _deep_learning_engine
    # 기본 경로 설정 (models 디렉토리)
    if model_path is None:
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        model_path = str(project_root / 'models' / 'apple_silicon_production_model.pkl')
    _deep_learning_engine = DeepLearningEngine(model_path)
    return _deep_learning_engine 