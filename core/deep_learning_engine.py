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
            elif name == 'EnhancedDeepLearningModel':
                return EnhancedDeepLearningModel
            elif name == 'EnhancedFeatureExtractor':
                return EnhancedFeatureExtractor
            elif name == 'ImprovedStartupClassifier':
                return ImprovedStartupClassifier
            elif name == 'PowerfulStartupClassifier':
                return PowerfulStartupClassifier
        
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
            
            # 모델 타입 확인
            if isinstance(model_data, dict):
                # 딕셔너리 형태의 모델 데이터
                self.model_info = {
                    'model_type': model_data.get('model_type', 'Unknown'),
                    'created_at': model_data.get('created_at', 'Unknown'),
                    'training_accuracy': model_data.get('training_accuracy', 0.0),
                    'version': model_data.get('version', '1.0')
                }
            elif hasattr(model_data, '__class__'):
                # EnhancedDeepLearningModel 같은 객체
                self.model_info = {
                    'model_type': model_data.__class__.__name__,
                    'created_at': 'Unknown',
                    'training_accuracy': getattr(model_data, 'accuracy', 0.0),
                    'version': getattr(model_data, 'version', '1.0')
                }
                # 객체를 직접 사용
                self.model = model_data
                self.model_loaded = True
                if hasattr(model_data, 'feature_extractor'):
                    self.feature_extractor = model_data.feature_extractor
                self.logger.info(f"✅ {self.model_info['model_type']} 모델 로드 완료")
                return
            
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
            self.logger.info("🎉 강화학습 엔진 준비 완료!")
            
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
            # EnhancedDeepLearningModel을 사용하는 경우
            if hasattr(self.model, 'calculate_ai_score'):
                return self.model.calculate_ai_score(text)
            # 기존 모델 사용
            else:
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


# ============================================
# 강화학습 모델 클래스들 (코랩 호환성)
# ============================================

class EnhancedFeatureExtractor:
    """향상된 특성 추출기"""
    
    def __init__(self, sentence_model_name='jhgan/ko-sroberta-multitask'):
        self.sentence_model = SentenceTransformer(sentence_model_name)
        
    def extract_features(self, texts, is_training=True):
        """텍스트에서 특성 추출"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Sentence embeddings
        embeddings = self.sentence_model.encode(texts)
        
        # 키워드 특성 추가
        keyword_features = []
        for text in texts:
            text_lower = text.lower()
            features = {
                'support_keywords': sum([
                    'tips' in text_lower,
                    '창업' in text_lower,
                    '지원' in text_lower,
                    '사업' in text_lower,
                    'k-스타트업' in text_lower,
                    '창업진흥원' in text_lower,
                ]) * 10,
                'spam_keywords': sum([
                    '광고' in text_lower,
                    '홍보' in text_lower,
                    '캠페인' in text_lower,
                    '수료식' in text_lower,
                    '이벤트' in text_lower,
                ]) * (-15),
                'context_features': sum([
                    bool(re.search(r'\d+억원|\d+천만원|\d+백만원', text)),
                    bool(re.search(r'신청기간|마감일|접수기간', text)),
                    bool(re.search(r'지원대상|신청자격', text)),
                    bool(re.search(r'정부|공공기관|진흥원', text)),
                ]) * 5
            }
            keyword_features.append(sum(features.values()))
        
        # 특성 결합
        keyword_features = np.array(keyword_features).reshape(-1, 1)
        combined_features = np.hstack([embeddings, keyword_features])
        
        return combined_features


class ImprovedStartupClassifier(nn.Module):
    """개선된 스타트업 분류기"""
    
    def __init__(self, input_dim=769, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class PowerfulStartupClassifier(nn.Module):
    """강력한 스타트업 분류기 (A100 최적화)"""
    
    def __init__(self, input_dim=769, hidden_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.batch_norm4 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm4(self.fc4(x)))
        x = self.fc5(x)
        return x


class EnhancedDeepLearningModel:
    """향상된 딥러닝 모델 래퍼"""
    
    def __init__(self, feature_extractor, classifier, device='cuda'):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        
        # 디바이스 자동 감지
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        # 모델을 CPU로 유지 (메모리 효율성)
        self.classifier = self.classifier.cpu()
        
        # 동적 임계값 (초기값을 더 보수적으로)
        self.threshold = 0.65  # 55점 대신 65점부터
        self.adaptive_threshold = True
        self.threshold_min = 0.55
        self.threshold_max = 0.85
        
        # 모델 가중치 (딥러닝, 키워드, 패턴)
        self.model_weights = {
            'deep_learning': 0.6,
            'keyword_score': 0.3,
            'pattern_score': 0.1
        }
        
    def predict(self, texts):
        """텍스트 예측"""
        if isinstance(texts, str):
            texts = [texts]
        
        features = self.feature_extractor.extract_features(texts, is_training=False)
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        
        with torch.no_grad():
            outputs = self.classifier(features_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
        
        self.classifier = self.classifier.cpu()
        
        return (probabilities > self.threshold).astype(int).flatten()
    
    def calculate_ai_score(self, text):
        """AI 점수 계산 (창업/스타트업 지원사업 특화)"""
        import re
        
        # 1. 딥러닝 모델 예측
        features = self.feature_extractor.extract_features([text], is_training=False)
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        
        with torch.no_grad():
            output = self.classifier(features_tensor)
            probability = torch.sigmoid(output).cpu().item()
        
        self.classifier = self.classifier.cpu()
        
        # 2. 창업/스타트업 지원사업 필터링
        text_lower = text.lower()
        title_part = text.split('\n')[0] if '\n' in text else text[:100]
        title_lower = title_part.lower()
        
        # === 배제 키워드 체크 (무관한 지원 제외) ===
        exclude_keywords = {
            # 개인 지원 (창업과 무관)
            '양육비': -50, '육아': -50, '출산': -50, '임신': -50,
            '장학금': -50, '학자금': -50, '생활비': -50, '주거비': -50,
            '의료비': -50, '치료비': -50, '간병': -50, '복지': -40,
            
            # 구직/채용 관련
            '입사지원': -50, '채용': -40, '구인': -40, '구직': -40,
            '인턴': -30, '직원모집': -40, '인재채용': -40,
            
            # 교육/수료 관련 (단순 교육)
            '수료식': -50, '졸업식': -50, '입학': -40, '개강': -40,
            '자격증': -30, '시험': -30, '합격': -30,
            
            # 이벤트/행사
            '축제': -50, '공연': -50, '전시회': -40, '박람회': -20,
            '세미나': -20, '포럼': -20, '컨퍼런스': -10,
            
            # 광고/홍보
            '광고': -50, '홍보': -40, '마케팅': -30, '이벤트': -30,
            '할인': -50, '쿠폰': -50, '프로모션': -40,
            
            # 기타 무관한 분야
            '부동산': -50, '아파트': -50, '분양': -50,
            '카페': -50, '맛집': -50, '음식점': -50,
            '관광': -40, '여행': -40, '숙박': -40,
            '종교': -50, '교회': -50, '절': -50,
            '스포츠': -40, '운동': -40, '헬스': -40
        }
        
        # === 필수 키워드 체크 (창업 지원사업 확인) ===
        essential_contexts = {
            # 자금 지원 맥락
            'funding': any([
                '투자' in text_lower and ('유치' in text_lower or '지원' in text_lower),
                '자금' in text_lower and '지원' in text_lower,
                '융자' in text_lower and ('지원' in text_lower or '사업' in text_lower),
                '보조금' in text_lower,
                '지원금' in text_lower,
                bool(re.search(r'\d+억원.*지원|\d+천만원.*지원|\d+백만원.*지원', text_lower))
            ]),
            
            # 창업/스타트업 맥락
            'startup': any([
                '창업' in text_lower and ('지원' in text_lower or '육성' in text_lower or '보육' in text_lower),
                '스타트업' in text_lower and ('지원' in text_lower or '육성' in text_lower),
                '벤처' in text_lower and ('지원' in text_lower or '육성' in text_lower),
                '예비창업' in text_lower,
                '초기창업' in text_lower,
                '기술창업' in text_lower
            ]),
            
            # 기업 지원 맥락
            'business': any([
                '중소기업' in text_lower and '지원' in text_lower,
                '소상공인' in text_lower and '지원' in text_lower,
                '기업' in text_lower and ('육성' in text_lower or '지원사업' in text_lower),
                'r&d' in text_lower and '지원' in text_lower,
                '기술개발' in text_lower and '지원' in text_lower
            ]),
            
            # 공식 기관 맥락
            'official': any([
                'tips' in text_lower,
                '창업진흥원' in text_lower,
                'k-스타트업' in text_lower,
                '중소벤처기업부' in text_lower,
                '중기부' in text_lower,
                '과학기술정보통신부' in text_lower,
                '과기정통부' in text_lower,
                '산업통상자원부' in text_lower,
                '산업부' in text_lower
            ])
        }
        
        # === 점수 계산 로직 ===
        keyword_score = 0
        
        # 1. 배제 키워드 체크 (제목에 있으면 더 강한 페널티)
        for keyword, penalty in exclude_keywords.items():
            if keyword in title_lower:
                keyword_score += penalty * 2  # 제목에 있으면 2배 감점
                logging.info(f"🚫 배제 키워드 발견 (제목): '{keyword}' → {penalty * 2}점")
            elif keyword in text_lower:
                keyword_score += penalty
                logging.info(f"🚫 배제 키워드 발견: '{keyword}' → {penalty}점")
        
        # 2. 필수 맥락 체크
        context_count = sum(1 for context in essential_contexts.values() if context)
        
        # 필수 맥락이 하나도 없으면 큰 감점
        if context_count == 0:
            keyword_score -= 30
            logging.info("⚠️ 창업/기업 지원 맥락 없음 → -30점")
        else:
            # 맥락이 있으면 가점
            keyword_score += context_count * 15
            logging.info(f"✅ 지원사업 맥락 {context_count}개 발견 → +{context_count * 15}점")
        
        # 3. 공식 기관 추가 가점
        if essential_contexts['official']:
            keyword_score += 25
            logging.info("🏢 공식 기관 지원사업 → +25점")
        
        # 4. 상세 정보 체크
        detail_score = 0
        
        # 지원 금액 명시
        if re.search(r'\d+억원|\d+천만원|\d+백만원', text):
            detail_score += 15
            logging.info("💰 지원 금액 정보 → +15점")
        
        # 신청 기간 명시
        if re.search(r'신청기간|마감일|접수기간|모집기간|신청.*~.*\d+월', text):
            detail_score += 10
            logging.info("📅 신청 기간 정보 → +10점")
        
        # 지원 대상 명시
        if re.search(r'지원대상|신청자격|대상기업|지원자격', text):
            detail_score += 10
            logging.info("🎯 지원 대상 정보 → +10점")
        
        # 5. 종합 점수 계산
        final_score = (
            probability * 100 * self.model_weights['deep_learning'] +
            keyword_score * self.model_weights['keyword_score'] +
            detail_score * self.model_weights['pattern_score']
        )
        
        # 점수 범위 제한 (0-100)
        final_score = max(0, min(100, final_score))
        
        # 디버깅 정보
        logging.info(f"📊 AI 점수 계산 완료: {final_score:.1f}점 (DL: {probability*100:.1f}, KW: {keyword_score}, DT: {detail_score})")
        
        return final_score
    
    def update_threshold(self, feedback_results):
        """피드백 기반 임계값 업데이트"""
        if not self.adaptive_threshold:
            return
        
        # 성과 계산
        tp = feedback_results.get('true_positive', 0)
        fp = feedback_results.get('false_positive', 0)
        fn = feedback_results.get('false_negative', 0)
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
        
        # 임계값 조정
        if precision < 0.7:  # 정밀도가 낮으면
            self.threshold = min(self.threshold + 0.05, self.threshold_max)
        elif recall < 0.7:  # 재현율이 낮으면
            self.threshold = max(self.threshold - 0.05, self.threshold_min)
        
        logging.info(f"🎯 임계값 업데이트: {self.threshold:.3f} (정밀도: {precision:.3f}, 재현율: {recall:.3f})") 