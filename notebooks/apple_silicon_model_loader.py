#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🍎 Apple Silicon 호환 딥러닝 모델 로더
코랩에서 훈련된 모델을 로컬 Apple Silicon Mac에서 로드하는 스크립트
"""

import os
import sys
import logging
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 디바이스 설정 (Apple Silicon 우선)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "Apple Silicon GPU (MPS)"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = "NVIDIA GPU (CUDA)"
else:
    device = torch.device("cpu")
    device_name = "CPU"

logger.info(f"🖥️ 사용 디바이스: {device_name}")

class AppleSiliconFeatureExtractor:
    """Apple Silicon 호환 특성 추출기"""
    
    def __init__(self, device='mps'):
        self.device = device
        logger.info(f"🔧 특성 추출기 디바이스: {device}")
        
        # TF-IDF 벡터라이저
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,
            ngram_range=(1, 2)
        )
        
        # Sentence Transformer 모델
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            logger.info("✅ Sentence Transformer 로드 완료")
        except Exception as e:
            logger.warning(f"Sentence Transformer 로드 실패: {e}")
            self.sentence_model = None
        
        # 표준화 스케일러
        self.scaler = StandardScaler()
        
        # 훈련 상태
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        """특성 추출기 훈련"""
        logger.info("🔧 특성 추출기 훈련 중...")
        
        # TF-IDF 훈련
        self.tfidf_vectorizer.fit(texts)
        logger.info("✅ TF-IDF 훈련 완료")
        
        self.is_fitted = True
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """텍스트를 특성 벡터로 변환"""
        if not self.is_fitted:
            raise ValueError("특성 추출기가 훈련되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        logger.info("🔍 특성 추출 시작...")
        features_list = []
        
        # 1. TF-IDF 특성
        logger.info("📊 TF-IDF 특성 추출 중...")
        tfidf_features = self.tfidf_vectorizer.transform(texts).toarray()
        features_list.append(tfidf_features)
        
        # 2. Sentence Embedding 특성 (사용 가능한 경우)
        if self.sentence_model is not None:
            try:
                logger.info("🧠 Sentence Embedding 특성 추출 중...")
                
                # 배치 처리로 메모리 효율성 향상
                batch_size = 32
                sentence_features_list = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_embeddings = self.sentence_model.encode(
                        batch_texts,
                        convert_to_tensor=False,
                        show_progress_bar=False,
                        batch_size=min(batch_size, len(batch_texts))
                    )
                    sentence_features_list.extend(batch_embeddings)
                
                sentence_features = np.array(sentence_features_list)
                logger.info(f"✅ Sentence Embedding 형태: {sentence_features.shape}")
                features_list.append(sentence_features)
                
            except Exception as e:
                logger.warning(f"Sentence Embedding 추출 실패: {e}")
        
        # 3. 수동 특성 추출
        logger.info("🔧 수동 특성 추출 중...")
        manual_features = np.array([
            [len(text), len(text.split()), text.count('지원'), text.count('사업'), 
             text.count('기업'), text.count('창업'), len(text.split('.')), text.count('?')]
            for text in texts
        ])
        features_list.append(manual_features)
        
        # 모든 특성 결합
        combined_features = np.hstack(features_list)
        logger.info(f"✅ 전체 특성 형태: {combined_features.shape}")
        
        return combined_features

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

class AppleSiliconModelLoader:
    """코랩 훈련 모델을 Apple Silicon에서 로드하는 클래스"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.feature_extractor = None
        self.is_loaded = False
        
    def load_model(self, model_path='apple_silicon_production_model.pkl'):
        """코랩에서 훈련한 모델 로드"""
        logger.info(f"📂 모델 로딩 중: {model_path}")
        
        try:
            # 모델 데이터 로드 (Apple Silicon 호환)
            # pickle 내부의 torch 텐서도 CPU로 매핑
            import torch
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, map_location='cpu') if 'map_location' not in kwargs else original_load(*args, **kwargs)
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 원래 torch.load 복원
            torch.load = original_load
            
            logger.info(f"✅ 모델 타입: {model_data['model_type']}")
            logger.info(f"✅ 생성 시간: {model_data['created_at']}")
            logger.info(f"✅ 훈련 정확도: {model_data['training_accuracy']:.4f}")
            
            # 특성 추출기 복원
            self.feature_extractor = model_data['feature_extractor']
            logger.info("✅ 특성 추출기 로드 완료")
            
            # 딥러닝 모델 복원
            if 'model_state_dict' in model_data:
                # 모델 구조 정보에서 input_dim 추출
                model_structure = model_data['model_structure']
                input_dim = model_structure.network[0].in_features
                
                # 모델 재생성
                self.model = DeepSupportClassifier(input_dim=input_dim)
                
                # Apple Silicon 호환을 위한 map_location 설정
                state_dict = model_data['model_state_dict']
                if isinstance(state_dict, dict):
                    # CPU로 매핑하여 로드
                    self.model.load_state_dict(state_dict)
                else:
                    # torch.load가 필요한 경우
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
                        torch.save(state_dict, tmp.name)
                        self.model.load_state_dict(torch.load(tmp.name, map_location='cpu'))
                
                self.model = self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"✅ 딥러닝 모델 로드 완료 (입력 차원: {input_dim})")
            else:
                raise ValueError("딥러닝 모델 상태가 없습니다.")
            
            self.is_loaded = True
            logger.info("🎉 모델 로딩 완료!")
            
        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패: {e}")
            
    def predict(self, texts):
        """예측"""
        if not self.is_loaded:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        # 특성 추출
        features = self.feature_extractor.transform(texts)
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features_tensor)
            _, predicted = torch.max(outputs.data, 1)
            
        return predicted.cpu().numpy()
    
    def predict_proba(self, texts):
        """확률 예측"""
        if not self.is_loaded:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        # 특성 추출
        features = self.feature_extractor.transform(texts)
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
        return probabilities.cpu().numpy()

def test_model():
    """모델 테스트"""
    logger.info("�� Apple Silicon 딥러닝 모델 테스트 시작!")
    
    # 모델 로더 생성
    loader = AppleSiliconModelLoader(device=device)
    loader.load_model('../apple_silicon_production_model.pkl')
    
    # 테스트 데이터
    test_texts = [
        "중소기업 창업 지원 사업에 참여하고 싶습니다",
        "오늘 친구와 맛있는 저녁을 먹었습니다", 
        "기술개발 R&D 지원 프로그램 안내",
        "주말에 영화를 보러 갈 예정입니다",
        "벤처기업 투자 유치 지원사업",
        "새로운 드라마가 재미있었습니다",
        "스타트업 육성 프로그램 모집",
        "헬스장에서 운동을 했습니다"
    ]
    
    logger.info("\n🔍 예측 결과:")
    predictions = loader.predict(test_texts)
    probabilities = loader.predict_proba(test_texts)
    
    for i, text in enumerate(test_texts):
        pred_label = "지원사업" if predictions[i] == 1 else "일반텍스트"
        confidence = probabilities[i][predictions[i]]
        logger.info(f"📝 '{text}'")
        logger.info(f"   → {pred_label} (신뢰도: {confidence:.3f})")
        logger.info()

if __name__ == "__main__":
    logger.info("🍎 Apple Silicon 딥러닝 모델 로더 시작!")
    test_model() 