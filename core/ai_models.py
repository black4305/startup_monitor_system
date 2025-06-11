"""
AI 모델 관리 모듈 - 모델 로딩, 예측, 점수 계산
"""

import logging
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import os

from .config import Config

logger = logging.getLogger(__name__)

class AIModelManager:
    """AI 모델 관리자"""
    
    def __init__(self):
        self.device = Config.setup_device()
        self.bert_model = None
        self.bert_tokenizer = None
        self.sentence_model = None
        self.colab_models = {}
        self.deep_learning_model = None
        
        # 모델 상태
        self.models_loaded = {
            'bert': False,
            'sentence': False,
            'colab': False,
            'deep_learning': False
        }
        
    def initialize_models(self):
        """모든 모델 초기화"""
        logger.info("🤖 AI 모델 초기화 시작...")
        
        try:
            # BERT 모델 로드
            self.load_bert_model()
            
            # Sentence Transformer 로드
            self.load_sentence_model()
            
            # Colab 훈련 모델 로드
            self.load_colab_models()
            
            # 딥러닝 모델 로드
            self.load_deep_learning_model()
            
            logger.info("✅ AI 모델 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ AI 모델 초기화 실패: {e}")
            self._setup_fallback_models()
    
    def load_bert_model(self):
        """BERT 모델 로드"""
        try:
            logger.info("🔤 BERT 모델 로딩 중...")
            self.bert_tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
            self.bert_model = AutoModel.from_pretrained(Config.BERT_MODEL_NAME)
            self.bert_model.to(self.device)
            self.bert_model.eval()
            self.models_loaded['bert'] = True
            logger.info("✅ BERT 모델 로드 완료")
        except Exception as e:
            logger.warning(f"⚠️ BERT 모델 로드 실패: {e}")
            self.models_loaded['bert'] = False
    
    def load_sentence_model(self):
        """Sentence Transformer 모델 로드"""
        try:
            logger.info("🔤 Sentence Transformer 로딩 중...")
            self.sentence_model = SentenceTransformer(Config.SENTENCE_MODEL_NAME)
            if str(self.device) != 'cpu':
                self.sentence_model = self.sentence_model.to(self.device)
            self.models_loaded['sentence'] = True
            logger.info("✅ Sentence Transformer 로드 완료")
        except Exception as e:
            logger.warning(f"⚠️ Sentence Transformer 로드 실패: {e}")
            self.models_loaded['sentence'] = False
    
    def load_colab_models(self):
        """Colab에서 훈련된 모델들 로드"""
        try:
            logger.info("📚 Colab 모델 로딩 중...")
            
            model_paths = {
                'rf': Config.COLAB_RF_MODEL_PATH,
                'gb': Config.COLAB_GB_MODEL_PATH,
                'complete': Config.COLAB_COMPLETE_MODEL_PATH
            }
            
            loaded_count = 0
            for model_name, model_path in model_paths.items():
                if model_path.exists():
                    try:
                        with open(model_path, 'rb') as f:
                            self.colab_models[model_name] = pickle.load(f)
                        loaded_count += 1
                        logger.info(f"✅ {model_name} 모델 로드 완료")
                    except Exception as e:
                        logger.warning(f"⚠️ {model_name} 모델 로드 실패: {e}")
            
            if loaded_count > 0:
                self.models_loaded['colab'] = True
                logger.info(f"✅ Colab 모델 {loaded_count}개 로드 완료")
            else:
                logger.info("ℹ️ Colab 별도 훈련 모델 파일들이 없습니다. (딥러닝 모델로 대체)")
                self.models_loaded['colab'] = False
                
        except Exception as e:
            logger.error(f"❌ Colab 모델 로드 실패: {e}")
            self.models_loaded['colab'] = False
    
    def load_deep_learning_model(self):
        """딥러닝 모델 로드"""
        try:
            from .deep_learning_engine import get_deep_learning_engine
            
            logger.info("🧠 딥러닝 모델 로딩 중...")
            self.deep_learning_model = get_deep_learning_engine()
            
            if self.deep_learning_model and hasattr(self.deep_learning_model, 'is_loaded'):
                self.models_loaded['deep_learning'] = self.deep_learning_model.is_loaded
                logger.info("✅ 딥러닝 모델 로드 완료")
            else:
                logger.warning("⚠️ 딥러닝 모델 로드 실패")
                
        except Exception as e:
            logger.error(f"❌ 딥러닝 모델 로드 실패: {e}")
            self.models_loaded['deep_learning'] = False
    
    def _setup_fallback_models(self):
        """폴백 모델 설정"""
        logger.info("🔄 폴백 모델 설정 중...")
        
        class FallbackModel:
            def calculate_score(self, text):
                # 간단한 키워드 기반 점수
                keywords = ['지원', '사업', '창업', '기업', '투자', 'R&D']
                score = sum(10 for keyword in keywords if keyword in text)
                return min(score, 100)
            
            def get_model_info(self):
                return {'type': 'fallback', 'status': 'active'}
                
        self.fallback_model = FallbackModel()
    
    def calculate_bert_relevance(self, text: str) -> float:
        """BERT 모델을 사용한 관련성 점수 계산"""
        if not self.models_loaded['bert']:
            return 0.0
            
        try:
            # 토큰화
            inputs = self.bert_tokenizer(
                text, 
                return_tensors='pt', 
                max_length=512, 
                truncation=True, 
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 모델 추론
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # 간단한 점수 계산 (실제로는 더 복잡한 로직 필요)
                score = torch.sigmoid(embeddings.sum()).item() * 100
                
            return min(max(score, 0), 100)
            
        except Exception as e:
            logger.warning(f"BERT 점수 계산 실패: {e}")
            return 0.0
    
    def calculate_sentence_relevance(self, text: str) -> float:
        """Sentence Transformer를 사용한 관련성 점수"""
        if not self.models_loaded['sentence']:
            return 0.0
            
        try:
            # 지원사업 관련 기준 문장들
            reference_sentences = [
                "창업 지원 사업 공고",
                "중소기업 투자 유치 프로그램",
                "기술개발 R&D 지원",
                "벤처기업 육성 사업"
            ]
            
            # 임베딩 계산
            text_embedding = self.sentence_model.encode([text])
            ref_embeddings = self.sentence_model.encode(reference_sentences)
            
            # 코사인 유사도 계산
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(text_embedding, ref_embeddings)
            
            # 최대 유사도를 점수로 사용
            max_similarity = similarities.max()
            score = max_similarity * 100
            
            return min(max(score, 0), 100)
            
        except Exception as e:
            logger.warning(f"Sentence 점수 계산 실패: {e}")
            return 0.0
    
    def calculate_colab_relevance(self, text: str) -> float:
        """Colab 훈련 모델을 사용한 점수 계산"""
        if not self.models_loaded['colab']:
            return 0.0
            
        try:
            # 가장 성능이 좋은 모델 사용 (complete > gb > rf 순)
            model_priority = ['complete', 'gb', 'rf']
            
            for model_name in model_priority:
                if model_name in self.colab_models:
                    model = self.colab_models[model_name]
                    
                    if hasattr(model, 'predict_proba'):
                        # 특성 추출 (간단한 버전)
                        features = self._extract_simple_features(text)
                        
                        # 예측
                        proba = model.predict_proba([features])[0]
                        score = proba[1] * 100 if len(proba) > 1 else proba[0] * 100
                        
                        return min(max(score, 0), 100)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Colab 모델 점수 계산 실패: {e}")
            return 0.0
    
    def calculate_deep_learning_relevance(self, text: str) -> float:
        """딥러닝 모델을 사용한 점수 계산"""
        if not self.models_loaded['deep_learning']:
            return 0.0
            
        try:
            if self.deep_learning_model and hasattr(self.deep_learning_model, 'calculate_score'):
                score = self.deep_learning_model.calculate_score(text)
                return min(max(score, 0), 100)
            return 0.0
            
        except Exception as e:
            logger.warning(f"딥러닝 모델 점수 계산 실패: {e}")
            return 0.0
    
    def _extract_simple_features(self, text: str) -> List[float]:
        """간단한 특성 추출"""
        features = [
            len(text),
            len(text.split()),
            text.count('지원'),
            text.count('사업'),
            text.count('창업'),
            text.count('기업'),
            text.count('투자'),
            text.count('R&D'),
            text.count('개발'),
            text.count('기술')
        ]
        return features
    
    def get_model_status(self) -> Dict[str, Any]:
        """모델 상태 정보 반환"""
        return {
            'models_loaded': self.models_loaded,
            'device': str(self.device),
            'device_name': Config.DEVICE_NAME,
            'colab_models_count': len(self.colab_models),
            'total_models': sum(self.models_loaded.values())
        }
    
    def predict(self, text: str) -> float:
        """통합 예측 점수 계산"""
        scores = []
        weights = []
        
        # BERT 점수
        if self.models_loaded['bert']:
            bert_score = self.calculate_bert_relevance(text)
            scores.append(bert_score)
            weights.append(Config.BERT_WEIGHT)
        
        # Sentence 점수
        if self.models_loaded['sentence']:
            sentence_score = self.calculate_sentence_relevance(text)
            scores.append(sentence_score)
            weights.append(0.3)
        
        # Colab 모델 점수
        if self.models_loaded['colab']:
            colab_score = self.calculate_colab_relevance(text)
            scores.append(colab_score)
            weights.append(Config.COLAB_MODEL_WEIGHT)
        
        # 딥러닝 점수
        if self.models_loaded['deep_learning']:
            dl_score = self.calculate_deep_learning_relevance(text)
            scores.append(dl_score)
            weights.append(0.5)
        
        # 가중 평균 계산
        if scores:
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            return min(max(weighted_score, 0), 100)
        
        # 폴백 모델 사용
        if hasattr(self, 'fallback_model'):
            return self.fallback_model.calculate_score(text)
        
        return 0.0 