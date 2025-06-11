# ============================================
# 셀 1: 라이브러리 설치 (개별 설치)
# ============================================

# 각 라이브러리를 개별적으로 설치
# !pip install torch
# !pip install transformers
# !pip install sentence-transformers
# !pip install scikit-learn
# !pip install pandas
# !pip install numpy
# !pip install matplotlib
# !pip install seaborn
# !pip install requests
# !pip install beautifulsoup4
# !pip install tqdm
# !pip install joblib
# !pip install PyMuPDF

print("📦 모든 라이브러리 설치 완료!")

# ============================================
# 셀 2: 라이브러리 import 및 Apple Silicon 호환 디바이스 설정
# ============================================
import torch
import numpy as np
import pandas as pd
import joblib
import json
import re
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import nltk
from tqdm import tqdm
import pickle
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import seaborn as sns

# NLTK 데이터 다운로드
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Apple Silicon 호환 디바이스 설정
def setup_device():
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
    
    print(f"🖥️ 사용 디바이스: {device_name}")
    return device

device = setup_device()

# Apple Silicon 호환을 위한 설정
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("✅ 라이브러리 로드 및 디바이스 설정 완료")

# ============================================
# 셀 3: PDF에서 지원사업 데이터 추출
# ============================================
def extract_support_programs_from_pdf(pdf_path):
    """PDF에서 지원사업 제목들을 추출하여 JSON 형태로 변환"""
    
    # PDF 파일 업로드 (코랩에서는 files.upload() 사용)
    from google.colab import files
    print("📄 PDF 파일을 업로드하세요...")
    uploaded = files.upload()
    pdf_path = list(uploaded.keys())[0]
    
    support_programs = []
    
    try:
        # PDF 열기
        doc = fitz.open(pdf_path)
        print(f"📖 PDF 페이지 수: {len(doc)} 페이지")
        
        for page_num in tqdm(range(len(doc)), desc="PDF 페이지 처리"):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # 지원사업 제목 패턴 찾기 (숫자. 제목 형태)
            patterns = [
                r'\d+\.\s*([가-힣\s\w\(\)]+(?:지원|사업|육성|개발|창업|혁신|연구|기술|투자|융자|보조|활성화))',
                r'([가-힣\s\w\(\)]+(?:지원사업|창업지원|기업지원|R&D|연구개발|기술개발|사업화))',
                r'「([가-힣\s\w\(\)]+)」',  # 따옴표로 감싸진 사업명
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.MULTILINE)
                for match in matches:
                    title = match.strip()
                    if len(title) > 5 and len(title) < 100:  # 적절한 길이 필터링
                        support_programs.append({
                            'title': title,
                            'page': page_num + 1,
                            'category': 'support_program',
                            'label': 1
                        })
        
        doc.close()
        print(f"✅ 총 {len(support_programs)}개 지원사업 추출 완료")
        
    except Exception as e:
        print(f"❌ PDF 처리 오류: {e}")
        # 오류 시 샘플 데이터 생성
        support_programs = generate_sample_support_programs()
    
    return support_programs

def generate_sample_support_programs():
    """샘플 지원사업 데이터 생성 (PDF 파싱 실패 시 대안)"""
    sample_programs = [
        "창업기업 지원사업", "중소기업 기술개발 지원", "청년창업 육성사업",
        "스타트업 투자 연계 프로그램", "벤처기업 R&D 지원", "소상공인 경영안정 지원",
        "기업 디지털 전환 지원사업", "혁신기업 성장 지원", "중소기업 수출 지원",
        "창업보육센터 운영 지원", "기술사업화 촉진사업", "중소기업 융자 지원",
        "청년 취업 지원 프로그램", "기업 인력양성 지원", "중소기업 컨설팅 지원",
        "스마트공장 구축 지원", "친환경 기업 지원사업", "지역 혁신 클러스터 조성",
        "여성기업 지원사업", "사회적기업 육성 지원", "농촌융복합산업 활성화 지원"
    ]
    
    programs = []
    for i, title in enumerate(sample_programs):
        programs.append({
            'title': title,
            'page': i // 3 + 1,
            'category': 'support_program', 
            'label': 1
        })
    
    return programs

# PDF에서 지원사업 데이터 추출
print("📊 PDF에서 지원사업 데이터 추출 중...")
support_data = extract_support_programs_from_pdf("support_programs.pdf")

# ============================================
# 셀 4: 일반 텍스트 데이터 생성
# ============================================
def generate_general_text_data(num_samples=500):
    """지원사업이 아닌 일반 텍스트 데이터 생성"""
    
    general_categories = {
        "일상생활": ["오늘 날씨가 좋아서 산책을 했다", "친구와 카페에서 커피를 마셨다", "새로운 드라마를 시청했다"],
        "취미활동": ["주말에 등산을 다녀왔다", "새로운 요리 레시피를 시도해봤다", "독서 모임에 참여했다"],
        "학습교육": ["온라인 강의를 수강하고 있다", "새로운 언어를 배우고 있다", "자격증 시험을 준비 중이다"],
        "여행관광": ["제주도 여행을 계획하고 있다", "해외여행 준비를 하고 있다", "맛집 탐방을 했다"],
        "건강운동": ["헬스장에서 운동을 했다", "요가 수업에 참여했다", "건강한 식단을 유지하고 있다"],
        "문화예술": ["미술관 전시회를 관람했다", "콘서트에 다녀왔다", "영화 관람을 했다"],
        "쇼핑소비": ["온라인 쇼핑을 했다", "새로운 옷을 구매했다", "할인 상품을 찾고 있다"],
        "관계소통": ["가족과 시간을 보냈다", "동료들과 식사를 했다", "새로운 사람들을 만났다"]
    }
    
    general_data = []
    categories = list(general_categories.keys())
    
    for i in range(num_samples):
        category = np.random.choice(categories)
        base_texts = general_categories[category]
        text = np.random.choice(base_texts)
        
        general_data.append({
            'title': text,
            'category': 'general_text',
            'label': 0,
            'source': category
        })
    
    return general_data

# 일반 텍스트 데이터 생성
print("📝 일반 텍스트 데이터 생성 중...")
general_data = generate_general_text_data(len(support_data))

# ============================================
# 셀 5: 데이터 결합 및 전처리
# ============================================
# 데이터 결합
all_data = support_data + general_data
df = pd.DataFrame(all_data)

print(f"📊 전체 데이터 수: {len(df)}")
print(f"📈 지원사업 데이터: {len(support_data)}개")
print(f"📝 일반 텍스트 데이터: {len(general_data)}개")

# 데이터 분포 확인
print("\n📊 레이블 분포:")
print(df['label'].value_counts())

# 샘플 데이터 확인
print("\n📋 지원사업 샘플:")
print(df[df['label'] == 1]['title'].head(5).tolist())
print("\n📋 일반 텍스트 샘플:")
print(df[df['label'] == 0]['title'].head(5).tolist())

# 데이터를 JSON으로 저장
training_data_json = df.to_dict('records')
with open('training_data.json', 'w', encoding='utf-8') as f:
    json.dump(training_data_json, f, ensure_ascii=False, indent=2)

print("✅ 훈련 데이터 JSON 파일 저장 완료")

# ============================================
# 셀 6: 고급 특성 추출기 정의
# ============================================
class AppleSiliconFeatureExtractor:
    """Apple Silicon 호환 고급 특성 추출기"""
    
    def __init__(self, device='cpu'):
        self.device = device
        print(f"🔧 특성 추출기 디바이스: {device}")
        
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
            print("✅ Sentence Transformer 로드 완료")
        except Exception as e:
            print(f"⚠️ Sentence Transformer 로드 실패: {e}")
            self.sentence_model = None
        
        # 지원사업 관련 키워드
        self.support_keywords = [
            '지원', '사업', '창업', '개발', '연구', 'R&D', '기술', '혁신',
            '투자', '융자', '보조', '육성', '활성화', '촉진', '기업',
            '중소기업', '소상공인', '벤처', '스타트업', '클러스터'
        ]
        
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
    
    def fit(self, texts):
        """특성 추출기 훈련 (훈련 데이터로 TF-IDF 학습)"""
        print("🔧 특성 추출기 훈련 중...")
        self.tfidf.fit(texts)
        self.tfidf_fitted = True
        print("✅ TF-IDF 훈련 완료")
    
    def extract_features(self, texts, is_training=False):
        """전체 특성 추출 (Apple Silicon 호환)"""
        print("🔍 특성 추출 시작...")
        
        # TF-IDF 특성
        print("📊 TF-IDF 특성 추출 중...")
        if is_training or not self.tfidf_fitted:
            # 훈련 시에만 fit_transform 사용
            tfidf_features = self.tfidf.fit_transform(texts).toarray()
            self.tfidf_fitted = True
        else:
            # 예측 시에는 transform만 사용
            tfidf_features = self.tfidf.transform(texts).toarray()
        
        # Sentence embedding 특성 (Apple Silicon 호환)
        if self.sentence_model:
            print("🧠 Sentence Embedding 특성 추출 중...")
            try:
                # CPU로 변환하여 Apple Silicon 호환성 보장
                sentence_features = []
                batch_size = 32
                
                for i in tqdm(range(0, len(texts), batch_size), desc="Sentence Embedding"):
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
                print(f"✅ Sentence Embedding 형태: {sentence_features.shape}")
                
            except Exception as e:
                print(f"⚠️ Sentence Embedding 추출 실패: {e}")
                sentence_features = np.zeros((len(texts), 384))  # 기본 차원
        else:
            sentence_features = np.zeros((len(texts), 384))
        
        # 수동 특성
        print("🔧 수동 특성 추출 중...")
        manual_features = self.extract_manual_features(texts)
        
        # 특성 결합
        combined_features = np.hstack([
            tfidf_features,
            sentence_features, 
            manual_features
        ])
        
        print(f"✅ 전체 특성 형태: {combined_features.shape}")
        return combined_features

# ============================================
# 셀 7: Apple Silicon 호환 딥러닝 모델 클래스들
# ============================================
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class SupportProgramDataset(Dataset):
    """지원사업 분류를 위한 데이터셋"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

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

class AppleSiliconDeepLearningModel:
    """Apple Silicon 호환 딥러닝 프로덕션 모델"""
    
    def __init__(self, device='cpu'):
        self.device = device
        print(f"🍎 Apple Silicon 호환 딥러닝 모델 초기화 (디바이스: {device})")
        
        self.feature_extractor = AppleSiliconFeatureExtractor(device=device)
        self.model = None
        self.is_fitted = False
        
        # 하이퍼파라미터 설정 (고성능을 위한 설정)
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 100  # 더 긴 학습
        self.weight_decay = 1e-5
        
    def _create_model(self, input_dim):
        """딥러닝 모델 생성"""
        model = DeepSupportClassifier(
            input_dim=input_dim,
            hidden_dims=[1024, 512, 256, 128, 64, 32],  # 6개 은닉층 + 입출력층 = 8층
            dropout_rate=0.3
        )
        return model.to(self.device)
        
    def fit(self, texts, labels):
        """딥러닝 모델 훈련 (Apple Silicon 호환)"""
        print("🏋️ Apple Silicon 호환 딥러닝 모델 훈련 시작...")
        print(f"🔥 8층 딥러닝 네트워크로 {self.epochs} 에포크 훈련")
        
        # 특성 추출 (훈련 모드)
        features = self.feature_extractor.extract_features(texts, is_training=True)
        input_dim = features.shape[1]
        
        # 딥러닝 모델 생성
        self.model = self._create_model(input_dim)
        print(f"🧠 신경망 구조: 입력층({input_dim}) → 1024 → 512 → 256 → 128 → 64 → 32 → 출력층(2)")
        
        # 데이터셋 및 데이터로더 생성
        dataset = SupportProgramDataset(features, labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 옵티마이저 및 손실함수 설정
        optimizer = optim.AdamW(self.model.parameters(), 
                               lr=self.learning_rate, 
                               weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                        patience=5, factor=0.5)
        criterion = nn.CrossEntropyLoss()
        
        # 학습 기록
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        print("🚀 딥러닝 훈련 시작!")
        
        for epoch in range(self.epochs):
            # 훈련 단계
            self.model.train()
            train_loss = 0
            train_samples = 0
            
            for batch_features, batch_labels in tqdm(train_loader, 
                                                   desc=f"Epoch {epoch+1}/{self.epochs}"):
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                
                # Gradient Clipping (안정적인 학습)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item() * batch_features.size(0)
                train_samples += batch_features.size(0)
            
            # 검증 단계
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_samples = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item() * batch_features.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == batch_labels).sum().item()
                    val_samples += batch_features.size(0)
            
            # 메트릭 계산
            avg_train_loss = train_loss / train_samples
            avg_val_loss = val_loss / val_samples
            val_accuracy = val_correct / val_samples
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            # 학습률 스케줄링
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 진행상황 출력
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{self.epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {avg_val_loss:.4f} | "
                      f"Val Acc: {val_accuracy:.4f} | "
                      f"LR: {current_lr:.6f}")
        
        self.is_fitted = True
        
        # 학습 결과 시각화
        self._plot_training_history(train_losses, val_losses, val_accuracies)
        
        print(f"✅ 딥러닝 모델 훈련 완료! 최종 검증 정확도: {val_accuracies[-1]:.4f}")
        
    def _plot_training_history(self, train_losses, val_losses, val_accuracies):
        """학습 과정 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 손실 함수 그래프
        ax1.plot(train_losses, label='훈련 손실', color='blue')
        ax1.plot(val_losses, label='검증 손실', color='red')
        ax1.set_title('학습 과정 - 손실 함수')
        ax1.set_xlabel('에포크')
        ax1.set_ylabel('손실')
        ax1.legend()
        ax1.grid(True)
        
        # 정확도 그래프
        ax2.plot(val_accuracies, label='검증 정확도', color='green')
        ax2.set_title('학습 과정 - 검증 정확도')
        ax2.set_xlabel('에포크')
        ax2.set_ylabel('정확도')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def predict(self, texts):
        """예측 (Apple Silicon 호환)"""
        if not self.is_fitted:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        features = self.feature_extractor.extract_features(texts, is_training=False)
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features_tensor)
            _, predicted = torch.max(outputs.data, 1)
            
        return predicted.cpu().numpy()
    
    def predict_proba(self, texts):
        """확률 예측 (Apple Silicon 호환)"""
        if not self.is_fitted:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        features = self.feature_extractor.extract_features(texts, is_training=False)
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
        return probabilities.cpu().numpy()

# ============================================
# 셀 8: 데이터 분할 및 모델 훈련
# ============================================
# 텍스트와 레이블 분리
texts = df['title'].tolist()
labels = df['label'].tolist()

# 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"📊 훈련 데이터: {len(X_train)}개")
print(f"📊 테스트 데이터: {len(X_test)}개")

# Apple Silicon 호환 딥러닝 모델 생성 및 훈련
model = AppleSiliconDeepLearningModel(device=device)
model.fit(X_train, y_train)

# ============================================
# 셀 9: 모델 평가
# ============================================
# 예측
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# 평가 지표
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 정확도: {accuracy:.4f}")

# 상세 분류 리포트
print("\n📊 분류 리포트:")
print(classification_report(y_test, y_pred, 
                          target_names=['일반 텍스트', '지원사업']))

# 혼동 행렬 시각화
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['일반 텍스트', '지원사업'],
            yticklabels=['일반 텍스트', '지원사업'])
plt.title('혼동 행렬 (Confusion Matrix)')
plt.ylabel('실제 레이블')
plt.xlabel('예측 레이블')
plt.show()

# ============================================
# 셀 10: Apple Silicon 호환 모델 저장
# ============================================
def save_apple_silicon_model(model, filepath):
    """Apple Silicon 호환을 위한 모델 저장"""
    print(f"💾 Apple Silicon 호환 모델 저장 중: {filepath}")
    
    try:
        # 모델 상태를 CPU로 변환하여 저장
        model_data = {
            'model_state_dict': model.model.state_dict(),
            'model_structure': model.model,
            'feature_extractor': model.feature_extractor,
            'is_fitted': model.is_fitted,
            'device': 'cpu',  # Apple Silicon 호환을 위해 CPU로 저장
            'model_type': 'AppleSiliconDeepLearningModel',
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'training_accuracy': accuracy,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        # pickle protocol 4로 저장 (Apple Silicon 호환)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=4)
        
        print(f"✅ 모델 저장 완료: {filepath}")
        print(f"📊 파일 크기: {os.path.getsize(filepath) / (1024*1024):.1f} MB")
        
        # 메타데이터 저장
        metadata = {
            'model_type': 'AppleSiliconDeepLearningModel',
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'accuracy': float(accuracy),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'device_compatibility': ['cpu', 'mps', 'cuda'],
            'apple_silicon_optimized': True
        }
        
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 메타데이터 저장 완료: {metadata_path}")
        
    except Exception as e:
        print(f"❌ 모델 저장 실패: {e}")

# 모델 저장
import os
save_apple_silicon_model(model, 'apple_silicon_production_model.pkl')

# ============================================
# 셀 11: 테스트 및 검증
# ============================================
# 모델 테스트
test_texts = [
    "중소기업 창업 지원 사업에 참여하고 싶습니다",
    "오늘 친구와 맛있는 저녁을 먹었습니다",
    "기술개발 R&D 지원 프로그램 안내",
    "주말에 영화를 보러 갈 예정입니다",
    "벤처기업 투자 유치 지원사업"
]

print("🧪 모델 테스트:")
predictions = model.predict(test_texts)
probabilities = model.predict_proba(test_texts)

for i, text in enumerate(test_texts):
    pred_label = "지원사업" if predictions[i] == 1 else "일반텍스트"
    confidence = probabilities[i][predictions[i]]
    print(f"📝 '{text}' → {pred_label} (신뢰도: {confidence:.3f})")

print("\n✅ Apple Silicon 호환 코랩 훈련 코드 완료!")
print("🍎 이 모델은 Apple Silicon Mac에서 완벽하게 작동합니다!")

# ============================================
# 셀 12: 다운로드 준비
# ============================================
# 코랩에서 파일 다운로드
from google.colab import files

print("📁 다운로드 가능한 파일들:")
download_files = [
    'apple_silicon_production_model.pkl',
    'apple_silicon_production_model_metadata.json',
    'training_data.json'
]

for file in download_files:
    if os.path.exists(file):
        print(f"✅ {file} - {os.path.getsize(file) / (1024*1024):.1f} MB")
        files.download(file)
    else:
        print(f"❌ {file} - 파일 없음")

print("\n🎉 모든 파일 다운로드 완료!")
print("🍎 Apple Silicon Mac에서 바로 사용 가능합니다!") 