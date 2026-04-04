#!/usr/bin/env python3
"""
NLP модели для классификации вакансий
Сравнение TF-IDF, FastText и BERT (RuBERT)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Для FastText
try:
    import gensim
    from gensim.models import FastText as GensimFastText
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

# Для BERT
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False


class BaseNLPModel:
    """Базовый класс для NLP моделей"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def fit(self, texts: List[str], y: np.ndarray) -> 'BaseNLPModel':
        raise NotImplementedError
    
    def predict(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError
    
    def evaluate(self, texts: List[str], y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(texts)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
        }


class TFIDFModel(BaseNLPModel):
    """TF-IDF + Logistic Regression"""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        super().__init__("TF-IDF + LogReg")
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    
    def fit(self, texts: List[str], y: np.ndarray) -> 'TFIDFModel':
        print(f"🔧 Обучение {self.name}...")
        
        # Векторизация
        X = self.vectorizer.fit_transform(texts)
        print(f"   TF-IDF признаков: {X.shape[1]}")
        
        # Кодирование меток
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Обучение классификатора
        self.classifier.fit(X, y_encoded)
        self.is_fitted = True
        
        print(f"✅ {self.name} обучен")
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Модель не обучена")
        X = self.vectorizer.transform(texts)
        y_pred = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(y_pred)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)


class FastTextModel(BaseNLPModel):
    """FastText embeddings + Classifier"""
    
    def __init__(self, vector_size: int = 100, classifier_type: str = 'logistic'):
        super().__init__("FastText")
        self.vector_size = vector_size
        self.classifier_type = classifier_type
        self.fasttext_model = None
        self.classifier = None
    
    def _text_to_tokens(self, text: str) -> List[str]:
        """Токенизация текста"""
        import re
        text = text.lower()
        tokens = re.findall(r'\b[a-zа-яё0-9]+\b', text)
        return tokens
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Получение эмбеддинга для текста"""
        tokens = self._text_to_tokens(text)
        if not tokens:
            return np.zeros(self.vector_size)
        
        embeddings = []
        for token in tokens:
            if token in self.fasttext_model.wv:
                embeddings.append(self.fasttext_model.wv[token])
        
        if not embeddings:
            return np.zeros(self.vector_size)
        
        return np.mean(embeddings, axis=0)
    
    def fit(self, texts: List[str], y: np.ndarray) -> 'FastTextModel':
        print(f"🔧 Обучение {self.name}...")
        
        # Токенизация
        tokenized_texts = [self._text_to_tokens(t) for t in texts]
        
        # Обучение FastText
        print(f"   Обучение FastText (vector_size={self.vector_size})...")
        self.fasttext_model = GensimFastText(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=5,
            min_count=1,
            epochs=10,
            seed=42
        )
        
        # Создание признаков
        X = np.array([self._get_embedding(t) for t in texts])
        
        # Кодирование меток
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Обучение классификатора
        if self.classifier_type == 'logistic':
            self.classifier = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        elif self.classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        else:
            self.classifier = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        
        self.classifier.fit(X, y_encoded)
        self.is_fitted = True
        
        print(f"✅ {self.name} обучен")
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Модель не обучена")
        X = np.array([self._get_embedding(t) for t in texts])
        y_pred = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(y_pred)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = np.array([self._get_embedding(t) for t in texts])
        return self.classifier.predict_proba(X)


class BERTModel(BaseNLPModel):
    """RuBERT embeddings + Classifier"""
    
    def __init__(self, model_name: str = 'cointegrated/rubert-tiny2', device: str = 'cpu'):
        super().__init__("RuBERT")
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.bert_model = None
        self.classifier = None
        self.batch_size = 32
    
    def _load_bert(self):
        """Загрузка BERT модели"""
        if self.tokenizer is None:
            print(f"   Загрузка {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.bert_model = AutoModel.from_pretrained(self.model_name)
            self.bert_model.to(self.device)
            self.bert_model.eval()
    
    @torch.no_grad()
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Получение BERT эмбеддингов"""
        self._load_bert()
        
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Токенизация
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Получение эмбеддингов
            outputs = self.bert_model(**encoded)
            
            # CLS токен или mean pooling
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def fit(self, texts: List[str], y: np.ndarray) -> 'BERTModel':
        print(f"🔧 Обучение {self.name}...")
        
        # Получение BERT эмбеддингов
        X = self._get_embeddings(texts)
        print(f"   BERT эмбеддингов: {X.shape[1]}")
        
        # Кодирование меток
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Обучение классификатора
        self.classifier = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight='balanced',
            random_state=42
        )
        self.classifier.fit(X, y_encoded)
        self.is_fitted = True
        
        print(f"✅ {self.name} обучен")
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Модель не обучена")
        X = self._get_embeddings(texts)
        y_pred = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(y_pred)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self._get_embeddings(texts)
        return self.classifier.predict_proba(X)


class NLPModelComparator:
    """Сравнение NLP моделей"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def add_model(self, model: BaseNLPModel) -> None:
        """Добавление модели"""
        self.models[model.name] = model
    
    def compare(
        self,
        train_texts: List[str],
        test_texts: List[str],
        y_train: np.ndarray,
        y_test: np.ndarray,
        cv: int = 3
    ) -> pd.DataFrame:
        """Сравнение всех моделей"""
        print("=" * 60)
        print("📊 СРАВНЕНИЕ NLP МОДЕЛЕЙ")
        print("=" * 60)
        
        results = []
        
        for name, model in self.models.items():
            print(f"\n{'='*40}")
            
            # Обучение
            model.fit(train_texts, y_train)
            
            # Оценка на train
            train_metrics = model.evaluate(train_texts, y_train)
            
            # Оценка на test
            test_metrics = model.evaluate(test_texts, y_test)
            
            results.append({
                'Model': name,
                'Train_Accuracy': train_metrics['accuracy'],
                'Train_F1': train_metrics['f1'],
                'Test_Accuracy': test_metrics['accuracy'],
                'Test_F1': test_metrics['f1'],
                'Train_Precision': train_metrics['precision'],
                'Test_Precision': test_metrics['precision'],
                'Train_Recall': train_metrics['recall'],
                'Test_Recall': test_metrics['recall']
            })
        
        self.results = pd.DataFrame(results).sort_values('Test_F1', ascending=False)
        
        print("\n" + "=" * 60)
        print("📋 РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
        print("=" * 60)
        print(self.results.to_string(index=False))
        
        return self.results
    
    def get_best_model(self, metric: str = 'Test_F1') -> Tuple[str, BaseNLPModel]:
        """Получение лучшей модели"""
        if self.results.empty:
            raise ValueError("Сначала выполните compare()")
        
        best_name = self.results.sort_values(metric, ascending=False).iloc[0]['Model']
        return best_name, self.models[best_name]


def prepare_text_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Подготовка текстовых данных
    
    ВАЖНО: НЕ используем title, так как он содержит название профессии,
    что приводит к утечке данных (модель просто находит ключевое слово)
    """
    
    # Объединяем только requirements и responsibilities (БЕЗ title!)
    df['full_text'] = (
        df['requirements'].fillna('') + ' ' +
        df['responsibilities'].fillna('')
    )
    
    # Целевая переменная
    y = df['Job'].values
    
    return df, y


if __name__ == "__main__":
    print("NLP Models module loaded")
    print(f"BERT available: {BERT_AVAILABLE}")
    print(f"Gensim available: {GENSIM_AVAILABLE}")