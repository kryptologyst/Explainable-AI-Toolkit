# Test file for Modern XAI Toolkit
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Import our modules
from modern_xai import MockDatabase, ModernXAIExplainer, XAIVisualizer
from xai_metrics import XAIMetrics

class TestMockDatabase:
    """Test MockDatabase functionality"""
    
    def test_database_initialization(self):
        """Test database initialization"""
        db = MockDatabase()
        assert len(db.datasets) > 0
        assert 'breast_cancer' in db.datasets
    
    def test_get_dataset(self):
        """Test getting a specific dataset"""
        db = MockDatabase()
        dataset = db.get_dataset('breast_cancer')
        assert dataset is not None
        assert 'data' in dataset
        assert 'target' in dataset
        assert 'feature_names' in dataset
    
    def test_list_datasets(self):
        """Test listing available datasets"""
        db = MockDatabase()
        datasets = db.list_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0

class TestModernXAIExplainer:
    """Test ModernXAIExplainer functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.db = MockDatabase()
        self.dataset = self.db.get_dataset('breast_cancer')
        self.X = self.dataset['data']
        self.y = self.dataset['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.explainer = ModernXAIExplainer()
    
    def test_model_training(self):
        """Test model training"""
        models = self.explainer.train_models(self.X_train, self.y_train, 'breast_cancer')
        assert len(models) > 0
        assert 'breast_cancer_rf' in models
        assert 'breast_cancer_gb' in models
    
    def test_lime_explanation(self):
        """Test LIME explanation generation"""
        models = self.explainer.train_models(self.X_train, self.y_train, 'breast_cancer')
        model = models['breast_cancer_rf']
        
        lime_exp = self.explainer.lime_explanation(
            model, self.X_train, self.X_test, 0,
            self.dataset['feature_names'], self.dataset['target_names']
        )
        assert lime_exp is not None
    
    def test_shap_explanation(self):
        """Test SHAP explanation generation"""
        models = self.explainer.train_models(self.X_train, self.y_train, 'breast_cancer')
        model = models['breast_cancer_rf']
        
        shap_explainer, shap_values = self.explainer.shap_explanation(model, self.X_test)
        assert shap_explainer is not None
        assert shap_values is not None

class TestXAIVisualizer:
    """Test XAIVisualizer functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.db = MockDatabase()
        self.dataset = self.db.get_dataset('breast_cancer')
        self.X = self.dataset['data']
        self.y = self.dataset['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.explainer = ModernXAIExplainer()
        self.visualizer = XAIVisualizer()
    
    def test_plot_lime_explanation(self):
        """Test LIME explanation plotting"""
        models = self.explainer.train_models(self.X_train, self.y_train, 'breast_cancer')
        model = models['breast_cancer_rf']
        
        lime_exp = self.explainer.lime_explanation(
            model, self.X_train, self.X_test, 0,
            self.dataset['feature_names'], self.dataset['target_names']
        )
        
        if lime_exp:
            fig = self.visualizer.plot_lime_explanation(lime_exp)
            assert fig is not None
    
    def test_plot_feature_importance_comparison(self):
        """Test feature importance comparison plotting"""
        models = self.explainer.train_models(self.X_train, self.y_train, 'breast_cancer')
        
        fig = self.visualizer.plot_feature_importance_comparison(
            models, self.dataset['feature_names']
        )
        assert fig is not None

class TestXAIMetrics:
    """Test XAIMetrics functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.metrics = XAIMetrics()
        self.test_explanations = [
            np.random.randn(10),
            np.random.randn(10),
            np.random.randn(10)
        ]
    
    def test_stability_score(self):
        """Test stability score calculation"""
        stability = self.metrics.stability_score(self.test_explanations)
        assert isinstance(stability, float)
        assert 0 <= stability <= 1
    
    def test_explanation_coverage(self):
        """Test explanation coverage calculation"""
        feature_names = [f'feature_{i}' for i in range(20)]
        coverage = self.metrics.explanation_coverage(
            np.random.randn(20), feature_names, top_k=10
        )
        assert isinstance(coverage, float)
        assert 0 <= coverage <= 1

def test_integration():
    """Integration test"""
    # Test full pipeline
    db = MockDatabase()
    dataset = db.get_dataset('breast_cancer')
    
    X = dataset['data']
    y = dataset['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    explainer = ModernXAIExplainer()
    models = explainer.train_models(X_train, y_train, 'breast_cancer')
    
    # Test LIME
    model = models['breast_cancer_rf']
    lime_exp = explainer.lime_explanation(
        model, X_train, X_test, 0,
        dataset['feature_names'], dataset['target_names']
    )
    
    # Test SHAP
    shap_explainer, shap_values = explainer.shap_explanation(model, X_test)
    
    # Test metrics
    metrics = XAIMetrics()
    if lime_exp and shap_values is not None:
        quality_results = metrics.evaluate_explanation_quality(
            lime_exp, shap_values, model, X_test, 0, dataset['feature_names']
        )
        assert isinstance(quality_results, dict)
        assert 'overall_quality' in quality_results

if __name__ == "__main__":
    pytest.main([__file__])
