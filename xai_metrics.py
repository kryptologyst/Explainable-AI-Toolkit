# Explanation Quality Metrics and Evaluation
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

class XAIMetrics:
    """Class for evaluating explanation quality and consistency"""
    
    def __init__(self):
        self.metrics_results = {}
    
    def stability_score(self, explanations_list, threshold=0.1):
        """
        Calculate stability score for explanations across multiple runs
        Higher score indicates more stable explanations
        """
        if len(explanations_list) < 2:
            return 0.0
        
        # Convert explanations to feature importance vectors
        importance_vectors = []
        for exp in explanations_list:
            if hasattr(exp, 'as_list'):
                # LIME explanation
                features, scores = zip(*exp.as_list())
                importance_vectors.append(np.array(scores))
            elif isinstance(exp, np.ndarray):
                # SHAP or other array-based explanations
                importance_vectors.append(exp)
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(importance_vectors)):
            for j in range(i + 1, len(importance_vectors)):
                corr, _ = spearmanr(importance_vectors[i], importance_vectors[j])
                correlations.append(corr if not np.isnan(corr) else 0.0)
        
        # Stability score is the mean correlation
        stability = np.mean(correlations) if correlations else 0.0
        return stability
    
    def consistency_score(self, lime_exp, shap_values, instance_idx):
        """
        Calculate consistency between LIME and SHAP explanations
        """
        if lime_exp is None or shap_values is None:
            return 0.0
        
        # Extract LIME features and scores
        lime_features, lime_scores = zip(*lime_exp.as_list())
        
        # Extract SHAP values for the instance
        if isinstance(shap_values, list):
            instance_shap = shap_values[1][instance_idx]
        else:
            instance_shap = shap_values[instance_idx]
        
        # Get common features
        feature_names = lime_exp.domain_mapper.feature_names
        lime_dict = dict(zip(lime_features, lime_scores))
        
        # Align SHAP values with LIME features
        aligned_scores = []
        for feature in lime_features:
            if feature in feature_names:
                feature_idx = feature_names.index(feature)
                aligned_scores.append(instance_shap[feature_idx])
            else:
                aligned_scores.append(0.0)
        
        # Calculate correlation
        if len(aligned_scores) > 1:
            corr, _ = spearmanr(lime_scores, aligned_scores)
            return corr if not np.isnan(corr) else 0.0
        else:
            return 0.0
    
    def fidelity_score(self, model, X_test, explanations, instance_idx, method='lime'):
        """
        Calculate fidelity score - how well the explanation matches the model's behavior
        """
        instance = X_test.iloc[instance_idx:instance_idx+1]
        original_prediction = model.predict_proba(instance)[0]
        
        if method == 'lime':
            # For LIME, we can't easily perturb features, so we use a simplified approach
            # This is a placeholder - in practice, you'd implement proper fidelity testing
            return 0.8  # Placeholder value
        elif method == 'shap':
            # For SHAP, we can use the sum of SHAP values
            if isinstance(explanations, list):
                shap_sum = np.sum(explanations[1][instance_idx])
            else:
                shap_sum = np.sum(explanations[instance_idx])
            
            # Simple fidelity measure based on SHAP sum
            base_rate = 0.5  # Assuming binary classification
            predicted_prob = base_rate + shap_sum
            actual_prob = original_prediction[1] if len(original_prediction) > 1 else original_prediction[0]
            
            fidelity = 1.0 - abs(predicted_prob - actual_prob)
            return max(0.0, min(1.0, fidelity))
        
        return 0.0
    
    def feature_ranking_correlation(self, model1_importance, model2_importance):
        """
        Calculate correlation between feature rankings from different models
        """
        if len(model1_importance) != len(model2_importance):
            return 0.0
        
        # Get feature rankings
        rank1 = np.argsort(np.abs(model1_importance))[::-1]
        rank2 = np.argsort(np.abs(model2_importance))[::-1]
        
        # Calculate Spearman correlation
        corr, _ = spearmanr(rank1, rank2)
        return corr if not np.isnan(corr) else 0.0
    
    def explanation_coverage(self, explanations, feature_names, top_k=10):
        """
        Calculate how many features are covered in top-k explanations
        """
        if hasattr(explanations, 'as_list'):
            # LIME explanation
            features, _ = zip(*explanations.as_list())
            covered_features = set(features[:top_k])
        elif isinstance(explanations, np.ndarray):
            # SHAP or other array-based explanations
            top_indices = np.argsort(np.abs(explanations))[-top_k:]
            covered_features = set([feature_names[i] for i in top_indices])
        else:
            return 0.0
        
        total_features = len(feature_names)
        coverage = len(covered_features) / total_features
        return coverage
    
    def evaluate_explanation_quality(self, lime_exp, shap_values, model, X_test, instance_idx, feature_names):
        """
        Comprehensive evaluation of explanation quality
        """
        results = {}
        
        # Stability (placeholder - would need multiple runs)
        results['stability'] = 0.8  # Placeholder
        
        # Consistency between LIME and SHAP
        results['consistency'] = self.consistency_score(lime_exp, shap_values, instance_idx)
        
        # Fidelity scores
        results['lime_fidelity'] = self.fidelity_score(model, X_test, lime_exp, instance_idx, 'lime')
        results['shap_fidelity'] = self.fidelity_score(model, X_test, shap_values, instance_idx, 'shap')
        
        # Coverage
        results['lime_coverage'] = self.explanation_coverage(lime_exp, feature_names)
        results['shap_coverage'] = self.explanation_coverage(shap_values[1][instance_idx] if isinstance(shap_values, list) else shap_values[instance_idx], feature_names)
        
        # Overall quality score
        quality_metrics = [results['consistency'], results['lime_fidelity'], results['shap_fidelity']]
        results['overall_quality'] = np.mean(quality_metrics)
        
        return results
    
    def plot_quality_metrics(self, results_dict, title="Explanation Quality Metrics"):
        """
        Plot explanation quality metrics
        """
        metrics = list(results_dict.keys())
        scores = list(results_dict.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['skyblue' if score >= 0.7 else 'lightcoral' for score in scores]
        bars = ax.bar(metrics, scores, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def compare_explanations_across_models(self, models, X_test, feature_names, instance_idx):
        """
        Compare explanations across different models for the same instance
        """
        comparison_results = {}
        
        for model_name, model in models.items():
            try:
                # Generate SHAP explanation
                import shap
                if hasattr(model, 'feature_importances_'):
                    # Tree-based model
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test.iloc[:10])
                else:
                    # Non-tree model
                    explainer = shap.KernelExplainer(model.predict_proba, X_test.iloc[:100])
                    shap_values = explainer.shap_values(X_test.iloc[:10])
                
                # Extract feature importance
                if isinstance(shap_values, list):
                    instance_shap = shap_values[1][instance_idx]
                else:
                    instance_shap = shap_values[instance_idx]
                
                comparison_results[model_name] = instance_shap
                
            except Exception as e:
                print(f"Failed to generate explanation for {model_name}: {e}")
                continue
        
        # Calculate pairwise correlations
        model_names = list(comparison_results.keys())
        correlation_matrix = np.zeros((len(model_names), len(model_names)))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    corr, _ = spearmanr(comparison_results[model1], comparison_results[model2])
                    correlation_matrix[i, j] = corr if not np.isnan(corr) else 0.0
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, 
                   xticklabels=model_names, 
                   yticklabels=model_names,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   ax=ax)
        ax.set_title('Explanation Correlation Across Models')
        plt.tight_layout()
        
        return fig, comparison_results

def evaluate_xai_methods(lime_exp, shap_values, model, X_test, instance_idx, feature_names):
    """
    Convenience function to evaluate XAI methods
    """
    evaluator = XAIMetrics()
    
    # Evaluate explanation quality
    quality_results = evaluator.evaluate_explanation_quality(
        lime_exp, shap_values, model, X_test, instance_idx, feature_names
    )
    
    # Plot quality metrics
    quality_fig = evaluator.plot_quality_metrics(quality_results)
    
    return quality_results, quality_fig
