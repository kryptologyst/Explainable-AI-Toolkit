# Simplified XAI Demo - Works with basic packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Basic ML libraries
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Try to import XAI libraries, but continue if they fail
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️  SHAP not available - skipping SHAP explanations")
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    print("⚠️  LIME not available - skipping LIME explanations")
    LIME_AVAILABLE = False

class SimpleXAIExplainer:
    """Simplified XAI explainer that works with basic packages"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def train_models(self, X_train, y_train, dataset_name):
        """Train multiple model types for comparison"""
        
        # Scale data for neural networks
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers[dataset_name] = scaler
        
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        self.models[f'{dataset_name}_rf'] = rf_model
        
        # Gradient Boosting
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        self.models[f'{dataset_name}_gb'] = gb_model
        
        # Neural Network
        nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        nn_model.fit(X_train_scaled, y_train)
        self.models[f'{dataset_name}_nn'] = nn_model
        
        return self.models
    
    def lime_explanation(self, model, X_train, X_test, instance_idx, feature_names, target_names):
        """Generate LIME explanation if available"""
        if not LIME_AVAILABLE:
            return None
            
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.array(X_train),
                feature_names=feature_names,
                class_names=target_names,
                mode="classification",
                random_state=42
            )
            
            instance = X_test.iloc[instance_idx:instance_idx+1].values[0]
            exp = explainer.explain_instance(
                instance, 
                model.predict_proba, 
                num_features=min(10, len(feature_names))
            )
            
            return exp
        except Exception as e:
            print(f"LIME explanation failed: {e}")
            return None
    
    def shap_explanation(self, model, X_test):
        """Generate SHAP explanation if available"""
        if not SHAP_AVAILABLE:
            return None, None
            
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test.iloc[:20])  # Use smaller sample
            return explainer, shap_values
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return None, None
    
    def feature_importance_analysis(self, models, feature_names):
        """Analyze feature importance across models"""
        importance_data = {}
        
        for model_name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance_data[model_name] = model.feature_importances_
        
        return importance_data

def plot_model_performance(models, X_test, y_test, scalers, dataset_name):
    """Plot model performance comparison"""
    performance_data = []
    
    for model_name, model in models.items():
        if 'nn' in model_name:
            X_test_scaled = scalers[dataset_name].transform(X_test)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        performance_data.append({
            'Model': model_name.replace(f'{dataset_name}_', ''),
            'Accuracy': accuracy
        })
    
    performance_df = pd.DataFrame(performance_data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(performance_df['Model'], performance_df['Accuracy'], 
                  color=['skyblue', 'lightcoral', 'lightgreen'])
    
    # Add value labels on bars
    for bar, acc in zip(bars, performance_df['Accuracy']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{acc:.3f}', ha='center', va='bottom')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance Comparison')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig, performance_df

def plot_feature_importance(importance_data, feature_names):
    """Plot feature importance comparison"""
    if not importance_data:
        return None
    
    fig, axes = plt.subplots(1, len(importance_data), figsize=(15, 5))
    if len(importance_data) == 1:
        axes = [axes]
    
    for i, (model_name, importances) in enumerate(importance_data.items()):
        # Get top 10 features
        top_indices = np.argsort(importances)[-10:]
        top_features = [feature_names[j] for j in top_indices]
        top_importances = importances[top_indices]
        
        axes[i].barh(range(len(top_features)), top_importances)
        axes[i].set_yticks(range(len(top_features)))
        axes[i].set_yticklabels(top_features)
        axes[i].set_xlabel('Importance')
        axes[i].set_title(f'{model_name.replace("_", " ").title()} Feature Importance')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_dataset_analysis(dataset):
    """Plot dataset analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Class distribution
    class_counts = pd.Series(dataset['target']).value_counts()
    axes[0, 0].pie(class_counts.values, labels=dataset['target_names'], autopct='%1.1f%%')
    axes[0, 0].set_title('Class Distribution')
    
    # Feature correlation heatmap (top 10 features)
    corr_matrix = dataset['data'].corr()
    top_features = corr_matrix.abs().sum().nlargest(10).index
    sns.heatmap(corr_matrix.loc[top_features, top_features], 
                annot=True, cmap='coolwarm', ax=axes[0, 1])
    axes[0, 1].set_title('Feature Correlation (Top 10)')
    
    # Feature distributions
    dataset['data'].iloc[:, :5].hist(ax=axes[1, 0], bins=20, alpha=0.7)
    axes[1, 0].set_title('Feature Distributions')
    
    # Box plots for top features by variance
    top_var_features = dataset['data'].std().nlargest(5).index
    dataset['data'][top_var_features].boxplot(ax=axes[1, 1])
    axes[1, 1].set_title('Top 5 Features by Variance')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def main():
    """Main function to demonstrate simplified XAI techniques"""
    
    print("🚀 Starting Simplified XAI Demonstration")
    print("=" * 60)
    
    # Load dataset
    data = load_breast_cancer()
    dataset = {
        'data': pd.DataFrame(data.data, columns=data.feature_names),
        'target': data.target,
        'target_names': data.target_names,
        'feature_names': data.feature_names,
        'description': 'Breast cancer classification dataset'
    }
    
    print(f"📊 Using dataset: {dataset['description']}")
    print(f"📈 Dataset shape: {dataset['data'].shape}")
    print(f"🎯 Classes: {dataset['target_names']}")
    
    # Prepare data
    X = dataset['data']
    y = dataset['target']
    feature_names = dataset['feature_names']
    target_names = dataset['target_names']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"📈 Training data shape: {X_train.shape}")
    print(f"📈 Test data shape: {X_test.shape}")
    
    # Initialize explainer and train models
    explainer = SimpleXAIExplainer()
    print("\n🤖 Training multiple models...")
    models = explainer.train_models(X_train, y_train, 'breast_cancer')
    
    # Evaluate models
    print("\n📊 Model Performance:")
    perf_fig, performance_df = plot_model_performance(models, X_test, y_test, explainer.scalers, 'breast_cancer')
    print(performance_df.to_string(index=False))
    plt.show()
    
    # Dataset analysis
    print("\n📊 Dataset Analysis:")
    dataset_fig = plot_dataset_analysis(dataset)
    plt.show()
    
    # Feature importance analysis
    print("\n📈 Feature Importance Analysis:")
    importance_data = explainer.feature_importance_analysis(models, feature_names)
    if importance_data:
        importance_fig = plot_feature_importance(importance_data, feature_names)
        if importance_fig:
            plt.show()
    
    # Generate explanations for a specific instance
    instance_idx = 10
    print(f"\n🔍 Generating explanations for instance {instance_idx}")
    print(f"📋 True class: {target_names[y_test[instance_idx]]}")
    
    # LIME Explanation
    if LIME_AVAILABLE:
        print("\n🧠 LIME Explanation:")
        lime_exp = explainer.lime_explanation(
            models['breast_cancer_rf'], X_train, X_test, instance_idx, feature_names, target_names
        )
        
        if lime_exp:
            print("✅ LIME explanation generated successfully!")
            # Show explanation in text format
            exp_list = lime_exp.as_list()
            print("Top features affecting prediction:")
            for feature, score in exp_list[:5]:
                print(f"  {feature}: {score:.4f}")
        else:
            print("❌ LIME explanation failed")
    
    # SHAP Explanation
    if SHAP_AVAILABLE:
        print("\n🔍 SHAP Explanation:")
        shap_explainer, shap_values = explainer.shap_explanation(models['breast_cancer_rf'], X_test)
        
        if shap_explainer and shap_values is not None:
            print("✅ SHAP explanation generated successfully!")
            # Show top features
            if isinstance(shap_values, list):
                instance_shap = shap_values[1][instance_idx]
            else:
                instance_shap = shap_values[instance_idx]
            
            top_indices = np.argsort(np.abs(instance_shap))[-5:]
            print("Top features affecting prediction:")
            for idx in top_indices:
                print(f"  {feature_names[idx]}: {instance_shap[idx]:.4f}")
        else:
            print("❌ SHAP explanation failed")
    
    # Feature importance comparison
    print("\n📊 Feature Importance Comparison:")
    if importance_data:
        for model_name, importances in importance_data.items():
            top_features = np.argsort(importances)[-5:]
            print(f"\n{model_name.replace('_', ' ').title()} - Top 5 features:")
            for idx in top_features:
                print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    
    print("\n✅ XAI demonstration completed!")
    print("\n🧠 What This Demonstration Shows:")
    print("• Multiple model types: Random Forest, Gradient Boosting, Neural Networks")
    print("• Model performance comparison and evaluation")
    print("• Feature importance analysis across different models")
    print("• Dataset analysis and visualization")
    if LIME_AVAILABLE:
        print("• LIME explanations for individual predictions")
    if SHAP_AVAILABLE:
        print("• SHAP explanations for feature contributions")
    print("• Comprehensive visualization with Matplotlib and Seaborn")

if __name__ == "__main__":
    main()
