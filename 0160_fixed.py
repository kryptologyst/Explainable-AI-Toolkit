# Project 160. Explainable AI techniques - Fixed Version
# Description:
# Explainable AI (XAI) helps humans understand how and why machine learning models make predictions. 
# This project demonstrates core techniques like LIME (Local Interpretable Model-Agnostic Explanations) 
# and SHAP (SHapley Additive exPlanations) to explain predictions of a trained classifier on tabular data.

# Python Implementation: LIME + SHAP on Tabular Classifier (Breast Cancer Dataset)
# Install if not already: pip install scikit-learn matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Try to import XAI libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️  SHAP not available - install with: pip install shap")
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    print("⚠️  LIME not available - install with: pip install lime")
    LIME_AVAILABLE = False

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
feature_names = data.feature_names

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train a black-box model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy:.4f}")

# Pick an instance to explain
idx = 10
instance = X_test.iloc[idx:idx+1].values
true_class = data.target_names[y_test[idx]]
predicted_class = data.target_names[y_pred[idx]]

print(f"\n🔍 Explaining instance {idx}")
print(f"📋 True class: {true_class}")
print(f"🎯 Predicted class: {predicted_class}")

# -----------------------
# 🔍 LIME Explanation
# -----------------------
if LIME_AVAILABLE:
    print("\n🧠 LIME Explanation:")
    try:
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=feature_names,
            class_names=data.target_names,
            mode="classification"
        )
        
        exp = explainer_lime.explain_instance(instance[0], model.predict_proba, num_features=5)
        print("✅ LIME explanation generated!")
        
        # Show explanation in text format
        exp_list = exp.as_list()
        print("Top features affecting prediction:")
        for feature, score in exp_list:
            print(f"  {feature}: {score:.4f}")
            
    except Exception as e:
        print(f"❌ LIME explanation failed: {e}")
else:
    print("\n🧠 LIME Explanation: Skipped (not available)")

# -----------------------
# 🔍 SHAP Explanation
# -----------------------
if SHAP_AVAILABLE:
    print("\n🔍 SHAP Explanation:")
    try:
        explainer_shap = shap.TreeExplainer(model)
        shap_values = explainer_shap.shap_values(X_test.iloc[:20])  # Use smaller sample
        
        print("✅ SHAP explanation generated!")
        
        # Show SHAP values for the instance
        if isinstance(shap_values, list):
            instance_shap = shap_values[1][idx]  # Use positive class
        else:
            instance_shap = shap_values[idx]
        
        # Get top features
        top_indices = np.argsort(np.abs(instance_shap))[-5:]
        print("Top features affecting prediction:")
        for i in top_indices:
            print(f"  {feature_names[i]}: {instance_shap[i]:.4f}")
            
    except Exception as e:
        print(f"❌ SHAP explanation failed: {e}")
else:
    print("\n🔍 SHAP Explanation: Skipped (not available)")

# -----------------------
# 📊 Feature Importance Analysis
# -----------------------
print("\n📊 Feature Importance Analysis:")
feature_importance = model.feature_importances_
top_features = np.argsort(feature_importance)[-10:]

print("Top 10 most important features:")
for i in top_features:
    print(f"  {feature_names[i]}: {feature_importance[i]:.4f}")

# -----------------------
# 📈 Visualizations
# -----------------------
print("\n📈 Creating visualizations...")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Feature Importance
axes[0, 0].barh(range(len(top_features)), feature_importance[top_features])
axes[0, 0].set_yticks(range(len(top_features)))
axes[0, 0].set_yticklabels([feature_names[i] for i in top_features])
axes[0, 0].set_xlabel('Importance')
axes[0, 0].set_title('Top 10 Feature Importance')
axes[0, 0].grid(True, alpha=0.3)

# 2. Class Distribution
class_counts = pd.Series(y).value_counts()
axes[0, 1].pie(class_counts.values, labels=data.target_names, autopct='%1.1f%%')
axes[0, 1].set_title('Class Distribution')

# 3. Feature Correlation Heatmap (top 10 features)
corr_matrix = X.corr()
top_corr_features = corr_matrix.abs().sum().nlargest(10).index
sns.heatmap(corr_matrix.loc[top_corr_features, top_corr_features], 
            annot=True, cmap='coolwarm', ax=axes[1, 0])
axes[1, 0].set_title('Feature Correlation (Top 10)')

# 4. Model Performance Metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
# Calculate additional metrics
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

values = [accuracy, precision, recall, f1]
bars = axes[1, 1].bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Model Performance Metrics')
axes[1, 1].set_ylim(0, 1)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\n✅ XAI demonstration completed!")
print("\n🧠 What This Project Demonstrates:")
print("• Uses Random Forest for breast cancer classification")
print("• Shows model performance metrics and evaluation")
print("• Analyzes feature importance for model interpretability")
print("• Creates comprehensive visualizations")
if LIME_AVAILABLE:
    print("• Uses LIME to provide local explanation by approximating model locally")
if SHAP_AVAILABLE:
    print("• Uses SHAP to show feature importance using Shapley values from game theory")
print("• Visualizes feature importance, correlations, and model performance")
