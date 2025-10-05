# Streamlit Web UI for Modern XAI Toolkit
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import our XAI components
from modern_xai import MockDatabase, ModernXAIExplainer, XAIVisualizer

# Configure Streamlit page
st.set_page_config(
    page_title="Modern XAI Toolkit",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitXAIApp:
    """Streamlit application for XAI demonstrations"""
    
    def __init__(self):
        self.db = MockDatabase()
        self.explainer = ModernXAIExplainer()
        self.visualizer = XAIVisualizer()
        
        # Initialize session state
        if 'models' not in st.session_state:
            st.session_state.models = {}
        if 'dataset' not in st.session_state:
            st.session_state.dataset = None
        if 'scaler' not in st.session_state:
            st.session_state.scaler = None
    
    def run(self):
        """Main application runner"""
        
        # Header
        st.title("🧠 Modern Explainable AI Toolkit")
        st.markdown("""
        This interactive toolkit demonstrates various XAI techniques including LIME, SHAP, 
        Integrated Gradients, and Counterfactual Explanations across multiple datasets and model types.
        """)
        
        # Sidebar controls
        with st.sidebar:
            st.header("🎛️ Controls")
            
            # Dataset selection
            dataset_name = st.selectbox(
                "Select Dataset",
                self.db.list_datasets(),
                index=0
            )
            
            # Show dataset info
            dataset = self.db.get_dataset(dataset_name)
            if dataset:
                st.subheader("📊 Dataset Info")
                st.write(f"**Description:** {dataset['description']}")
                st.write(f"**Features:** {len(dataset['feature_names'])}")
                st.write(f"**Samples:** {len(dataset['data'])}")
                st.write(f"**Classes:** {len(dataset['target_names'])}")
                
                # Show feature names
                with st.expander("Feature Names"):
                    st.write(dataset['feature_names'])
            
            # Train models button
            if st.button("🚀 Train Models", type="primary"):
                self.train_models(dataset_name)
            
            # Model selection
            if st.session_state.models:
                model_name = st.selectbox(
                    "Select Model",
                    list(st.session_state.models.keys())
                )
            else:
                model_name = None
                st.info("Please train models first")
            
            # Instance selection for LIME
            instance_idx = st.slider(
                "Instance Index for LIME",
                min_value=0,
                max_value=100,
                value=10,
                step=1
            )
        
        # Main content area
        if st.session_state.models and model_name:
            self.show_explanations(dataset_name, model_name, instance_idx)
        else:
            st.info("👈 Please select a dataset and train models to get started")
    
    def train_models(self, dataset_name):
        """Train models for selected dataset"""
        with st.spinner("Training models..."):
            dataset = self.db.get_dataset(dataset_name)
            if dataset is None:
                st.error("❌ Dataset not found!")
                return
            
            st.session_state.dataset = dataset
            
            # Prepare data
            X = dataset['data']
            y = dataset['target']
            
            # Train-test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train models
            models = self.explainer.train_models(X_train, y_train, dataset_name)
            st.session_state.models = models
            st.session_state.scaler = self.explainer.scalers[dataset_name]
            
            # Show results
            st.success("✅ Models trained successfully!")
            
            # Display model performance
            from sklearn.metrics import accuracy_score
            st.subheader("📊 Model Performance")
            
            performance_data = []
            for model_name, model in models.items():
                if 'nn' in model_name:
                    X_test_scaled = st.session_state.scaler.transform(X_test)
                    y_pred = model.predict(X_test_scaled)
                else:
                    y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                performance_data.append({
                    'Model': model_name,
                    'Accuracy': accuracy
                })
            
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)
            
            # Performance chart
            fig = px.bar(
                performance_df, 
                x='Model', 
                y='Accuracy',
                title='Model Performance Comparison',
                color='Accuracy',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def show_explanations(self, dataset_name, model_name, instance_idx):
        """Show XAI explanations"""
        
        model = st.session_state.models[model_name]
        dataset = st.session_state.dataset
        
        # Prepare data
        X = dataset['data']
        y = dataset['target']
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Tabs for different explanations
        tab1, tab2, tab3, tab4 = st.tabs(["🧠 LIME", "🔍 SHAP", "📈 Feature Importance", "🔄 Counterfactual"])
        
        with tab1:
            st.subheader("LIME Explanation")
            st.write(f"Explaining prediction for instance {instance_idx} using {model_name}")
            
            if st.button("Generate LIME Explanation", key="lime_btn"):
                with st.spinner("Generating LIME explanation..."):
                    lime_exp = self.explainer.lime_explanation(
                        model, X_train, X_test, instance_idx,
                        dataset['feature_names'], dataset['target_names']
                    )
                    
                    if lime_exp:
                        # Show LIME explanation
                        fig = self.visualizer.plot_lime_explanation(
                            lime_exp, f"LIME Explanation - {model_name} (Instance {instance_idx})"
                        )
                        st.pyplot(fig)
                        
                        # Show explanation details
                        st.subheader("Explanation Details")
                        exp_list = lime_exp.as_list()
                        exp_df = pd.DataFrame(exp_list, columns=['Feature', 'Importance'])
                        st.dataframe(exp_df, use_container_width=True)
                    else:
                        st.error("Failed to generate LIME explanation")
        
        with tab2:
            st.subheader("SHAP Explanation")
            st.write(f"Global SHAP explanation for {model_name}")
            
            if st.button("Generate SHAP Explanation", key="shap_btn"):
                with st.spinner("Generating SHAP explanation..."):
                    shap_explainer, shap_values = self.explainer.shap_explanation(model, X_test)
                    
                    if shap_explainer and shap_values is not None:
                        # Show SHAP summary plot
                        fig = self.visualizer.plot_shap_summary(
                            shap_values, X_test, dataset['feature_names'],
                            f"SHAP Summary - {model_name}"
                        )
                        st.pyplot(fig)
                        
                        # Show SHAP values for first few instances
                        st.subheader("SHAP Values (First 5 Instances)")
                        if isinstance(shap_values, list):
                            shap_df = pd.DataFrame(shap_values[1][:5], columns=dataset['feature_names'])
                        else:
                            shap_df = pd.DataFrame(shap_values[:5], columns=dataset['feature_names'])
                        st.dataframe(shap_df, use_container_width=True)
                    else:
                        st.error("Failed to generate SHAP explanation")
        
        with tab3:
            st.subheader("Feature Importance Comparison")
            st.write("Comparing feature importance across all trained models")
            
            if st.button("Generate Feature Importance", key="importance_btn"):
                with st.spinner("Generating feature importance comparison..."):
                    fig = self.visualizer.plot_feature_importance_comparison(
                        st.session_state.models, dataset['feature_names']
                    )
                    st.pyplot(fig)
        
        with tab4:
            st.subheader("Counterfactual Explanation")
            st.write(f"Finding counterfactual for instance {instance_idx}")
            
            if st.button("Generate Counterfactual", key="counterfactual_btn"):
                with st.spinner("Generating counterfactual explanation..."):
                    counterfactual = self.explainer.counterfactual_explanation(
                        model, X_test, instance_idx
                    )
                    
                    if counterfactual is not None:
                        st.subheader("Original vs Counterfactual")
                        
                        original = X_test.iloc[instance_idx]
                        cf = counterfactual[0]
                        
                        comparison_df = pd.DataFrame({
                            'Feature': dataset['feature_names'],
                            'Original': original.values,
                            'Counterfactual': cf,
                            'Difference': cf - original.values
                        })
                        
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Show prediction difference
                        original_pred = model.predict([original.values])[0]
                        cf_pred = model.predict([cf])[0]
                        
                        st.write(f"**Original Prediction:** {dataset['target_names'][original_pred]}")
                        st.write(f"**Counterfactual Prediction:** {dataset['target_names'][cf_pred]}")
                    else:
                        st.error("Failed to generate counterfactual explanation")
        
        # Additional analysis
        st.subheader("📊 Additional Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Dataset statistics
            st.write("**Dataset Statistics:**")
            stats_df = dataset['data'].describe()
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            # Class distribution
            st.write("**Class Distribution:**")
            class_counts = pd.Series(y).value_counts()
            class_df = pd.DataFrame({
                'Class': [dataset['target_names'][i] for i in class_counts.index],
                'Count': class_counts.values
            })
            
            fig = px.pie(class_df, values='Count', names='Class', title='Class Distribution')
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function to run Streamlit app"""
    app = StreamlitXAIApp()
    app.run()

if __name__ == "__main__":
    main()
