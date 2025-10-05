# Interactive Web UI for Modern XAI Toolkit using Gradio
import gradio as gr
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

class XAIWebApp:
    """Interactive web application for XAI demonstrations"""
    
    def __init__(self):
        self.db = MockDatabase()
        self.explainer = ModernXAIExplainer()
        self.visualizer = XAIVisualizer()
        self.current_models = {}
        self.current_dataset = None
        
    def train_models_for_dataset(self, dataset_name):
        """Train models for selected dataset"""
        dataset = self.db.get_dataset(dataset_name)
        if dataset is None:
            return "❌ Dataset not found!"
        
        self.current_dataset = dataset
        
        # Prepare data
        X = dataset['data']
        y = dataset['target']
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train models
        self.current_models = self.explainer.train_models(X_train, y_train, dataset_name)
        
        # Evaluate models
        from sklearn.metrics import accuracy_score
        results = []
        for model_name, model in self.current_models.items():
            if 'nn' in model_name:
                X_test_scaled = self.explainer.scalers[dataset_name].transform(X_test)
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            results.append(f"✅ {model_name}: {accuracy:.4f}")
        
        return "\n".join(results)
    
    def generate_lime_explanation(self, dataset_name, model_name, instance_idx):
        """Generate LIME explanation"""
        if not self.current_models:
            return "❌ Please train models first!"
        
        if model_name not in self.current_models:
            return "❌ Model not found!"
        
        dataset = self.current_dataset
        model = self.current_models[model_name]
        
        # Prepare data
        X = dataset['data']
        y = dataset['target']
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Generate LIME explanation
        lime_exp = self.explainer.lime_explanation(
            model, X_train, X_test, instance_idx, 
            dataset['feature_names'], dataset['target_names']
        )
        
        if lime_exp:
            # Create visualization
            fig = self.visualizer.plot_lime_explanation(
                lime_exp, f"LIME Explanation - {model_name} (Instance {instance_idx})"
            )
            
            # Convert to base64 for display
            import io
            import base64
            
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
            
            return f"data:image/png;base64,{image_base64}"
        else:
            return "❌ Failed to generate LIME explanation"
    
    def generate_shap_explanation(self, dataset_name, model_name):
        """Generate SHAP explanation"""
        if not self.current_models:
            return "❌ Please train models first!"
        
        if model_name not in self.current_models:
            return "❌ Model not found!"
        
        dataset = self.current_dataset
        model = self.current_models[model_name]
        
        # Prepare data
        X = dataset['data']
        y = dataset['target']
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Generate SHAP explanation
        shap_explainer, shap_values = self.explainer.shap_explanation(model, X_test)
        
        if shap_explainer and shap_values is not None:
            # Create visualization
            fig = self.visualizer.plot_shap_summary(
                shap_values, X_test, dataset['feature_names'], 
                f"SHAP Summary - {model_name}"
            )
            
            # Convert to base64 for display
            import io
            import base64
            
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
            
            return f"data:image/png;base64,{image_base64}"
        else:
            return "❌ Failed to generate SHAP explanation"
    
    def generate_feature_importance(self, dataset_name):
        """Generate feature importance comparison"""
        if not self.current_models:
            return "❌ Please train models first!"
        
        dataset = self.current_dataset
        
        # Create visualization
        fig = self.visualizer.plot_feature_importance_comparison(
            self.current_models, dataset['feature_names']
        )
        
        # Convert to base64 for display
        import io
        import base64
        
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        
        return f"data:image/png;base64,{image_base64}"
    
    def get_dataset_info(self, dataset_name):
        """Get dataset information"""
        dataset = self.db.get_dataset(dataset_name)
        if dataset is None:
            return "❌ Dataset not found!"
        
        info = f"""
📊 **Dataset Information:**
- **Name:** {dataset_name}
- **Description:** {dataset['description']}
- **Features:** {len(dataset['feature_names'])}
- **Samples:** {len(dataset['data'])}
- **Classes:** {len(dataset['target_names'])}
- **Target Names:** {', '.join(dataset['target_names'])}

📋 **Feature Names:**
{', '.join(dataset['feature_names'][:10])}{'...' if len(dataset['feature_names']) > 10 else ''}
        """
        return info

def create_gradio_interface():
    """Create Gradio interface"""
    app = XAIWebApp()
    
    with gr.Blocks(title="Modern XAI Toolkit", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🧠 Modern Explainable AI Toolkit
        
        This interactive toolkit demonstrates various XAI techniques including LIME, SHAP, 
        Integrated Gradients, and Counterfactual Explanations across multiple datasets and model types.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Controls")
                
                dataset_dropdown = gr.Dropdown(
                    choices=app.db.list_datasets(),
                    value="breast_cancer",
                    label="Select Dataset"
                )
                
                train_btn = gr.Button("🚀 Train Models", variant="primary")
                
                model_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select Model for Explanations"
                )
                
                instance_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=10,
                    step=1,
                    label="Instance Index for LIME"
                )
                
                gr.Markdown("### 📊 Actions")
                lime_btn = gr.Button("🧠 Generate LIME Explanation")
                shap_btn = gr.Button("🔍 Generate SHAP Explanation")
                importance_btn = gr.Button("📈 Feature Importance Comparison")
                
            with gr.Column(scale=2):
                gr.Markdown("### 📋 Dataset Information")
                dataset_info = gr.Markdown()
                
                gr.Markdown("### 🤖 Model Training Results")
                training_results = gr.Markdown()
                
                gr.Markdown("### 🧠 Explanations")
                explanation_output = gr.Image(label="XAI Visualization")
        
        # Event handlers
        dataset_dropdown.change(
            fn=app.get_dataset_info,
            inputs=[dataset_dropdown],
            outputs=[dataset_info]
        )
        
        train_btn.click(
            fn=app.train_models_for_dataset,
            inputs=[dataset_dropdown],
            outputs=[training_results]
        ).then(
            fn=lambda: list(app.current_models.keys()),
            outputs=[model_dropdown]
        )
        
        lime_btn.click(
            fn=app.generate_lime_explanation,
            inputs=[dataset_dropdown, model_dropdown, instance_slider],
            outputs=[explanation_output]
        )
        
        shap_btn.click(
            fn=app.generate_shap_explanation,
            inputs=[dataset_dropdown, model_dropdown],
            outputs=[explanation_output]
        )
        
        importance_btn.click(
            fn=app.generate_feature_importance,
            inputs=[dataset_dropdown],
            outputs=[explanation_output]
        )
        
        # Initialize with default dataset info
        interface.load(
            fn=app.get_dataset_info,
            inputs=[dataset_dropdown],
            outputs=[dataset_info]
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
