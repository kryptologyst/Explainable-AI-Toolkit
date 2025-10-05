# Explainable AI Toolkit

A comprehensive toolkit for demonstrating and implementing modern Explainable AI (XAI) techniques across multiple datasets and machine learning models.

## Features

### XAI Techniques
- **LIME** (Local Interpretable Model-Agnostic Explanations)
- **SHAP** (SHapley Additive exPlanations)
- **Integrated Gradients** (for neural networks)
- **Counterfactual Explanations**
- **Feature Importance Analysis**

### Model Support
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- Neural Networks (MLPClassifier)

### Datasets
- Breast Cancer Classification
- Wine Quality Classification
- Iris Flower Classification
- Synthetic Classification Dataset

### Visualizations
- Interactive Matplotlib plots
- Seaborn statistical visualizations
- Plotly interactive dashboards
- Feature importance comparisons
- SHAP summary plots
- LIME explanation visualizations

### Web Interfaces
- **Gradio** - Interactive web app with real-time explanations
- **Streamlit** - Comprehensive dashboard with multiple tabs

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Explainable-AI-Toolkit.git
cd Explainable-AI-Toolkit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Command Line Interface
Run the main script to see XAI techniques in action:

```bash
python modern_xai.py
```

#### 2. Gradio Web Interface
Launch the interactive web app:

```bash
python xai_web_app.py
```

Then open your browser to `http://localhost:7860`

#### 3. Streamlit Dashboard
Launch the comprehensive dashboard:

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

## 📁 Project Structure

```
modern-xai-toolkit/
├── modern_xai.py          # Core XAI implementation
├── xai_web_app.py         # Gradio web interface
├── streamlit_app.py       # Streamlit dashboard
├── 0160.py               # Original implementation
├── requirements.txt       # Dependencies
├── README.md             # This file
└── examples/            # Example notebooks and scripts
```

## 🔧 Core Components

### MockDatabase
- Manages multiple datasets for comprehensive testing
- Supports breast cancer, wine, iris, and synthetic datasets
- Provides easy dataset switching and information retrieval

### ModernXAIExplainer
- Implements multiple XAI techniques
- Supports various model types
- Handles data preprocessing and scaling
- Provides error handling and fallbacks

### XAIVisualizer
- Creates static and interactive visualizations
- Supports Matplotlib, Seaborn, and Plotly
- Generates comparison charts and dashboards
- Handles multi-class and binary classification

## Example Outputs

### LIME Explanation
Shows local feature importance for individual predictions with positive/negative contributions.

### SHAP Summary Plot
Displays global feature importance across all instances with feature value distributions.

### Feature Importance Comparison
Compares feature importance across different model types to identify consistent patterns.

### Interactive Dashboard
Plotly-based dashboard with multiple views of model behavior and explanations.

## 🛠️ Advanced Features

### Model Comparison
- Train multiple model types simultaneously
- Compare performance metrics
- Analyze feature importance consistency
- Identify model-specific behaviors

### Explanation Quality Metrics
- Consistency across different XAI methods
- Stability analysis
- Feature ranking correlation
- Explanation fidelity measures

### Custom Dataset Support
- Easy integration of new datasets
- Automatic feature engineering
- Support for different data types
- Custom preprocessing pipelines

## Technical Details

### Dependencies
- **Core ML**: scikit-learn, numpy, pandas
- **XAI Libraries**: lime, shap, captum, alibi
- **Visualization**: matplotlib, seaborn, plotly
- **Web UI**: gradio, streamlit
- **Deep Learning**: torch, tensorflow
- **Gradient Boosting**: xgboost, lightgbm

### Performance Considerations
- Efficient SHAP computation using TreeExplainer
- Parallel model training
- Memory-optimized data handling
- Caching for repeated explanations

## Use Cases

### Research & Education
- Understanding XAI method differences
- Comparing explanation quality
- Teaching XAI concepts
- Prototyping new techniques

### Model Development
- Feature selection guidance
- Model debugging and validation
- Bias detection and mitigation
- Performance optimization

### Business Applications
- Regulatory compliance
- Stakeholder communication
- Risk assessment
- Decision support systems

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Areas for Contribution
- Additional XAI techniques
- New visualization methods
- Performance optimizations
- Documentation improvements
- Bug fixes and enhancements

## References

- [LIME Paper](https://arxiv.org/abs/1602.04938)
- [SHAP Paper](https://arxiv.org/abs/1705.07874)
- [Integrated Gradients](https://arxiv.org/abs/1703.01365)
- [Counterfactual Explanations](https://arxiv.org/abs/1711.00399)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Scikit-learn team for excellent ML tools
- SHAP and LIME developers for XAI libraries
- Plotly and Streamlit teams for visualization frameworks
- The open-source ML community

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact the maintainers
- Join our community discussions


# Explainable-AI-Toolkit
