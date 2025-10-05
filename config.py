# Configuration file for Modern XAI Toolkit
import os

class Config:
    """Configuration settings for the XAI toolkit"""
    
    # Data settings
    DATA_DIR = "data"
    OUTPUT_DIR = "outputs"
    LOG_DIR = "logs"
    
    # Model settings
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    
    # Model parameters
    RF_PARAMS = {
        'n_estimators': 100,
        'random_state': RANDOM_STATE,
        'max_depth': 10
    }
    
    GB_PARAMS = {
        'n_estimators': 100,
        'random_state': RANDOM_STATE,
        'learning_rate': 0.1
    }
    
    XGB_PARAMS = {
        'n_estimators': 100,
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss',
        'verbosity': 0
    }
    
    LGB_PARAMS = {
        'n_estimators': 100,
        'random_state': RANDOM_STATE,
        'verbose': -1
    }
    
    NN_PARAMS = {
        'hidden_layer_sizes': (100, 50),
        'max_iter': 500,
        'random_state': RANDOM_STATE
    }
    
    # XAI settings
    LIME_NUM_FEATURES = 10
    SHAP_MAX_SAMPLES = 100
    INTEGRATED_GRADIENTS_STEPS = 50
    
    # Visualization settings
    FIGURE_SIZE = (10, 6)
    DPI = 150
    STYLE = 'seaborn-v0_8'
    
    # Web interface settings
    GRADIO_PORT = 7860
    STREAMLIT_PORT = 8501
    
    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for directory in [cls.DATA_DIR, cls.OUTPUT_DIR, cls.LOG_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_model_params(cls, model_type):
        """Get parameters for specific model type"""
        param_map = {
            'rf': cls.RF_PARAMS,
            'gb': cls.GB_PARAMS,
            'xgb': cls.XGB_PARAMS,
            'lgb': cls.LGB_PARAMS,
            'nn': cls.NN_PARAMS
        }
        return param_map.get(model_type, {})
