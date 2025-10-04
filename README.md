# NO2 Forecasting Model Pipeline

This repository contains a complete pipeline for training and forecasting NO2 concentrations using a U-Net based deep learning model.

## 🔗 Related Resources

### Development Notebook
The original research and development for this model was conducted in this Kaggle notebook:
- **[Big Hero 6 - NASA Space Apps Development Notebook](https://www.kaggle.com/code/mo15fr/big-hero-6)**

### Dataset
The TEMPO NO2 satellite data used for training this model is available on Kaggle:
- **[TEMPO NO2 Data](https://www.kaggle.com/datasets/moatazmohamed8804/tempo-no2-data)**

## Files Overview

- **`data.py`**: Data handling and preprocessing utilities
- **`model.py`**: Complete model pipeline with training, fine-tuning, and forecasting
- **`example.py`**: Example usage scripts
- **`notebook.ipynb`**: Original research and development notebook
- **`NO2.csv`**: Input data file (your satellite data)

## Features

### Data Processing (`DataHandler`)
- Automatic data loading and preprocessing
- IQR-based outlier clipping
- MinMax scaling with persistent scaler
- Data interpolation for missing values
- Automatic train/validation/test splitting

### Model Pipeline (`ModelPipeline`)
- Pre-trained U-Net architecture adapted for multi-channel input
- Complete training pipeline with early stopping
- Fine-tuning on validation data subset (default 10%)
- Model evaluation with original and scaled metrics
- 7-day forecasting capability
- Automatic model saving and loading

## Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn opencv-python matplotlib tqdm
```

### 2. Prepare Your Data
Ensure your CSV file has the format:
- First column: `datetime` (date timestamps)
- Remaining columns: pixel values (one column per pixel)

### 3. Run Complete Pipeline
```python
from model import ModelPipeline
from data import DataHandler

# Load and preprocess data
data_handler = DataHandler()
data_handler.load_csv("NO2.csv")

# Initialize pipeline
pipeline = ModelPipeline(data_handler=data_handler)
pipeline.prepare_data()

# Train model
train_results = pipeline.train(epochs=80)

# Fine-tune model
finetune_results = pipeline.fine_tune()

# Evaluate model
test_results = pipeline.evaluate('test')

# Generate forecast
forecast = pipeline.forecast(num_days=7)
```

### 4. Or Use Example Script
```bash
# Run complete pipeline
python example.py

# Generate forecast with existing model
python example.py quick

# Evaluate existing model
python example.py eval
```

## Model Parameters

### Training Parameters (with defaults from notebook)
- **epochs**: 80 (default from notebook)
- **learning_rate**: 7e-6 (default from notebook)
- **patience**: 5 (early stopping)
- **batch_size**: 4
- **lookback_days**: 7

### Fine-tuning Parameters
- **finetune_percentage**: 0.1 (10% of validation data)
- **finetune_epochs**: 10
- **finetune_learning_rate**: 2e-6
- **patience**: 3

## Key Functions

### `ModelPipeline.train()`
```python
train_results = pipeline.train(
    epochs=80,                    # Number of training epochs
    learning_rate=7e-6,          # Learning rate
    patience=5,                  # Early stopping patience
    save_best=True,              # Save best model
    reset_model=True             # Reset model before training
)
```

### `ModelPipeline.fine_tune()`
```python
finetune_results = pipeline.fine_tune(
    finetune_percentage=0.1,     # 10% of validation data
    finetune_epochs=10,          # Fine-tuning epochs
    finetune_learning_rate=2e-6, # Lower learning rate
    patience=3,                  # Early stopping patience
    load_best=True               # Load best model first
)
```

### `ModelPipeline.forecast()`
```python
forecast_results = pipeline.forecast(
    num_days=7,                  # Number of days to forecast
    save_csv=True,               # Save as CSV
    output_file="forecast.csv"   # Output filename
)
```

## Output Files

- **`models/best_model.pth`**: Best model from initial training
- **`models/finetuned_model.pth`**: Fine-tuned model
- **`models/forecast_7days.csv`**: 7-day forecast in original data format
- **`models/training_history.png`**: Training progress plots
- **`scaler.pkl`**: Fitted data scaler for consistency

## Data Format

### Input CSV Format
```
datetime,0,1,2,3,...,36579
2024-01-01,0.5,0.3,0.8,...,0.2
2024-01-02,0.4,0.6,0.7,...,0.3
...
```

### Forecast CSV Format
Same format as input, with predicted values for future dates.

## Architecture Details

- **Base Model**: Pre-trained U-Net from brain-segmentation-pytorch
- **Input Channels**: 7 (lookback days) → 3 (adapter) → U-Net
- **Output**: Single channel prediction
- **Image Size**: 256×256 (resized from original 118×310)
- **Training Strategy**: Initial training → Fine-tuning on validation subset

## Error Handling

The pipeline includes comprehensive error handling:
- Data validation and cleaning
- Model loading/saving verification
- Training progress monitoring
- Graceful fallbacks for missing components

## Customization

You can easily customize the pipeline by:
- Adjusting model parameters in function calls
- Modifying data preprocessing in `DataHandler`
- Extending the model architecture in `UNetWrapper`
- Adding custom metrics in evaluation functions

## Troubleshooting

1. **CUDA Issues**: Set `device='cpu'` if GPU is not available
2. **Memory Issues**: Reduce `batch_size` or `num_workers`
3. **Data Issues**: Check CSV format and ensure datetime column exists
4. **Model Loading**: Ensure model files exist in the `models/` directory

## Performance Notes

- Training time: ~1-2 hours on GPU for 80 epochs
- Memory usage: ~4-6GB GPU memory with batch_size=4
- Forecast generation: ~1-2 minutes for 7 days
- Fine-tuning: ~10-20 minutes for 10 epochs

## 📚 Development & Data Sources

This machine learning pipeline was developed as part of the NASA Space Apps Challenge. The complete development process, including data exploration, model experimentation, and validation, can be found in our Kaggle notebook:

**🔬 Development Notebook**: [Big Hero 6 - NASA Space Apps](https://www.kaggle.com/code/mo15fr/big-hero-6)
- Contains detailed EDA and data analysis
- Model architecture experiments and hyperparameter tuning
- Performance comparisons and validation results
- Visualization and interpretation of atmospheric data patterns

**📊 Dataset**: [TEMPO NO2 Data](https://www.kaggle.com/datasets/moatazmohamed8804/tempo-no2-data)
- High-resolution satellite observations from TEMPO instrument
- NO2 concentration measurements across North America
- Daily temporal resolution with spatial coverage
- Preprocessed and ready for machine learning applications

## 🏆 NASA Space Apps Challenge

This project was developed for the NASA Space Apps Challenge, focusing on atmospheric data analysis and forecasting. The goal is to predict air quality patterns using satellite data to support environmental monitoring and public health initiatives.