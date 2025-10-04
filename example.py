#!/usr/bin/env python3
"""
Example usage of the ModelPipeline fo    # Step 5: Fine-tune model
    print("\n5. Fine-tuning model...")
    finetune_results = pipeline.fine_tune(
        epochs=2,  # Reduced for quick test
        learning_rate=2e-6,
        patience=3
    )ecasting
This script demonstrates how to use the complete pipeline for training and forecasting
"""

from model import ModelPipeline
from data import DataHandler
import torch
import numpy as np


def run_complete_pipeline():
    """Run the complete model pipeline from data loading to forecasting"""
    
    print("="*60)
    print("NO2 FORECASTING PIPELINE")
    print("="*60)
    
    # Step 1: Initialize data handler and load data
    print("\n1. Loading and preprocessing data...")
    data_handler = DataHandler(scaler='scaler.pkl')
    
    # Load your CSV file (replace with actual filename)
    try:
        data_handler.load_csv("NO2.csv")
        print(f"Data loaded successfully. Shape: {data_handler.df.shape}")
        print(f"Date range: {data_handler.df['datetime'].min()} to {data_handler.df['datetime'].max()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure NO2.csv exists in the current directory")
        return
    
    # Step 2: Initialize model pipeline
    print("\n2. Initializing model pipeline...")
    pipeline = ModelPipeline(
        data_handler=data_handler,
        lookback_days=7,
        device='auto',
        model_save_dir='./models'
    )
    print(f"Using device: {pipeline.device}")
    
    # 3. Prepare data loaders
    print("3. Preparing data loaders...")
    pipeline.prepare_data(
        train_split=0.7,  # Reduced to leave more room for validation
        batch_size=4
    )
    
    # Step 4: Train the model
    print("\n4. Training model...")
    train_results = pipeline.train(
        epochs=3,  # Reduced for quick test
        learning_rate=7e-6,
        patience=5,
        save_best=True,
        reset_model=True
    )
    
    if train_results:
        print(f"Training completed in {train_results['epochs_trained']} epochs")
        print(f"Best validation loss: {train_results['best_val_loss']:.6f}")
    
    # Step 5: Fine-tune the model
    print("\n5. Fine-tuning model...")
    finetune_results = pipeline.fine_tune(
        finetune_percentage=0.1,  # Use 10% of validation data
        finetune_epochs=10,
        finetune_learning_rate=2e-6,
        patience=3,
        load_best=True
    )
    
    if finetune_results:
        print(f"Fine-tuning improvement: {finetune_results['improvement_percentage']:.2f}%")
    
    # Step 6: Evaluate the model
    print("\n6. Evaluating model...")
    test_results = pipeline.evaluate('test')
    
    print(f"Test MSE (scaled): {test_results['mse_scaled']:.6f}")
    print(f"Test MAE (scaled): {test_results['mae_scaled']:.6f}")
    if test_results['mse_original']:
        print(f"Test MSE (original): {test_results['mse_original']:.6f}")
        print(f"Test RMSE (original): {test_results['rmse_original']:.6f}")
    
    # Step 7: Generate 7-day forecast
    print("\n7. Generating 7-day forecast...")
    forecast_results = pipeline.forecast(
        num_days=7,
        save_csv=True,
        output_file="no2_forecast_7days.csv"
    )
    
    print(f"Forecast summary:")
    print(f"  Date range: {forecast_results['dates'][0].date()} to {forecast_results['dates'][-1].date()}")
    print(f"  Mean value: {forecast_results['summary_stats']['mean']:.6f}")
    print(f"  Value range: [{forecast_results['summary_stats']['min']:.6f}, {forecast_results['summary_stats']['max']:.6f}]")
    
    # Step 8: Plot training history
    print("\n8. Plotting training history...")
    pipeline.plot_training_history(save_plot=True)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Models saved in: {pipeline.model_save_dir}")
    print("Generated files:")
    print("  - best_model.pth (initial training)")
    print("  - finetuned_model.pth (fine-tuned model)")
    print("  - no2_forecast_7days.csv (forecast data)")
    print("  - training_history.png (training plots)")


def run_quick_forecast():
    """Quick example for generating forecast with pre-trained model"""
    
    print("\nQUICK FORECAST EXAMPLE")
    print("="*40)
    
    # Load data
    data_handler = DataHandler(scaler='scaler.pkl')
    data_handler.load_csv("NO2.csv")
    
    # Initialize pipeline
    pipeline = ModelPipeline(data_handler=data_handler)
    pipeline.prepare_data()
    
    # Load pre-trained model (if exists)
    if pipeline._load_model('finetuned_model.pth') or pipeline._load_model('best_model.pth'):
        # Generate forecast
        forecast = pipeline.forecast(num_days=7)
        print(f"Forecast generated for {len(forecast['dates'])} days")
    else:
        print("No pre-trained model found. Please run training first.")


def run_evaluation_only():
    """Quick example for evaluating a pre-trained model"""
    
    print("\nEVALUATION EXAMPLE")
    print("="*30)
    
    # Load data
    data_handler = DataHandler(scaler='scaler.pkl')
    data_handler.load_csv("NO2.csv")
    
    # Initialize pipeline
    pipeline = ModelPipeline(data_handler=data_handler)
    pipeline.prepare_data()
    
    # Load pre-trained model
    if pipeline._load_model('finetuned_model.pth') or pipeline._load_model('best_model.pth'):
        # Evaluate on test set
        results = pipeline.evaluate('test')
        print("Evaluation completed!")
    else:
        print("No pre-trained model found. Please run training first.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            run_quick_forecast()
        elif sys.argv[1] == "eval":
            run_evaluation_only()
        else:
            print("Usage: python example.py [quick|eval]")
            print("  quick: Generate forecast with existing model")
            print("  eval:  Evaluate existing model")
            print("  (no args): Run complete pipeline")
    else:
        run_complete_pipeline()