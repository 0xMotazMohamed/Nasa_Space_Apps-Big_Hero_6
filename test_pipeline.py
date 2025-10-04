#!/usr/bin/env python3
"""
Test script to verify the model pipeline components work correctly
"""

import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import tempfile
import shutil


def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        from data import DataHandler, DaySequenceDataset
        from model import ModelPipeline, UNetWrapper
        import torch
        import cv2
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def create_dummy_data(filename="test_NO2.csv", num_days=30):
    """Create dummy data for testing"""
    print(f"Creating dummy data with {num_days} days...")
    
    # Create dummy data with same structure as real data
    H, W = 118, 310
    num_pixels = H * W
    
    dates = pd.date_range('2024-01-01', periods=num_days, freq='D')
    
    # Generate realistic-looking NO2 data with some patterns
    np.random.seed(42)
    data_rows = []
    
    for i, date in enumerate(dates):
        # Create spatial pattern with some temporal variation
        base_pattern = np.random.random((H, W)) * 0.5 + 0.2
        temporal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 30)  # Monthly cycle
        daily_noise = np.random.normal(0, 0.1, (H, W))
        
        pixel_values = (base_pattern * temporal_factor + daily_noise).flatten()
        pixel_values = np.clip(pixel_values, 0, 1)  # Ensure positive values
        
        row = [date] + pixel_values.tolist()
        data_rows.append(row)
    
    # Create DataFrame
    columns = ['datetime'] + [str(i) for i in range(num_pixels)]
    df = pd.DataFrame(data_rows, columns=columns)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"✓ Dummy data saved to {filename}")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    return filename


def test_data_handler():
    """Test DataHandler functionality"""
    print("\nTesting DataHandler...")
    
    try:
        from data import DataHandler
        
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy data
            csv_file = Path(temp_dir) / "test_data.csv"
            create_dummy_data(csv_file, num_days=20)
            
            # Test DataHandler
            data_handler = DataHandler(scaler=str(Path(temp_dir) / 'test_scaler.pkl'))
            data_handler.load_csv(str(csv_file))
            
            # Check if data was loaded correctly
            assert hasattr(data_handler, 'df'), "df not created"
            assert hasattr(data_handler, 'df_scaled'), "df_scaled not created"
            assert hasattr(data_handler, 'meta_df'), "meta_df not created"
            assert hasattr(data_handler, 'scaler'), "scaler not created"
            
            print(f"✓ Data loaded: {data_handler.df.shape}")
            print(f"✓ Scaled data: {data_handler.df_scaled.shape}")
            print(f"✓ Metadata: {data_handler.meta_df.shape}")
            
            # Test data splitting
            train_ds, val_ds, test_ds = data_handler.split(train_size=0.7, LOOKBACK_DAYS=5, test_size=2)
            
            if train_ds is not None:
                print(f"✓ Data split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
            else:
                print("✗ Data splitting failed")
                return False
                
        print("✓ DataHandler test passed")
        return True
        
    except Exception as e:
        print(f"✗ DataHandler test failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation without training"""
    print("\nTesting model creation...")
    
    try:
        from data import DataHandler
        from model import ModelPipeline
        import torch
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / "test_data.csv"
            create_dummy_data(csv_file, num_days=15)
            
            # Initialize components
            data_handler = DataHandler(scaler=str(Path(temp_dir) / 'test_scaler.pkl'))
            data_handler.load_csv(str(csv_file))
            
            pipeline = ModelPipeline(
                data_handler=data_handler,
                lookback_days=5,
                device='cpu',  # Force CPU for testing
                model_save_dir=temp_dir
            )
            
            # Test data preparation
            success = pipeline.prepare_data(batch_size=2, num_workers=0)  # No multiprocessing for test
            if not success:
                print("✗ Data preparation failed")
                return False
            
            print("✓ Data preparation successful")
            
            # Test model creation
            if pipeline._create_model():
                print("✓ Model creation successful")
                print(f"  Model parameters: {sum(p.numel() for p in pipeline.model.parameters())}")
                
                # Test a forward pass
                sample_input = torch.randn(1, 5, 256, 256)  # Batch=1, Channels=5, H=256, W=256
                with torch.no_grad():
                    output = pipeline.model(sample_input)
                print(f"✓ Forward pass successful: {sample_input.shape} -> {output.shape}")
                
            else:
                print("✗ Model creation failed")
                return False
                
        print("✓ Model creation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        traceback.print_exc()
        return False


def test_quick_training():
    """Test a very short training run"""
    print("\nTesting quick training (2 epochs)...")
    
    try:
        from data import DataHandler
        from model import ModelPipeline
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / "test_data.csv"
            create_dummy_data(csv_file, num_days=20)
            
            # Initialize pipeline
            data_handler = DataHandler(scaler=str(Path(temp_dir) / 'test_scaler.pkl'))
            data_handler.load_csv(str(csv_file))
            
            pipeline = ModelPipeline(
                data_handler=data_handler,
                lookback_days=3,
                device='cpu',
                model_save_dir=temp_dir
            )
            
            pipeline.prepare_data(batch_size=2, num_workers=0)
            
            # Quick training
            results = pipeline.train(
                epochs=2,
                learning_rate=1e-3,
                patience=10,
                save_best=True,
                reset_model=True
            )
            
            if results:
                print(f"✓ Training completed: {results['epochs_trained']} epochs")
                print(f"  Best val loss: {results['best_val_loss']:.6f}")
                
                # Test evaluation
                eval_results = pipeline.evaluate('test')
                if eval_results:
                    print(f"✓ Evaluation successful: MSE={eval_results['mse_scaled']:.6f}")
                
                # Test forecast
                forecast = pipeline.forecast(num_days=3, save_csv=False)
                if forecast:
                    print(f"✓ Forecast successful: {len(forecast['dates'])} days")
                
            else:
                print("✗ Training failed")
                return False
                
        print("✓ Quick training test passed")
        return True
        
    except Exception as e:
        print(f"✗ Quick training test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("RUNNING MODEL PIPELINE TESTS")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("DataHandler Test", test_data_handler),
        ("Model Creation Test", test_model_creation),
        ("Quick Training Test", test_quick_training),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-'*40}")
        print(f"Running: {test_name}")
        print(f"{'-'*40}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The pipeline is ready to use.")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Check the errors above.")
    
    return passed == len(results)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Run only import and data tests for quick verification
        print("Running quick tests...")
        test_imports()
        test_data_handler()
    else:
        # Run all tests
        run_all_tests()