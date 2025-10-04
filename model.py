import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import cv2
from data import DataHandler, DaySequenceDataset, H, W, TARGET_H, TARGET_W


class UNetWrapper(nn.Module):
    """Wrapper to adapt pre-trained U-Net for our multi-channel input"""
    def __init__(self, pretrained_model, input_channels=7):
        super().__init__()
        self.pretrained_model = pretrained_model
        # Add a 1x1 conv to convert from LOOKBACK_DAYS channels to 3 channels
        self.channel_adapter = nn.Conv2d(input_channels, 3, kernel_size=1)
        
    def forward(self, x):
        # Convert from LOOKBACK_DAYS channels to 3 channels
        x_adapted = self.channel_adapter(x)
        # Pass through pre-trained U-Net
        return self.pretrained_model(x_adapted)


class ModelPipeline:
    """Complete model training and inference pipeline"""
    
    def __init__(self, 
                 data_handler=None,
                 lookback_days=7,
                 device='auto',
                 model_save_dir='./models'):
        """
        Initialize the model pipeline
        
        Args:
            data_handler: DataHandler instance for data processing
            lookback_days: Number of days to look back for prediction
            device: Device to use ('auto', 'cuda', 'cpu')
            model_save_dir: Directory to save models
        """
        self.data_handler = data_handler
        self.lookback_days = lookback_days
        self.device = self._get_device(device)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        
        # Model and training components
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.scaler_amp = amp.GradScaler()
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'finetune_loss': []
        }
    
    def _get_device(self, device):
        """Determine the device to use"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _create_model(self):
        """Create and initialize the U-Net model"""
        print("Loading pre-trained U-Net model...")
        try:
            # Load the pre-trained model with original 3 channels
            model_hub = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                in_channels=3, out_channels=1, init_features=32, pretrained=True)
            
            # Wrap it to adapt input channels
            self.model = UNetWrapper(model_hub, input_channels=self.lookback_days).to(self.device)
            print(f'Model loaded successfully with {sum(p.numel() for p in self.model.parameters())} parameters')
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def prepare_data(self, 
                    train_split=0.9,
                    batch_size=4,
                    num_workers=3,
                    pin_memory=True):
        """
        Prepare data loaders for training
        
        Args:
            train_split: Percentage of data for training (validation is remaining)
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        if self.data_handler is None or not hasattr(self.data_handler, 'meta_df'):
            raise ValueError("DataHandler must be initialized and data must be loaded first")
        
        # Split data
        test_size = max(10, self.lookback_days + 3)  # Ensure enough test data
        train_dataset, val_dataset, test_dataset = self.data_handler.split(
            train_size=train_split,
            LOOKBACK_DAYS=self.lookback_days,
            test_size=test_size
        )
        
        if train_dataset is None:
            raise ValueError("Failed to split data")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=pin_memory
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=pin_memory
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=pin_memory
        )
        
        print(f"Data prepared: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        return True
    
    def train(self,
              epochs=80,
              learning_rate=7e-6,
              patience=5,
              save_best=True,
              reset_model=True):
        """
        Train the model from scratch
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            patience: Early stopping patience
            save_best: Whether to save the best model
            reset_model: Whether to reset/recreate the model
        
        Returns:
            dict: Training results and metrics
        """
        # Reset model if requested
        if reset_model or self.model is None:
            if not self._create_model():
                return None
        
        # Reset optimizer and scaler
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler_amp = amp.GradScaler()
        
        # Reset training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'finetune_loss': []
        }
        
        print("Starting training...")
        print("="*50)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            
            train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} (train)")
            for xb, yb in train_pbar:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()
                
                with amp.autocast():
                    preds = self.model(xb)
                    loss = self.criterion(preds, yb.unsqueeze(1))
                
                self.scaler_amp.scale(loss).backward()
                self.scaler_amp.step(self.optimizer)
                self.scaler_amp.update()
                
                running_loss += loss.item()
                train_pbar.set_postfix(loss=loss.item())
            
            train_loss = running_loss / len(self.train_loader)
            self.training_history['train_loss'].append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for xb, yb in self.val_loader:
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                    
                    with amp.autocast():
                        preds = self.model(xb)
                        loss = self.criterion(preds, yb.unsqueeze(1))
                    
                    val_loss += loss.item()
            
            val_loss /= len(self.val_loader)
            self.training_history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}: train loss {train_loss:.6f}, val loss {val_loss:.6f}")
            
            # Early stopping and model saving
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_counter = 0
                if save_best:
                    self._save_model('best_model.pth')
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epoch(s)")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
        
        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        
        return {
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1,
            'training_history': self.training_history
        }
    
    def fine_tune(self,
                  finetune_percentage=0.1,
                  finetune_epochs=10,
                  finetune_learning_rate=2e-6,
                  patience=3,
                  load_best=True):
        """
        Fine-tune the model on the last percentage of validation data
        
        Args:
            finetune_percentage: Percentage of validation data to use for fine-tuning
            finetune_epochs: Number of fine-tuning epochs
            finetune_learning_rate: Learning rate for fine-tuning
            patience: Early stopping patience
            load_best: Whether to load the best model before fine-tuning
        
        Returns:
            dict: Fine-tuning results and metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Load best model if requested
        if load_best:
            self._load_model('best_model.pth')
        
        print("\n" + "="*50)
        print("Starting fine-tuning phase...")
        print("="*50)
        
        # Create fine-tuning data loader (last percentage of validation data)
        val_dataset_length = len(self.val_loader.dataset)
        finetune_size = max(1, int(val_dataset_length * finetune_percentage))
        finetune_start_idx = val_dataset_length - finetune_size
        
        # Create subset of validation dataset for fine-tuning
        val_meta_df = self.val_loader.dataset.meta_df
        finetune_meta = val_meta_df.iloc[finetune_start_idx:]
        finetune_dataset = DaySequenceDataset(finetune_meta, self.lookback_days)
        
        finetune_loader = DataLoader(
            finetune_dataset,
            batch_size=self.val_loader.batch_size,
            shuffle=True,
            num_workers=self.val_loader.num_workers,
            pin_memory=True
        )
        
        print(f"Fine-tuning on {len(finetune_dataset)} samples ({finetune_percentage*100}% of validation data)")
        print(f"Learning rate: {finetune_learning_rate}")
        
        # Setup fine-tuning optimizer
        finetune_optimizer = optim.Adam(self.model.parameters(), lr=finetune_learning_rate)
        finetune_scaler = amp.GradScaler()
        
        best_finetune_loss = float('inf')
        finetune_patience_counter = 0
        initial_val_loss = self.training_history['val_loss'][-1] if self.training_history['val_loss'] else float('inf')
        
        for epoch in range(finetune_epochs):
            self.model.train()
            running_finetune_loss = 0.0
            
            finetune_pbar = tqdm(finetune_loader, desc=f"Fine-tune Epoch {epoch+1}/{finetune_epochs}")
            for xb, yb in finetune_pbar:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                
                finetune_optimizer.zero_grad()
                
                with amp.autocast():
                    preds = self.model(xb)
                    loss = self.criterion(preds, yb.unsqueeze(1))
                
                finetune_scaler.scale(loss).backward()
                finetune_scaler.step(finetune_optimizer)
                finetune_scaler.update()
                
                running_finetune_loss += loss.item()
                finetune_pbar.set_postfix(loss=loss.item())
            
            avg_finetune_loss = running_finetune_loss / len(finetune_loader)
            self.training_history['finetune_loss'].append(avg_finetune_loss)
            
            print(f"Fine-tune Epoch {epoch+1}: loss {avg_finetune_loss:.6f}")
            
            # Save best fine-tuned model
            if avg_finetune_loss < best_finetune_loss - 1e-7:
                best_finetune_loss = avg_finetune_loss
                finetune_patience_counter = 0
                self._save_model('finetuned_model.pth')
                print(f"  -> New best fine-tuned model saved (loss: {best_finetune_loss:.6f})")
            else:
                finetune_patience_counter += 1
                if finetune_patience_counter >= patience:
                    print(f"  -> Fine-tuning early stopping after {epoch+1} epochs")
                    break
        
        # Calculate improvement
        improvement = initial_val_loss - best_finetune_loss
        improvement_pct = (improvement / initial_val_loss * 100) if initial_val_loss > 0 else 0
        
        print(f"\nFine-tuning completed!")
        print(f"Initial validation loss: {initial_val_loss:.6f}")
        print(f"Best fine-tuned loss: {best_finetune_loss:.6f}")
        print(f"Improvement: {improvement:.6f} ({improvement_pct:.2f}%)")
        
        # Load best fine-tuned model
        self._load_model('finetuned_model.pth')
        
        return {
            'initial_loss': initial_val_loss,
            'best_finetune_loss': best_finetune_loss,
            'improvement': improvement,
            'improvement_percentage': improvement_pct,
            'epochs_finetuned': epoch + 1
        }
    
    def evaluate(self, loader_name='test'):
        """
        Evaluate the model on test data
        
        Args:
            loader_name: Which data loader to use ('test', 'val', 'train')
        
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Select appropriate data loader
        if loader_name == 'test':
            loader = self.test_loader
        elif loader_name == 'val':
            loader = self.val_loader
        elif loader_name == 'train':
            loader = self.train_loader
        else:
            raise ValueError(f"Unknown loader_name: {loader_name}")
        
        if loader is None:
            raise ValueError(f"{loader_name} loader is not available")
        
        print(f"Evaluating on {loader_name} data...")
        
        self.model.eval()
        test_predictions_scaled = []
        test_ground_truths_scaled = []
        
        with torch.no_grad():
            for xb, yb in tqdm(loader, desc=f"Evaluating {loader_name}"):
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                
                preds = self.model(xb)
                
                test_predictions_scaled.append(preds.cpu().numpy())
                test_ground_truths_scaled.append(yb.cpu().numpy())
        
        # Concatenate all batches
        all_test_preds_scaled = np.concatenate(test_predictions_scaled, axis=0)
        all_test_gts_scaled = np.concatenate(test_ground_truths_scaled, axis=0)
        
        # Ensure ground truths have channel dimension
        if all_test_gts_scaled.ndim == 3:
            all_test_gts_scaled = all_test_gts_scaled[:, np.newaxis, ...]
        
        # Calculate metrics on scaled data
        diff_scaled = all_test_preds_scaled - all_test_gts_scaled
        mse_scaled = np.mean(diff_scaled ** 2)
        mae_scaled = np.mean(np.abs(diff_scaled))
        
        # Convert to original scale if scaler is available
        mse_original = None
        mae_original = None
        
        if hasattr(self.data_handler, 'scaler') and self.data_handler.scaler is not None:
            test_predictions_original = self._convert_to_original_scale(all_test_preds_scaled)
            test_ground_truths_original = self._convert_to_original_scale(all_test_gts_scaled)
            
            diff_original = test_predictions_original - test_ground_truths_original
            mse_original = np.mean(diff_original ** 2)
            mae_original = np.mean(np.abs(diff_original))
        
        results = {
            'mse_scaled': mse_scaled,
            'mae_scaled': mae_scaled,
            'mse_original': mse_original,
            'mae_original': mae_original,
            'rmse_original': np.sqrt(mse_original) if mse_original is not None else None,
            'num_samples': len(all_test_preds_scaled)
        }
        
        print(f"\n{loader_name.title()} Results:")
        print(f"MSE (scaled): {mse_scaled:.6f}")
        print(f"MAE (scaled): {mae_scaled:.6f}")
        if mse_original is not None:
            print(f"MSE (original): {mse_original:.6f}")
            print(f"MAE (original): {mae_original:.6f}")
            print(f"RMSE (original): {np.sqrt(mse_original):.6f}")
        print(f"Number of samples: {len(all_test_preds_scaled)}")
        
        return results
    
    def _convert_to_original_scale(self, scaled_data):
        """Convert scaled predictions back to original scale"""
        original_data = []
        
        for i in range(len(scaled_data)):
            # Remove channel dimension and resize
            data_2d = scaled_data[i, 0] if scaled_data.ndim == 4 else scaled_data[i]
            data_resized = cv2.resize(data_2d, (W, H), interpolation=cv2.INTER_LINEAR)
            
            # Flatten and inverse transform
            data_flat = data_resized.flatten().reshape(1, -1)
            data_original = self.data_handler.scaler.inverse_transform(data_flat).flatten()
            original_data.append(data_original.reshape(H, W))
        
        return np.array(original_data)
    
    def forecast(self, num_days=7, save_csv=True, output_file="forecast_7days.csv"):
        """
        Generate forecast for the next num_days after the last available data
        
        Args:
            num_days: Number of days to forecast
            save_csv: Whether to save forecast as CSV
            output_file: Output CSV filename
        
        Returns:
            dict: Forecast results
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        print(f"Generating {num_days}-day forecast...")
        
        # Get the last LOOKBACK_DAYS from the dataset for forecasting
        last_sequence_files = self.data_handler.meta_df['file'].iloc[-self.lookback_days:].tolist()
        forecast_input = np.stack([np.load(f) for f in last_sequence_files], axis=0)
        forecast_input_tensor = torch.from_numpy(forecast_input).float().unsqueeze(0).to(self.device)
        
        # Generate forecast
        forecast_predictions_scaled = []
        current_input = forecast_input_tensor.clone()
        
        self.model.eval()
        with torch.no_grad():
            for day in range(num_days):
                # Predict next day
                next_day_pred = self.model(current_input)
                forecast_predictions_scaled.append(next_day_pred.cpu().numpy()[0, 0])
                
                # Update input sequence: remove oldest day, add predicted day
                next_day_pred_expanded = next_day_pred.squeeze(0)
                current_input = torch.cat([current_input[:, 1:], next_day_pred_expanded.unsqueeze(0)], dim=1)
        
        forecast_predictions_scaled = np.array(forecast_predictions_scaled)
        
        # Convert to original scale
        forecast_predictions_original = self._convert_to_original_scale(
            forecast_predictions_scaled[:, np.newaxis, :, :]
        )
        
        # Generate forecast dates
        forecast_dates = pd.date_range(
            start=self.data_handler.LAST_DATE + pd.Timedelta(days=1),
            periods=num_days,
            freq='D'
        )
        
        print(f"Forecast generated for dates: {forecast_dates[0].date()} to {forecast_dates[-1].date()}")
        
        # Save as CSV if requested
        if save_csv and hasattr(self.data_handler, 'scaler'):
            self._save_forecast_csv(forecast_predictions_original, forecast_dates, output_file)
        
        return {
            'dates': forecast_dates,
            'predictions_scaled': forecast_predictions_scaled,
            'predictions_original': forecast_predictions_original,
            'summary_stats': {
                'min': forecast_predictions_original.min(),
                'max': forecast_predictions_original.max(),
                'mean': forecast_predictions_original.mean(),
                'std': forecast_predictions_original.std()
            }
        }
    
    def _save_forecast_csv(self, forecast_data, dates, filename):
        """Save forecast data in CSV format"""
        forecast_rows = []
        num_pixels = H * W
        
        for i, date in enumerate(dates):
            forecast_flattened = forecast_data[i].flatten()
            row = [date] + forecast_flattened.tolist()
            forecast_rows.append(row)
        
        column_names = ['datetime'] + [str(i) for i in range(num_pixels)]
        forecast_df = pd.DataFrame(forecast_rows, columns=column_names)
        
        output_path = self.model_save_dir / filename
        forecast_df.to_csv(output_path, index=False)
        print(f"Forecast saved to: {output_path}")
    
    def _save_model(self, filename):
        """Save model state dict"""
        filepath = self.model_save_dir / filename
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to: {filepath}")
    
    def _load_model(self, filename):
        """Load model state dict"""
        filepath = self.model_save_dir / filename
        if filepath.exists():
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
            print(f"Model loaded from: {filepath}")
            return True
        else:
            print(f"Model file not found: {filepath}")
            return False
    
    def plot_training_history(self, save_plot=True, filename="training_history.png"):
        """Plot training history"""
        if not any(self.training_history.values()):
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training and validation loss
        if self.training_history['train_loss'] and self.training_history['val_loss']:
            epochs = range(1, len(self.training_history['train_loss']) + 1)
            axes[0].plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
            axes[0].plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training and Validation Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot fine-tuning progress
        if self.training_history['finetune_loss']:
            ft_epochs = range(1, len(self.training_history['finetune_loss']) + 1)
            axes[1].plot(ft_epochs, self.training_history['finetune_loss'], 'g-', marker='o')
            axes[1].set_xlabel('Fine-tune Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Fine-tuning Progress')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.model_save_dir / filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {plot_path}")
        
        plt.show()


def main():
    """Example usage of the ModelPipeline"""
    # Initialize data handler and load data
    data_handler = DataHandler()
    data_handler.load_csv("NO2.csv")  # Replace with your actual CSV file
    
    # Initialize model pipeline
    pipeline = ModelPipeline(
        data_handler=data_handler,
        lookback_days=7,
        device='auto'
    )
    
    # Prepare data
    pipeline.prepare_data(batch_size=4)
    
    # Train model
    train_results = pipeline.train(
        epochs=80,
        learning_rate=7e-6,
        patience=5
    )
    
    # Fine-tune model
    finetune_results = pipeline.fine_tune(
        finetune_percentage=0.1,
        finetune_epochs=10,
        finetune_learning_rate=2e-6
    )
    
    # Evaluate model
    test_results = pipeline.evaluate('test')
    
    # Generate forecast
    forecast_results = pipeline.forecast(num_days=7)
    
    # Plot training history
    pipeline.plot_training_history()
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
