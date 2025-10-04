import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch.cuda.amp as amp
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import os
import cv2
import pickle

H, W = 118, 310  # Original dimensions
TARGET_H, TARGET_W = 256, 256  # Target dimensions for pre-trained model
LAST_DATE=pd.to_datetime('2025-10-03')


# Working directories
WORK_DIR = Path('./data_cache')
days_dir = WORK_DIR / 'days_npy'
meta_path = WORK_DIR / 'days_metadata.csv'

class DaySequenceDataset(Dataset):
    def __init__(self, meta_df, lookback_days):
        self.meta_df = meta_df.reset_index(drop=True)
        self.lookback_days = lookback_days
    def __len__(self):
        return len(self.meta_df) - self.lookback_days
    def __getitem__(self, idx):
        files = self.meta_df['file'].iloc[idx:idx+self.lookback_days].tolist()
        x = np.stack([np.load(f) for f in files], axis=0)  # Shape: (LOOKBACK_DAYS, 256, 256)
        y = np.load(self.meta_df['file'].iloc[idx+self.lookback_days])  # Shape: (256, 256)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
    

class DataHandler():
    def __init__(self, scaler='scaler.pkl',):
        self.scaler_path = Path(scaler)
        self.LAST_DATE=LAST_DATE
        if self.scaler_path.exists():
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Loaded scaler from {self.scaler_path}")
        else:
            self.scaler=None
            print("No existing scaler found. Will create new one during data loading.")



    def load_csv(self, file="NO2.csv"):
        try:
            self.main_df = pd.read_csv(file)
            #convert to pd.datetime and sort
            df = self.main_df.copy()
            df['datetime'] = pd.to_datetime(df['datetime']) 
            df = df.sort_values('datetime').reset_index(drop=True)


            df = df.replace([np.inf, -np.inf], np.nan)

            # Fill NaN values in numeric columns with median
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if df[numeric_columns].isnull().sum().sum() > 0:
                print("Filling NaN values with median...")
                df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

            self.LAST_DATE = pd.to_datetime(df.iloc[-1]['datetime'])


            #handle indexing
            all_days = pd.date_range(df['datetime'].min().date(), df['datetime'].max().date(), freq="D")
            df_full = df.set_index('datetime').reindex(all_days).reset_index()
            df_full = df_full.rename(columns={'index': 'datetime'})

            #  IQR
            pixel_data = df.drop(columns=['datetime']).values
            self.Q1 = np.percentile(pixel_data, 25, axis=0)
            self.Q3 = np.percentile(pixel_data, 75, axis=0)
            self.IQR = self.Q3 - self.Q1
            self.lower = self.Q1 - 1.5 * self.IQR
            self.upper = self.Q3 + 1.5 * self.IQR
            pixel_data_capped = np.clip(pixel_data, self.lower, self.upper)
            pixel_data_capped = np.clip(pixel_data_capped, 0, None)

            df_capped = pd.DataFrame(pixel_data_capped, columns=df_full.columns[1:])
            df_capped.insert(0, 'datetime', df_full['datetime'])
            df=df_capped
            
            self.df=df

            # Interpolate missing values per pixel
            self.df.iloc[:,1:] = df.iloc[:,1:].interpolate(limit_direction='both')
            print("After filling gaps:", self.df.shape)

            self.scaler = MinMaxScaler()
            pixel_data_scaled = self.scaler.fit_transform(self.df.drop(columns=['datetime']))
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Fitted scaler and saved to {self.scaler_path}")
            self.df_scaled = pd.DataFrame(pixel_data_scaled, columns=self.df.columns[1:])
            self.df_scaled.insert(0, 'datetime', self.df['datetime'])

            if days_dir.exists():
                print('Cleaning existing days directory...')
                for f in days_dir.glob('*.npy'):
                    f.unlink()   # delete existing .npy files
            else:
                days_dir.mkdir(parents=True)

            dates = self.df_scaled['datetime']             # pandas Series of dates
            data = self.df_scaled.drop(columns=['datetime']).values  # numpy array of pixel values


            meta_rows = []
            for i, d in enumerate(dates):
                arr = data[i].reshape(H, W)
                # Resize to 256x256 using OpenCV
                arr_resized = cv2.resize(arr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
                fpath = days_dir/f"day_{i:04d}.npy"
                np.save(fpath, arr_resized.astype(np.float32))
                meta_rows.append([d, fpath.as_posix(), H, W])  # Store original dimensions

            self.meta_df = pd.DataFrame(meta_rows, columns=['datetime','file','original_h','original_w'])

            self.meta_df.to_csv(meta_path, index=False)
            print(f"Saved {len(self.meta_df)} resized days to {days_dir}")


        except FileNotFoundError:
            print(f"Error: File '{file}' not found. Please make sure the CSV file exists.")
            raise
        except pd.errors.EmptyDataError:
            print(f"Error: File '{file}' is empty or invalid.")
            raise
        except Exception as e:
            print(f"Error loading CSV file '{file}': {e}")
            raise


    def split(self, train_size=0.9, LOOKBACK_DAYS=7, test_size=3):
        try:
            if not hasattr(self, 'meta_df') or self.meta_df is None:
                raise ValueError("Data must be loaded first using load_csv()")
                
            total_samples = len(self.meta_df) - LOOKBACK_DAYS  # Available samples for training
            if total_samples <= 0:
                raise ValueError(f"Not enough data. Need at least {LOOKBACK_DAYS + 1} days, got {len(self.meta_df)}")
            
            train_size_idx = int(len(self.meta_df) * train_size)
            val_end_idx = len(self.meta_df) - test_size
            
            # Ensure we have enough data for each split
            if train_size_idx >= val_end_idx:
                raise ValueError("Not enough data for train/val/test split with current parameters")
            
            train_meta = self.meta_df.iloc[:train_size_idx]
            val_meta = self.meta_df.iloc[train_size_idx:val_end_idx]
            test_meta = self.meta_df.iloc[val_end_idx:]
            
            # Check if we have enough data for each dataset
            if len(train_meta) < LOOKBACK_DAYS + 1:
                raise ValueError(f"Not enough training data. Need at least {LOOKBACK_DAYS + 1} samples, got {len(train_meta)}")
            if len(val_meta) < LOOKBACK_DAYS + 1:
                raise ValueError(f"Not enough validation data. Need at least {LOOKBACK_DAYS + 1} samples, got {len(val_meta)}")
            if len(test_meta) < LOOKBACK_DAYS + 1:
                raise ValueError(f"Not enough test data. Need at least {LOOKBACK_DAYS + 1} samples, got {len(test_meta)}")

            train_dataset = DaySequenceDataset(train_meta, LOOKBACK_DAYS)
            val_dataset = DaySequenceDataset(val_meta, LOOKBACK_DAYS)
            test_dataset = DaySequenceDataset(test_meta, LOOKBACK_DAYS)
            
            return train_dataset, val_dataset, test_dataset
        except Exception as e:
            print(f"Unable to split the data: {e}")
            return None, None, None
    
    def add_seq(self, dates, rows , file="NO2.csv"):
        df = pd.read_csv(file)
        for date, row in zip(dates, rows):
            date = pd.to_datetime(date)
            if date in df['datetime'].values:
                print(f"Date {date} already exists. Skipping.")
                continue
            new_row = {'datetime': date}
            for i, val in enumerate(row):
                new_row[f'pixel_{i}'] = val
            row = pd.DataFrame([new_row])
            pixel_data = row.drop(columns=['datetime']).values
            pixel_data_capped = np.clip(pixel_data, self.lower, self.upper)
            pixel_data_capped = np.clip(pixel_data_capped, 0, None)

            row_capped = pd.DataFrame(pixel_data_capped, columns=row.columns[1:])
            row_capped.insert(0, 'datetime', row['datetime'])
            df = pd.concat([df, row_capped], ignore_index=True)
        df = df.sort_values('datetime').reset_index(drop=True)

        #handle indexing
        all_days = pd.date_range(df['datetime'].min().date(), df['datetime'].max().date(), freq="D")
        df_full = df.set_index('datetime').reindex(all_days).reset_index()
        df_full = df_full.rename(columns={'index': 'datetime'})
        df_full.iloc[:,1:] = df_full.iloc[:,1:].interpolate(limit_direction='both')
        df_full.to_csv(file, index=False)

        start_idx = len(self.meta_df)
        
        meta_rows = []
        for i, (date, row) in enumerate(zip(dates, rows)):
            arr = row.values.reshape(H, W)
            # Resize to 256x256 using OpenCV
            arr_resized = cv2.resize(arr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
            fpath = days_dir/f"day_{start_idx + i:04d}.npy"
            np.save(fpath, arr_resized.astype(np.float32))
            meta_rows.append([date, fpath.as_posix(), H, W])  # Store original dimensions

        new_meta_df = pd.DataFrame(meta_rows, columns=['datetime','file','original_h','original_w'])
        self.meta_df = pd.concat([self.meta_df, new_meta_df], ignore_index=True)
        self.meta_df.to_csv(meta_path, index=False)
        print(f"Added {len(new_meta_df)} new days and updated metadata.")

        self.LAST_DATE = pd.to_datetime(dates[-1])

