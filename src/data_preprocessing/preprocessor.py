"""
Main Data Preprocessor Class

Handles the complete preprocessing pipeline for sensor data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from scipy import signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .filters import SignalFilter
from .normalizers import DataNormalizer
from data_preprocessing.m5_parser import parse_m5_sensor_data


class DataPreprocessor:
    """
    Main class for preprocessing sensor data in the ADL classification pipeline.
    
    Handles filtering, normalization, and basic data cleaning operations.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the preprocessor with configuration parameters.
        
        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.filter = SignalFilter(config.get('preprocessing', {}))
        self.normalizer = DataNormalizer(config.get('preprocessing', {}))
        
        # Track preprocessing statistics
        self.stats = {}
        
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            data: Raw sensor data DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("Starting data preprocessing pipeline")
        
        # 1. Basic data cleaning
        data_clean = self._clean_data(data)
        
        # 2. Apply filtering
        data_filtered = self.filter.apply_filter(data_clean)
        
        # 3. Normalize data
        data_normalized = self.normalizer.normalize(data_filtered)
        
        # 4. Handle missing values
        data_final = self._handle_missing_values(data_normalized)
        
        self.logger.info("Preprocessing pipeline completed")
        return data_final
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Basic data cleaning operations.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Remove obvious outliers (e.g., sensor values beyond physical limits)
        cleaned_data = data.copy()
        
        # Example: Remove IMU values beyond reasonable limits
        for col in cleaned_data.columns:
            if 'accel' in col.lower():
                # Accelerometer values typically within ±20g
                cleaned_data = cleaned_data[
                    (cleaned_data[col] >= -200) & (cleaned_data[col] <= 200)
                ]
            elif 'gyro' in col.lower():
                # Gyroscope values typically within ±2000 deg/s
                cleaned_data = cleaned_data[
                    (cleaned_data[col] >= -2000) & (cleaned_data[col] <= 2000)
                ]
        
        self.logger.info(f"Data cleaning: {len(data) - len(cleaned_data)} outliers removed")
        return cleaned_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data: Input DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        # Strategy: Forward fill for short gaps, interpolation for longer gaps
        data_filled = data.copy()
        
        for col in data_filled.columns:
            # Forward fill for gaps <= 3 samples
            data_filled[col] = data_filled[col].fillna(method='ffill', limit=3)
            
            # Linear interpolation for remaining gaps
            data_filled[col] = data_filled[col].interpolate(method='linear')
        
        # Drop rows that still have NaN values
        initial_len = len(data_filled)
        data_filled = data_filled.dropna()
        
        if len(data_filled) < initial_len:
            self.logger.warning(f"Dropped {initial_len - len(data_filled)} rows with missing values")
        
        return data_filled
    
    def get_preprocessing_stats(self) -> Dict:
        """
        Get statistics about the preprocessing operation.
        
        Returns:
            Dictionary containing preprocessing statistics
        """
        return self.stats 