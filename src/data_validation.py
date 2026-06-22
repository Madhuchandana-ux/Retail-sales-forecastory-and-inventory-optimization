"""
Data Validation Module for Retail Sales Forecasting and Inventory Optimization

This module provides comprehensive data validation functions to ensure data quality
before processing through the forecasting and inventory optimization pipeline.
It includes checks for data types, missing values, outliers, and business logic rules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass


class RetailDataValidator:
    """
    Comprehensive data validator for retail sales data.
    
    Validates data structure, types, ranges, and business logic rules
    to ensure data quality for forecasting and inventory optimization.
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the validator.
        
        Args:
            strict_mode: If True, raises exceptions on validation errors.
                        If False, logs warnings and returns validation results.
        """
        self.strict_mode = strict_mode
        self.validation_results = {}
    
    def validate_sales_data(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validate sales training data.
        
        Args:
            df: Sales dataframe from data_ingestion
            
        Returns:
            Tuple of (is_valid: bool, results: Dict with validation details)
        """
        results = {}
        
        # Check required columns
        required_cols = [
            'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 
            'state_id', 'd', 'sales', 'date'
        ]
        results['has_required_columns'] = self._check_required_columns(
            df, required_cols
        )
        
        # Check data types
        results['correct_dtypes'] = self._validate_dtypes(
            df, 
            {
                'id': 'object',
                'item_id': 'object',
                'dept_id': 'object',
                'cat_id': 'object',
                'store_id': 'object',
                'state_id': 'object',
                'sales': ['int64', 'float64'],
                'date': 'datetime64[ns]'
            }
        )
        
        # Check for missing values
        results['missing_values'] = self._check_missing_values(df)
        
        # Check sales values are non-negative
        results['non_negative_sales'] = self._check_non_negative(df, 'sales')
        
        # Check for duplicates
        results['no_duplicates'] = self._check_duplicates(
            df, 
            subset=['id', 'date']
        )
        
        # Check date range is reasonable (e.g., past 10 years)
        results['valid_date_range'] = self._check_date_range(
            df['date'],
            min_years=1,
            max_years=10
        )
        
        # Check for outliers in sales
        results['outlier_check'] = self._check_outliers(
            df, 'sales', method='iqr'
        )
        
        is_valid = all(v for v in results.values() if isinstance(v, bool))
        
        logger.info(f"Sales data validation: {is_valid}")
        self.validation_results['sales_data'] = results
        
        return is_valid, results
    
    def validate_calendar_data(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validate calendar data.
        
        Args:
            df: Calendar dataframe
            
        Returns:
            Tuple of (is_valid: bool, results: Dict)
        """
        results = {}
        
        required_cols = ['d', 'date', 'wm_yr_wk', 'event_name_1']
        results['has_required_columns'] = self._check_required_columns(
            df, required_cols
        )
        
        results['date_is_datetime'] = pd.api.types.is_datetime64_any_dtype(
            df['date']
        )
        
        results['no_duplicate_dates'] = not df['date'].duplicated().any()
        
        is_valid = all(v for v in results.values() if isinstance(v, bool))
        
        logger.info(f"Calendar data validation: {is_valid}")
        self.validation_results['calendar_data'] = results
        
        return is_valid, results
    
    def validate_prices_data(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validate prices data.
        
        Args:
            df: Prices dataframe
            
        Returns:
            Tuple of (is_valid: bool, results: Dict)
        """
        results = {}
        
        required_cols = ['store_id', 'item_id', 'wm_yr_wk', 'sell_price']
        results['has_required_columns'] = self._check_required_columns(
            df, required_cols
        )
        
        results['positive_prices'] = self._check_positive(df, 'sell_price')
        
        results['no_duplicate_price_keys'] = not df.duplicated(
            subset=['store_id', 'item_id', 'wm_yr_wk']
        ).any()
        
        is_valid = all(v for v in results.values() if isinstance(v, bool))
        
        logger.info(f"Prices data validation: {is_valid}")
        self.validation_results['prices_data'] = results
        
        return is_valid, results
    
    def validate_merged_data(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validate merged/processed data after joins.
        
        Args:
            df: Merged dataframe
            
        Returns:
            Tuple of (is_valid: bool, results: Dict)
        """
        results = {}
        
        # Check if merge was successful
        results['no_all_nulls_after_merge'] = not df.isna().all().any()
        
        # Check critical columns have acceptable null rates
        critical_cols = {'sales': 0.05, 'sell_price': 0.10}  # 5%, 10% thresholds
        results['acceptable_null_rates'] = self._check_null_rates(
            df, critical_cols
        )
        
        # Validate data size
        results['adequate_rows'] = len(df) > 100
        results['adequate_columns'] = len(df.columns) >= 10
        
        is_valid = all(v for v in results.values() if isinstance(v, bool))
        
        logger.info(f"Merged data validation: {is_valid}")
        self.validation_results['merged_data'] = results
        
        return is_valid, results
    
    # -------------------------
    # Helper Methods
    # -------------------------
    
    def _check_required_columns(
        self, 
        df: pd.DataFrame, 
        required_cols: List[str]
    ) -> bool:
        """Check if all required columns are present"""
        missing = set(required_cols) - set(df.columns)
        if missing:
            msg = f"Missing required columns: {missing}"
            self._handle_error(msg)
            return False
        return True
    
    def _validate_dtypes(
        self, 
        df: pd.DataFrame, 
        dtype_dict: Dict[str, str]
    ) -> Dict[str, bool]:
        """Validate data types of columns"""
        results = {}
        
        for col, expected_dtype in dtype_dict.items():
            if col not in df.columns:
                results[col] = False
                continue
            
            # Handle multiple acceptable types
            if isinstance(expected_dtype, list):
                is_valid = str(df[col].dtype) in expected_dtype
            else:
                is_valid = str(df[col].dtype) == expected_dtype
            
            results[col] = is_valid
            
            if not is_valid:
                msg = (f"Column '{col}' has dtype {df[col].dtype}, "
                       f"expected {expected_dtype}")
                logger.warning(msg)
        
        return all(results.values())
    
    def _check_missing_values(self, df: pd.DataFrame) -> bool:
        """Check for missing values"""
        missing_count = df.isna().sum().sum()
        
        if missing_count > 0:
            missing_pct = (missing_count / (df.shape[0] * df.shape[1])) * 100
            msg = f"Found {missing_count} missing values ({missing_pct:.2f}%)"
            logger.warning(msg)
            
            # Log per-column missing rates
            missing_by_col = df.isna().sum()
            for col, count in missing_by_col[missing_by_col > 0].items():
                pct = (count / len(df)) * 100
                logger.warning(f"  {col}: {count} missing ({pct:.2f}%)")
        
        return missing_count == 0
    
    def _check_non_negative(self, df: pd.DataFrame, col: str) -> bool:
        """Check if column has non-negative values"""
        if col not in df.columns:
            return False
        
        negative_count = (df[col] < 0).sum()
        
        if negative_count > 0:
            msg = f"Column '{col}' has {negative_count} negative values"
            self._handle_error(msg)
            return False
        
        return True
    
    def _check_positive(self, df: pd.DataFrame, col: str) -> bool:
        """Check if column has positive values"""
        if col not in df.columns:
            return False
        
        non_positive = (df[col] <= 0).sum()
        
        if non_positive > 0:
            msg = f"Column '{col}' has {non_positive} non-positive values"
            self._handle_error(msg)
            return False
        
        return True
    
    def _check_duplicates(
        self, 
        df: pd.DataFrame, 
        subset: Optional[List[str]] = None
    ) -> bool:
        """Check for duplicate rows"""
        dup_count = df.duplicated(subset=subset).sum()
        
        if dup_count > 0:
            msg = f"Found {dup_count} duplicate rows (subset={subset})"
            logger.warning(msg)
            return False
        
        return True
    
    def _check_date_range(
        self, 
        dates: pd.Series, 
        min_years: int = 1, 
        max_years: int = 10
    ) -> bool:
        """Check if date range is within acceptable bounds"""
        if not pd.api.types.is_datetime64_any_dtype(dates):
            logger.warning("Date column is not datetime type")
            return False
        
        date_range = (dates.max() - dates.min()).days / 365.25
        
        if date_range < min_years:
            msg = f"Date range {date_range:.1f} years is less than {min_years}"
            logger.warning(msg)
            return False
        
        if date_range > max_years:
            msg = f"Date range {date_range:.1f} years exceeds {max_years}"
            logger.warning(msg)
            return False
        
        return True
    
    def _check_outliers(
        self, 
        df: pd.DataFrame, 
        col: str, 
        method: str = 'iqr'
    ) -> Dict[str, any]:
        """
        Detect outliers using specified method.
        
        Args:
            df: DataFrame
            col: Column to check
            method: 'iqr' for Interquartile Range or 'zscore' for Z-score
            
        Returns:
            Dict with outlier information
        """
        if col not in df.columns:
            return {'valid': False, 'reason': f'Column {col} not found'}
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = df[z_scores > 3]
        
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df)) * 100
        
        result = {
            'valid': True,
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_pct,
            'method': method
        }
        
        if outlier_pct > 5:  # Warn if > 5% outliers
            logger.warning(
                f"Column '{col}' has {outlier_pct:.2f}% outliers"
            )
        
        return result
    
    def _check_null_rates(
        self, 
        df: pd.DataFrame, 
        col_thresholds: Dict[str, float]
    ) -> bool:
        """
        Check that null rates are within acceptable thresholds.
        
        Args:
            df: DataFrame
            col_thresholds: Dict mapping column names to max null rate (0-1)
            
        Returns:
            True if all columns meet thresholds
        """
        all_valid = True
        
        for col, threshold in col_thresholds.items():
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found")
                all_valid = False
                continue
            
            null_rate = df[col].isna().sum() / len(df)
            
            if null_rate > threshold:
                msg = (f"Column '{col}' null rate {null_rate:.2%} "
                       f"exceeds threshold {threshold:.2%}")
                logger.warning(msg)
                all_valid = False
        
        return all_valid
    
    def _handle_error(self, msg: str):
        """Handle validation error based on strict_mode setting"""
        if self.strict_mode:
            raise DataValidationError(msg)
        else:
            logger.warning(msg)
    
    def get_validation_report(self) -> Dict:
        """Get complete validation report"""
        return self.validation_results


# Convenience functions
def validate_all_data(
    df_sales: pd.DataFrame,
    df_calendar: pd.DataFrame,
    df_prices: pd.DataFrame,
    df_merged: Optional[pd.DataFrame] = None,
    strict_mode: bool = False
) -> Tuple[bool, Dict]:
    """
    Validate all data sources at once.
    
    Args:
        df_sales: Sales data
        df_calendar: Calendar data
        df_prices: Prices data
        df_merged: Optional merged dataframe
        strict_mode: If True, raise exceptions on validation errors
        
    Returns:
        Tuple of (overall_valid: bool, all_results: Dict)
    """
    validator = RetailDataValidator(strict_mode=strict_mode)
    
    results = {}
    
    _, results['sales'] = validator.validate_sales_data(df_sales)
    _, results['calendar'] = validator.validate_calendar_data(df_calendar)
    _, results['prices'] = validator.validate_prices_data(df_prices)
    
    if df_merged is not None:
        _, results['merged'] = validator.validate_merged_data(df_merged)
    
    # Overall validation: all subsystem checks must pass
    overall_valid = all(
        all(v for v in subset.values() if isinstance(v, bool))
        for subset in results.values()
    )
    
    logger.info(f"Overall validation result: {overall_valid}")
    
    return overall_valid, results


if __name__ == "__main__":
    # Example usage
    print("Data validation module loaded successfully")
    print(f"Available validator class: {RetailDataValidator.__name__}")
    print(f"Available functions: validate_all_data, DataValidationError")
