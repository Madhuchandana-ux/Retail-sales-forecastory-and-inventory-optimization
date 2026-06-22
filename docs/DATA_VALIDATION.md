# Data Validation Module

## Overview

The Data Validation Module provides comprehensive validation for retail sales data before it enters the forecasting and inventory optimization pipeline. It ensures data quality, consistency, and compliance with business rules.

## Components

### RetailDataValidator Class

Main validator class that performs all validation checks.

```python
from src.data_validation import RetailDataValidator

# Create validator instance
validator = RetailDataValidator(strict_mode=False)

# Validate sales data
is_valid, results = validator.validate_sales_data(df_sales)
```

#### Parameters

- `strict_mode` (bool): 
  - `False` (default): Logs warnings and returns validation results
  - `True`: Raises exceptions on validation errors

## Validation Methods

### 1. validate_sales_data(df)

Validates sales training data.

**Checks:**
- ✓ Required columns present
- ✓ Correct data types
- ✓ No missing values
- ✓ Non-negative sales values
- ✓ No duplicate records
- ✓ Valid date range (1-10 years)
- ✓ Outlier detection (IQR method)

**Expected Columns:**
```python
[
    'id',           # Record identifier
    'item_id',      # Product identifier
    'dept_id',      # Department identifier
    'cat_id',       # Category identifier
    'store_id',     # Store identifier
    'state_id',     # State identifier
    'd',            # Day identifier
    'sales',        # Sales value (numeric)
    'date'          # Date (datetime)
]
```

**Expected Data Types:**
```python
{
    'id': 'object',
    'item_id': 'object',
    'dept_id': 'object',
    'cat_id': 'object',
    'store_id': 'object',
    'state_id': 'object',
    'sales': ['int64', 'float64'],  # Can be either
    'date': 'datetime64[ns]'
}
```

**Returns:**
```python
(is_valid: bool, results: Dict)
# Results contain boolean flags for each check
```

### 2. validate_calendar_data(df)

Validates calendar/date reference data.

**Checks:**
- ✓ Required columns present
- ✓ Date column is datetime type
- ✓ No duplicate dates

**Expected Columns:**
```python
[
    'd',              # Day identifier
    'date',           # Calendar date
    'wm_yr_wk',       # Walmart year-week
    'event_name_1'    # Event name
]
```

### 3. validate_prices_data(df)

Validates pricing data.

**Checks:**
- ✓ Required columns present
- ✓ Prices are positive (> 0)
- ✓ No duplicate price records by (store_id, item_id, wm_yr_wk)

**Expected Columns:**
```python
[
    'store_id',      # Store identifier
    'item_id',       # Item identifier
    'wm_yr_wk',      # Walmart year-week
    'sell_price'     # Selling price
]
```

### 4. validate_merged_data(df)

Validates merged/processed data after joins.

**Checks:**
- ✓ No columns are entirely null
- ✓ Acceptable null rates for critical columns
- ✓ Adequate number of rows (≥ 100)
- ✓ Adequate number of columns (≥ 10)

**Critical Column Thresholds:**
```python
{
    'sales': 0.05,      # Max 5% null
    'sell_price': 0.10  # Max 10% null
}
```

## Convenience Functions

### validate_all_data()

Validate all data sources at once.

```python
from src.data_validation import validate_all_data

is_valid, all_results = validate_all_data(
    df_sales=sales_data,
    df_calendar=calendar_data,
    df_prices=prices_data,
    df_merged=merged_data,      # Optional
    strict_mode=False
)
```

## Usage Examples

### Basic Validation

```python
import pandas as pd
from src.data_validation import RetailDataValidator

# Load data
df = pd.read_csv('sales_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Validate
validator = RetailDataValidator(strict_mode=False)
is_valid, results = validator.validate_sales_data(df)

# Check results
if is_valid:
    print("✓ All validation checks passed!")
else:
    print("✗ Validation failed")
    for check, result in results.items():
        print(f"  {check}: {result}")
```

### Strict Mode (Production)

```python
from src.data_validation import RetailDataValidator, DataValidationError

validator = RetailDataValidator(strict_mode=True)

try:
    is_valid, results = validator.validate_sales_data(df)
except DataValidationError as e:
    print(f"Validation error: {e}")
    # Handle error appropriately
```

### Full Pipeline Validation

```python
from src.data_validation import validate_all_data

# Validate all sources
is_valid, all_results = validate_all_data(
    df_sales=sales_df,
    df_calendar=calendar_df,
    df_prices=prices_df,
    df_merged=merged_df,
    strict_mode=False
)

# Get complete report
print(all_results)
```

## Configuration

Validation thresholds can be configured in `config.py`:

```python
DATA_VALIDATION_CONFIG = {
    "date_range": {
        "min_years": 1,
        "max_years": 10
    },
    "null_rate_thresholds": {
        "sales": 0.05,          # 5%
        "sell_price": 0.10,     # 10%
    },
    "outlier_detection": {
        "method": "iqr",
        "warning_threshold": 5  # Warn if > 5% outliers
    },
    "data_size": {
        "min_rows": 100,
        "min_columns": 10
    }
}
```

## Outlier Detection Methods

### IQR (Interquartile Range) - Default

```
outliers = values < (Q1 - 1.5 * IQR) OR values > (Q3 + 1.5 * IQR)
```

- Robust to extreme values
- Works well with skewed distributions
- Recommended for retail data

### Z-Score

```
outliers = |z_score| > 3
```

- Assumes normal distribution
- Sensitive to extreme values
- Use for normally distributed data

## Error Handling

### Custom Exception

```python
from src.data_validation import DataValidationError

try:
    validator.validate_sales_data(df)
except DataValidationError as e:
    logger.error(f"Data validation failed: {e}")
    # Recovery logic
```

### Logging

All validation checks are logged:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Logs will show:
# - Each validation check result
# - Missing columns
# - Data type mismatches
# - Outlier percentages
# - Null rate warnings
```

## Testing

Run the test suite:

```bash
python -m pytest tests/test_data_validation.py -v
```

Run specific test:

```bash
python -m pytest tests/test_data_validation.py::TestDataValidationModule::test_valid_sales_data -v
```

## Best Practices

1. **Always validate before processing**
   ```python
   is_valid, results = validator.validate_sales_data(df)
   if not is_valid:
       logger.warning("Proceeding with data quality warnings")
   ```

2. **Use strict mode in production**
   ```python
   validator = RetailDataValidator(strict_mode=True)
   ```

3. **Log validation results**
   ```python
   logger.info(f"Validation results: {results}")
   validator_report = validator.get_validation_report()
   ```

4. **Handle missing values appropriately**
   - Document null handling strategy
   - Use domain expertise for imputation
   - Never silently drop data

5. **Monitor outliers**
   - Investigate outliers before removal
   - May indicate data quality issues or legitimate patterns
   - Document decisions

## Troubleshooting

### Issue: "Missing required columns"

**Solution:** Verify column names match expected names:
```python
print(df.columns)
# Compare with required_cols in validate_sales_data()
```

### Issue: "Incorrect data types"

**Solution:** Convert columns to correct types:
```python
df['date'] = pd.to_datetime(df['date'])
df['sales'] = df['sales'].astype('int64')
```

### Issue: "Date range validation failed"

**Solution:** Check date span:
```python
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
date_span_years = (df['date'].max() - df['date'].min()).days / 365.25
print(f"Span: {date_span_years:.1f} years")
```

### Issue: High outlier percentage

**Solution:** Investigate outliers:
```python
Q1 = df['sales'].quantile(0.25)
Q3 = df['sales'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['sales'] < Q1 - 1.5*IQR) | (df['sales'] > Q3 + 1.5*IQR)]
print(f"Outlier count: {len(outliers)}")
```

## References

- [Pandas Documentation](https://pandas.pydata.org/)
- [NumPy Documentation](https://numpy.org/)
- [Data Validation Best Practices](https://en.wikipedia.org/wiki/Data_validation)

## Support

For issues or questions:
1. Check the documentation above
2. Review test cases in `tests/test_data_validation.py`
3. Check log messages for specific errors
4. Create an issue on GitHub
