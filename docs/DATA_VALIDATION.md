# Data Validation Module

## Overview

The Data Validation Module provides comprehensive, production-grade validation for retail sales data before it enters the forecasting and inventory optimization pipeline. It ensures data quality, consistency, schema compliance, and business logic adherence.

---

## Components

### RetailDataValidator Class

Main validator class that performs comprehensive validation checks.

```python
from src.data_validation import RetailDataValidator

# Create validator instance
validator = RetailDataValidator(strict_mode=False)

# Validate sales data
is_valid, results = validator.validate_sales_data(df_sales)
```

#### Parameters

- **`strict_mode`** (bool):
  - `False` (default): Logs warnings and returns validation results
  - `True`: Raises exceptions on validation errors (production mode)

---

## Validation Methods

### 1. validate_sales_data(df)

Validates historical sales training data for schema and content quality.

**Checks Performed:**
- ✅ Required columns present
- ✅ Correct data types
- ✅ No missing values (or acceptable null rate)
- ✅ Non-negative sales values
- ✅ No duplicate records
- ✅ Valid date range (1-10 years)
- ✅ Outlier detection (IQR method)

**Expected Columns:**
```python
[
    'id',           # Record identifier
    'item_id',      # Product SKU
    'dept_id',      # Department identifier
    'cat_id',       # Category identifier
    'store_id',     # Store identifier
    'state_id',     # State identifier
    'd',            # Day counter
    'sales',        # Sales quantity (numeric)
    'date'          # Calendar date
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
    'sales': ['int64', 'float64'],
    'date': 'datetime64[ns]'
}
```

**Returns:**
```python
(is_valid: bool, results: Dict)
# Results contain boolean flags and metrics for each check
```

---

### 2. validate_calendar_data(df)

Validates calendar/date reference data integrity.

**Checks Performed:**
- ✅ Required columns present
- ✅ Date column is datetime type
- ✅ No duplicate dates
- ✅ Sequential date continuity

**Expected Columns:**
```python
[
    'd',              # Day identifier
    'date',           # Calendar date
    'wm_yr_wk',       # Walmart year-week
    'event_name_1'    # Event description
]
```

---

### 3. validate_prices_data(df)

Validates pricing data quality and consistency.

**Checks Performed:**
- ✅ Required columns present
- ✅ Prices are positive (> 0)
- ✅ No duplicate price records
- ✅ Price range validation

**Expected Columns:**
```python
[
    'store_id',      # Store identifier
    'item_id',       # Product SKU
    'wm_yr_wk',      # Walmart year-week
    'sell_price'     # Unit selling price
]
```

---

### 4. validate_merged_data(df)

Validates merged/processed data after joins and feature engineering.

**Checks Performed:**
- ✅ No columns entirely null
- ✅ Acceptable null rates for critical columns
- ✅ Adequate number of rows (≥ 100)
- ✅ Adequate number of columns (≥ 10)
- ✅ No data type inconsistencies

**Critical Column Thresholds:**
```python
{
    'sales': 0.05,      # Maximum 5% null tolerance
    'sell_price': 0.10  # Maximum 10% null tolerance
}
```

---

## Convenience Functions

### validate_all_data()

Validate all data sources at once with unified reporting.

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

---

## Usage Examples

### Example 1: Basic Validation

```python
import pandas as pd
from src.data_validation import RetailDataValidator

# Load data
df = pd.read_csv('sales_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Create validator and validate
validator = RetailDataValidator(strict_mode=False)
is_valid, results = validator.validate_sales_data(df)

# Process results
if is_valid:
    print("✅ All validation checks passed!")
    print(f"Records validated: {len(df)}")
else:
    print("⚠️ Validation warnings detected:")
    for check, result in results.items():
        if not result:
            print(f"  - {check}")
```

---

### Example 2: Strict Mode (Production)

```python
from src.data_validation import RetailDataValidator, DataValidationError

validator = RetailDataValidator(strict_mode=True)

try:
    is_valid, results = validator.validate_sales_data(df)
    logger.info("Data validation passed")
except DataValidationError as e:
    logger.error(f"Validation failed: {e}")
    # Handle error - halt pipeline
    raise
```

---

### Example 3: Full Pipeline Validation

```python
from src.data_validation import validate_all_data

# Validate all sources in one call
is_valid, all_results = validate_all_data(
    df_sales=sales_df,
    df_calendar=calendar_df,
    df_prices=prices_df,
    df_merged=merged_df,
    strict_mode=False
)

# Generate comprehensive report
for data_source, validation_result in all_results.items():
    print(f"\n{data_source}:")
    print(validation_result)
```

---

## Configuration

Validation thresholds can be customized in `config.py`:

```python
DATA_VALIDATION_CONFIG = {
    "date_range": {
        "min_years": 1,
        "max_years": 10
    },
    "null_rate_thresholds": {
        "sales": 0.05,          # 5% null tolerance
        "sell_price": 0.10,     # 10% null tolerance
    },
    "outlier_detection": {
        "method": "iqr",        # 'iqr' or 'zscore'
        "warning_threshold": 5  # Warn if > 5% outliers
    },
    "data_size": {
        "min_rows": 100,
        "min_columns": 10
    }
}
```

---

## Outlier Detection Methods

### IQR (Interquartile Range) - Recommended

```
ourliers = values < (Q1 - 1.5 × IQR) OR values > (Q3 + 1.5 × IQR)
```

**Characteristics:**
- Robust to extreme values
- Effective for skewed distributions
- Best for retail sales data

---

### Z-Score Method

```
ourliers = |z_score| > 3
```

**Characteristics:**
- Assumes normal distribution
- Sensitive to extreme values
- Use for known normal distributions

---

## Error Handling

### Custom Exception

```python
from src.data_validation import DataValidationError

try:
    validator.validate_sales_data(df)
except DataValidationError as e:
    logger.error(f"❌ Data validation failed: {e}")
    # Implement recovery logic
    raise
```

---

### Logging Integration

All validation checks are logged for audit trails:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Logs include:
# - Each validation check result (PASS/FAIL)
# - Missing columns
# - Data type mismatches
# - Outlier percentages
# - Null rate warnings
```

---

## Testing

### Run Full Test Suite

```bash
python -m pytest tests/test_data_validation.py -v
```

### Run Specific Test

```bash
python -m pytest tests/test_data_validation.py::TestDataValidationModule::test_valid_sales_data -v
```

### Generate Coverage Report

```bash
python -m pytest tests/ --cov=src/data_validation --cov-report=html
```

---

## Best Practices

### 1. Always Validate Before Processing
```python
is_valid, results = validator.validate_sales_data(df)
if not is_valid:
    logger.warning("Proceeding with data quality warnings")
```

### 2. Use Strict Mode in Production
```python
validator = RetailDataValidator(strict_mode=True)
```

### 3. Log Validation Metrics
```python
logger.info(f"Validation results: {results}")
report = validator.get_validation_report()
logger.info(f"Report: {report}")
```

### 4. Handle Missing Values Appropriately
- Document null handling strategy
- Use domain expertise for imputation
- Never silently drop data
- Track all transformations

### 5. Monitor & Investigate Outliers
- Investigate outliers before removal
- May indicate legitimate patterns or data quality issues
- Document all decisions
- Maintain audit trail

---

## Troubleshooting

### Issue: "Missing Required Columns"

**Solution:** Verify column names match expected schema:
```python
print("Available columns:")
print(df.columns.tolist())
```

---

### Issue: "Incorrect Data Types"

**Solution:** Convert columns to correct types:
```python
df['date'] = pd.to_datetime(df['date'])
df['sales'] = df['sales'].astype('int64')
```

---

### Issue: "Date Range Validation Failed"

**Solution:** Check date span:
```python
date_span = (df['date'].max() - df['date'].min()).days / 365.25
print(f"Data span: {date_span:.1f} years")
```

---

### Issue: High Outlier Percentage

**Solution:** Investigate outliers:
```python
Q1 = df['sales'].quantile(0.25)
Q3 = df['sales'].quantile(0.75)
IQR = Q3 - Q1
ourliers = df[(df['sales'] < Q1 - 1.5*IQR) | (df['sales'] > Q3 + 1.5*IQR)]
print(f"Outlier count: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
```

---

## References

- [Pandas Documentation](https://pandas.pydata.org/)
- [NumPy Documentation](https://numpy.org/)
- [Data Validation Best Practices](https://en.wikipedia.org/wiki/Data_validation)
- [IQR Method](https://en.wikipedia.org/wiki/Interquartile_range)

---

## Support

For issues or questions:
1. Review troubleshooting section above
2. Check test cases in `tests/test_data_validation.py`
3. Review log messages for specific errors
4. Create an issue on [GitHub](https://github.com/Madhuchandana-ux/Retail-sales-forecastory-and-inventory-optimization)
