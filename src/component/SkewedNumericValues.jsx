import React from 'react';
import ProblemTemplate from './ProblemTemplate';

const SkewedNumericValues = () => {
  const data = {
    title: 'Skewed Numeric Values',
    description: 'Skewed numeric distributions have most values concentrated on one side with a long tail on the other. This violates assumptions of many statistical methods and ML algorithms, affects model performance, and makes outliers more influential. Common in income, prices, website visits, and other real-world metrics.',
    originalData: [
      { transaction_id: 'T001', amount: 50, visits: 5, response_time_ms: 120, income: 45000 },
      { transaction_id: 'T002', amount: 75, visits: 3, response_time_ms: 95, income: 52000 },
      { transaction_id: 'T003', amount: 100, visits: 8, response_time_ms: 150, income: 48000 },
      { transaction_id: 'T004', amount: 65, visits: 2, response_time_ms: 85, income: 55000 },
      { transaction_id: 'T005', amount: 5000, visits: 120, response_time_ms: 2500, income: 250000 },
      { transaction_id: 'T006', amount: 80, visits: 6, response_time_ms: 110, income: 51000 },
      { transaction_id: 'T007', amount: 90, visits: 4, response_time_ms: 130, income: 49000 },
      { transaction_id: 'T008', amount: 12000, visits: 350, response_time_ms: 5000, income: 500000 },
      { transaction_id: 'T009', amount: 70, visits: 5, response_time_ms: 100, income: 47000 },
      { transaction_id: 'T010', amount: 85, visits: 7, response_time_ms: 140, income: 53000 },
    ],
    cleanedData: [
      { transaction_id: 'T001', amount_log: 3.91, visits_log: 1.79, response_time_log: 4.79, income_log: 10.71 },
      { transaction_id: 'T002', amount_log: 4.32, visits_log: 1.39, response_time_log: 4.55, income_log: 10.86 },
      { transaction_id: 'T003', amount_log: 4.61, visits_log: 2.20, response_time_log: 5.01, income_log: 10.78 },
      { transaction_id: 'T004', amount_log: 4.17, visits_log: 0.69, response_time_log: 4.44, income_log: 10.91 },
      { transaction_id: 'T005', amount_log: 8.52, visits_log: 4.79, response_time_log: 7.82, income_log: 12.43 },
      { transaction_id: 'T006', amount_log: 4.38, visits_log: 1.95, response_time_log: 4.70, income_log: 10.84 },
      { transaction_id: 'T007', amount_log: 4.50, visits_log: 1.61, response_time_log: 4.87, income_log: 10.80 },
      { transaction_id: 'T008', amount_log: 9.39, visits_log: 5.86, response_time_log: 8.52, income_log: 13.12 },
      { transaction_id: 'T009', amount_log: 4.25, visits_log: 1.79, response_time_log: 4.61, income_log: 10.76 },
      { transaction_id: 'T010', amount_log: 4.44, visits_log: 2.08, response_time_log: 4.94, income_log: 10.88 },
    ],
    removedColumns: [],
    testDataset: `import pandas as pd
import numpy as np
from scipy import stats

# Create test dataset with skewed distributions
np.random.seed(42)

data = {
    'transaction_id': [f'T{str(i).zfill(3)}' for i in range(1, 11)],
    'amount': [50, 75, 100, 65, 5000, 80, 90, 12000, 70, 85],  # Highly skewed
    'visits': [5, 3, 8, 2, 120, 6, 4, 350, 5, 7],  # Highly skewed
    'response_time_ms': [120, 95, 150, 85, 2500, 110, 130, 5000, 100, 140],  # Skewed
    'income': [45000, 52000, 48000, 55000, 250000, 51000, 49000, 500000, 47000, 53000]  # Skewed
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print(f"\\nDataset Shape: {df.shape}")

# Calculate skewness for numeric columns
print("\\n" + "="*70)
print("SKEWNESS ANALYSIS")
print("="*70)
print("\\nSkewness values (|skew| > 1 indicates high skewness):")

for col in ['amount', 'visits', 'response_time_ms', 'income']:
    skewness = df[col].skew()
    print(f"{col}: {skewness:.2f} {'(HIGHLY SKEWED)' if abs(skewness) > 1 else '(MODERATE)' if abs(skewness) > 0.5 else ''}")

print("\\nBasic Statistics:")
print(df[['amount', 'visits', 'response_time_ms', 'income']].describe())`,
    solution: `# Solution: Handle skewed numeric values using transformations

def handle_skewed_distributions(df, columns=None, skew_threshold=1.0, method='log'):
    """
    Detect and transform skewed numeric distributions
    
    Parameters:
    df: pandas DataFrame
    columns: list of numeric columns to check (None = all numeric)
    skew_threshold: float, absolute skewness above this is considered skewed (default: 1.0)
    method: str, transformation method
           - 'log': Natural logarithm (log(x+1))
           - 'sqrt': Square root
           - 'boxcox': Box-Cox transformation
           - 'yeo-johnson': Yeo-Johnson transformation (handles negative values)
    
    Returns:
    transformed_df: DataFrame with transformed columns
    skewness_report: Dictionary with skewness analysis
    """
    import pandas as pd
    import numpy as np
    from scipy import stats
    
    df_transformed = df.copy()
    skewness_report = {}
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print("="*70)
    print("SKEWNESS DETECTION AND TRANSFORMATION")
    print("="*70)
    print(f"Skewness threshold: {skew_threshold}")
    print(f"Transformation method: {method}")
    
    for col in columns:
        if col not in df.columns:
            print(f"\\n⚠ Column '{col}' not found, skipping")
            continue
        
        # Calculate original skewness
        original_skew = df[col].skew()
        
        print(f"\\n{col}:")
        print(f"  Original skewness: {original_skew:.3f}")
        
        # Check if skewed
        if abs(original_skew) > skew_threshold:
            print(f"  ⚠ SKEWED (|skew| > {skew_threshold})")
            
            # Determine skew direction
            if original_skew > 0:
                print(f"  Direction: Right-skewed (positive skew)")
            else:
                print(f"  Direction: Left-skewed (negative skew)")
            
            # Apply transformation
            new_col_name = f"{col}_{method}"
            
            try:
                if method == 'log':
                    # Add 1 to handle zeros, then log
                    df_transformed[new_col_name] = np.log1p(df[col])
                    
                elif method == 'sqrt':
                    # Square root (only for non-negative)
                    if (df[col] < 0).any():
                        print(f"  ✗ Cannot apply sqrt to negative values")
                        continue
                    df_transformed[new_col_name] = np.sqrt(df[col])
                    
                elif method == 'boxcox':
                    # Box-Cox (requires positive values)
                    if (df[col] <= 0).any():
                        print(f"  ✗ Box-Cox requires positive values, using Yeo-Johnson instead")
                        df_transformed[new_col_name], _ = stats.yeojohnson(df[col])
                    else:
                        df_transformed[new_col_name], _ = stats.boxcox(df[col])
                    
                elif method == 'yeo-johnson':
                    # Yeo-Johnson (works with any values)
                    df_transformed[new_col_name], _ = stats.yeojohnson(df[col])
                
                # Calculate new skewness
                new_skew = df_transformed[new_col_name].skew()
                improvement = abs(original_skew) - abs(new_skew)
                
                print(f"  ✓ Transformed: {new_col_name}")
                print(f"  New skewness: {new_skew:.3f}")
                print(f"  Improvement: {improvement:.3f} (closer to 0 is better)")
                
                # Store report
                skewness_report[col] = {
                    'original_skew': original_skew,
                    'new_skew': new_skew,
                    'improvement': improvement,
                    'transformed_column': new_col_name,
                    'method': method
                }
                
            except Exception as e:
                print(f"  ✗ Transformation failed: {str(e)}")
        else:
            print(f"  ✓ Not skewed (within threshold)")
    
    return df_transformed, skewness_report


# Apply log transformation (most common)
print("\\n" + "#"*70)
print("# METHOD 1: LOG TRANSFORMATION")
print("#"*70)

transformed_df_log, report_log = handle_skewed_distributions(
    df,
    columns=['amount', 'visits', 'response_time_ms', 'income'],
    skew_threshold=1.0,
    method='log'
)

print("\\n" + "="*70)
print("TRANSFORMATION SUMMARY")
print("="*70)
for col, info in report_log.items():
    print(f"\\n{col}:")
    print(f"  Original skewness: {info['original_skew']:.3f}")
    print(f"  After {info['method']}: {info['new_skew']:.3f}")
    print(f"  Improvement: {info['improvement']:.3f}")

# Show before/after comparison
print("\\n" + "="*70)
print("BEFORE vs AFTER - STATISTICS")
print("="*70)

for col in ['amount', 'visits', 'response_time_ms', 'income']:
    if col in report_log:
        new_col = report_log[col]['transformed_column']
        print(f"\\n{col}:")
        print("BEFORE:")
        print(f"  Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}, Std: {df[col].std():.2f}")
        print(f"  Min: {df[col].min():.2f}, Max: {df[col].max():.2f}")
        print("AFTER:")
        print(f"  Mean: {transformed_df_log[new_col].mean():.2f}, Median: {transformed_df_log[new_col].median():.2f}, Std: {transformed_df_log[new_col].std():.2f}")
        print(f"  Min: {transformed_df_log[new_col].min():.2f}, Max: {transformed_df_log[new_col].max():.2f}")

print("\\n\\nTransformed Dataset (Log):")
display_cols = ['transaction_id'] + [report_log[c]['transformed_column'] for c in report_log.keys()]
print(transformed_df_log[display_cols])

# Compare multiple transformation methods
print("\\n" + "#"*70)
print("# COMPARING TRANSFORMATION METHODS")
print("#"*70)

methods = ['log', 'sqrt', 'yeo-johnson']
comparison = {}

for method in methods:
    print(f"\\nTrying {method} transformation...")
    _, report = handle_skewed_distributions(
        df,
        columns=['amount'],
        skew_threshold=0,  # Transform regardless
        method=method
    )
    if 'amount' in report:
        comparison[method] = report['amount']['new_skew']

if comparison:
    print("\\n" + "="*70)
    print("METHOD COMPARISON FOR 'amount' COLUMN")
    print("="*70)
    print(f"Original skewness: {df['amount'].skew():.3f}")
    for method, skew in comparison.items():
        print(f"{method}: {skew:.3f}")
    
    best_method = min(comparison.items(), key=lambda x: abs(x[1]))
    print(f"\\nBest method: {best_method[0]} (skewness closest to 0)")`,
    explanation: `**What is Skewness?**

Skewness measures the asymmetry of a distribution:
- **Skewness = 0**: Symmetric (normal distribution)
- **Skewness > 0**: Right-skewed (positive skew) - long tail on right
- **Skewness < 0**: Left-skewed (negative skew) - long tail on left

**Interpretation:**
- **|skew| < 0.5**: Fairly symmetric
- **0.5 < |skew| < 1**: Moderately skewed
- **|skew| > 1**: Highly skewed ⚠️

**Why Skewed Data is Problematic:**

1. **Violates Assumptions**: Many algorithms assume normal distribution
   - Linear Regression
   - ANOVA
   - t-tests
   
2. **Mean vs Median**: Mean pulled toward tail, not representative
   - Income: Mean $100k, Median $50k (skewed by billionaires)

3. **Outlier Sensitivity**: Extreme values dominate
   - Few large values heavily influence statistics

4. **Poor Model Performance**: 
   - Algorithms struggle with different scales
   - Predictions biased toward common values

5. **Gradient Descent Issues**: Large values cause instability

6. **Feature Importance**: Skewed features dominate due to scale

**Common Skewed Variables:**

**Right-Skewed (Most Common):**
- Income/Salary
- House prices
- Website visits/clicks
- Response times
- Transaction amounts
- Age (in some contexts)
- Product ratings (ceiling effect)

**Left-Skewed:**
- Test scores (floor effect)
- Survival times
- Age at death

**Detection Methods:**

\`\`\`python
# Method 1: Calculate skewness
df['column'].skew()

# Method 2: Visual inspection
import matplotlib.pyplot as plt
df['column'].hist(bins=50)
plt.show()

# Method 3: Q-Q plot
from scipy import stats
stats.probplot(df['column'], dist="norm", plot=plt)

# Method 4: Shapiro-Wilk test for normality
stat, p_value = stats.shapiro(df['column'])
# p < 0.05 suggests non-normal (possibly skewed)
\`\`\`

**Transformation Methods:**

**1. Log Transformation** ⭐ MOST COMMON

\`\`\`python
# Natural log (handles right skew)
df['log_col'] = np.log(df['col'])

# Log with +1 (handles zeros)
df['log_col'] = np.log1p(df['col'])  # log(x + 1)

# Log base 10
df['log10_col'] = np.log10(df['col'])
\`\`\`

**When to use:**
- Strong right skew (skew > 1)
- Positive values only
- Multiplicative relationships

**Advantages:**
✓ Simple and interpretable
✓ Handles wide range of values
✓ Works well for many distributions

**Disadvantages:**
✗ Can't handle zeros (use log1p)
✗ Can't handle negatives

**2. Square Root Transformation**

\`\`\`python
df['sqrt_col'] = np.sqrt(df['col'])
\`\`\`

**When to use:**
- Moderate right skew (0.5 < skew < 1)
- Count data
- Positive values

**Advantages:**
✓ Less aggressive than log
✓ Handles zeros naturally

**Disadvantages:**
✗ Can't handle negatives

**3. Box-Cox Transformation**

\`\`\`python
from scipy import stats
transformed, lambda_param = stats.boxcox(df['col'])
\`\`\`

**When to use:**
- Any right skew
- Positive values only
- Want optimal transformation

**Advantages:**
✓ Finds optimal power transformation
✓ Maximizes normality

**Disadvantages:**
✗ Requires positive values
✗ Less interpretable (uses lambda parameter)

**4. Yeo-Johnson Transformation**

\`\`\`python
from scipy import stats
transformed, lambda_param = stats.yeojohnson(df['col'])
\`\`\`

**When to use:**
- Any skewness direction
- Can handle negative values
- Want optimal transformation

**Advantages:**
✓ Works with any values (including negatives)
✓ Finds optimal transformation

**Disadvantages:**
✗ Less interpretable

**5. Reciprocal Transformation**

\`\`\`python
df['reciprocal'] = 1 / df['col']
\`\`\`

**When to use:**
- Strong right skew
- Rates or ratios

**6. Cube Root**

\`\`\`python
df['cbrt'] = np.cbrt(df['col'])
\`\`\`

**When to use:**
- Handles negative values
- Less aggressive than log

**Decision Framework:**

\`\`\`
Check skewness value:

|skew| < 0.5 → No transformation needed

0.5 < |skew| < 1 → Try sqrt or mild transformation

1 < |skew| < 2 → Use log transformation

|skew| > 2 → Try Box-Cox or Yeo-Johnson

Has zeros? → Use log1p instead of log

Has negatives? → Use Yeo-Johnson or cube root

Count data? → Use sqrt or log
\`\`\`

**Best Practices:**

1. **Transform Before Modeling:**
   \`\`\`python
   # In scikit-learn pipeline
   from sklearn.preprocessing import PowerTransformer
   transformer = PowerTransformer(method='yeo-johnson')
   \`\`\`

2. **Keep Original Column:**
   Create new transformed column, don't overwrite

3. **Apply to Train and Test:**
   Use same transformation on test data

4. **Check Distribution After:**
   Verify skewness reduced
   \`\`\`python
   print(f"Before: {df['col'].skew():.2f}")
   print(f"After: {df['col_log'].skew():.2f}")
   \`\`\`

5. **Document Transformation:**
   Note which method was used for reproducibility

6. **Consider Domain Context:**
   Some skewness is natural (income distribution)

**When NOT to Transform:**

- Tree-based models (Random Forest, XGBoost)
  - These handle skewness well naturally
  
- When skewness is meaningful
  - Income inequality studies
  - Long-tail distributions by design

- Very small datasets
  - Transformations less effective with < 30 samples

- Interpretability is critical
  - Log-transformed predictions harder to explain

**Inverse Transformation:**

After modeling, convert predictions back:

\`\`\`python
# If used log transformation
original_scale = np.expm1(predictions)  # exp(x) - 1

# If used sqrt
original_scale = predictions ** 2

# If used Box-Cox/Yeo-Johnson
from scipy.special import inv_boxcox
original_scale = inv_boxcox(predictions, lambda_param)
\`\`\`

**Real-World Impact:**

\`\`\`
Income Prediction Model:

BEFORE transformation:
- Skewness: 3.2 (highly skewed)
- RMSE: $150,000 (poor)
- Model predicts poorly for high earners

AFTER log transformation:
- Skewness: 0.3 (near normal)
- RMSE: $45,000 (much better)
- Model predicts well across income ranges
\`\`\`

**Alternative: Robust Methods**

Instead of transforming, use robust methods:
- Median instead of mean
- Quantile regression
- Robust scaling
- Tree-based models (don't need transformation)`
  };

  return <ProblemTemplate data={data} problemNumber={15} />;
};

export default SkewedNumericValues;