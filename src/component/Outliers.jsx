import React from 'react';
import ProblemTemplate from './ProblemTemplate';

const Outliers = () => {
  const data = {
    title: 'Outliers',
    description: 'Outliers are data points that significantly differ from other observations. They can result from measurement errors, data entry mistakes, or represent genuine extreme values. Outliers can heavily influence statistical analyses and machine learning models.',
    originalData: [
      { transaction_id: 'T001', customer_age: 28, purchase_amount: 150, items_bought: 3, time_on_site: 12 },
      { transaction_id: 'T002', customer_age: 35, purchase_amount: 220, items_bought: 5, time_on_site: 18 },
      { transaction_id: 'T003', customer_age: 42, purchase_amount: 180, items_bought: 4, time_on_site: 15 },
      { transaction_id: 'T004', customer_age: 150, purchase_amount: 200, items_bought: 4, time_on_site: 14 },
      { transaction_id: 'T005', customer_age: 31, purchase_amount: 9500, items_bought: 3, time_on_site: 16 },
      { transaction_id: 'T006', customer_age: 29, purchase_amount: 175, items_bought: 45, time_on_site: 13 },
      { transaction_id: 'T007', customer_age: 38, purchase_amount: 195, items_bought: 4, time_on_site: 250 },
      { transaction_id: 'T008', customer_age: 33, purchase_amount: 210, items_bought: 5, time_on_site: 17 },
      { transaction_id: 'T009', customer_age: 27, purchase_amount: 165, items_bought: 3, time_on_site: 11 },
      { transaction_id: 'T010', customer_age: 36, purchase_amount: 190, items_bought: 4, time_on_site: 19 },
    ],
    cleanedData: [
      { transaction_id: 'T001', customer_age: 28, purchase_amount: 150, items_bought: 3, time_on_site: 12 },
      { transaction_id: 'T002', customer_age: 35, purchase_amount: 220, items_bought: 5, time_on_site: 18 },
      { transaction_id: 'T003', customer_age: 42, purchase_amount: 180, items_bought: 4, time_on_site: 15 },
      { transaction_id: 'T008', customer_age: 33, purchase_amount: 210, items_bought: 5, time_on_site: 17 },
      { transaction_id: 'T009', customer_age: 27, purchase_amount: 165, items_bought: 3, time_on_site: 11 },
      { transaction_id: 'T010', customer_age: 36, purchase_amount: 190, items_bought: 4, time_on_site: 19 },
    ],
    removedColumns: [],
    testDataset: `import pandas as pd
import numpy as np

# Create test dataset with outliers
np.random.seed(42)

data = {
    'transaction_id': ['T001', 'T002', 'T003', 'T004', 'T005', 'T006', 'T007', 'T008', 'T009', 'T010'],
    'customer_age': [28, 35, 42, 150, 31, 29, 38, 33, 27, 36],  # 150 is outlier
    'purchase_amount': [150, 220, 180, 200, 9500, 175, 195, 210, 165, 190],  # 9500 is outlier
    'items_bought': [3, 5, 4, 4, 3, 45, 4, 5, 3, 4],  # 45 is outlier
    'time_on_site': [12, 18, 15, 14, 16, 13, 250, 17, 11, 19]  # 250 is outlier
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print(f"\\nDataset Shape: {df.shape}")

# Show basic statistics
print("\\nBasic Statistics:")
print(df.describe())

# Visualize outliers with simple stats
print("\\nPotential Outliers (values > 3 std deviations):")
for col in df.select_dtypes(include=[np.number]).columns:
    mean = df[col].mean()
    std = df[col].std()
    outliers = df[abs(df[col] - mean) > 3 * std]
    if len(outliers) > 0:
        print(f"\\n{col}:")
        print(f"  Mean: {mean:.2f}, Std: {std:.2f}")
        print(f"  Outlier values: {outliers[col].tolist()}")`,
    solution: `# Solution: Detect and handle outliers using multiple methods

def detect_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Detect outliers using Interquartile Range (IQR) method
    
    Parameters:
    df: pandas DataFrame
    columns: list of numeric columns to check (None = all numeric)
    threshold: IQR multiplier (default: 1.5, use 3.0 for extreme outliers only)
    
    Returns:
    outlier_info: Dictionary with outlier details
    outlier_mask: Boolean mask for outlier rows
    """
    import pandas as pd
    import numpy as np
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_info = {}
    outlier_indices = set()
    
    print("="*70)
    print("OUTLIER DETECTION - IQR METHOD")
    print("="*70)
    print(f"IQR Threshold: {threshold}x")
    print(f"Formula: Outliers are < Q1 - {threshold}×IQR OR > Q3 + {threshold}×IQR")
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Find outliers
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outliers = df[outlier_mask]
        
        if len(outliers) > 0:
            outlier_info[col] = {
                'count': len(outliers),
                'indices': outliers.index.tolist(),
                'values': outliers[col].tolist(),
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            outlier_indices.update(outliers.index.tolist())
            
            print(f"\\n{col}:")
            print(f"  Q1 (25th percentile): {Q1:.2f}")
            print(f"  Q3 (75th percentile): {Q3:.2f}")
            print(f"  IQR: {IQR:.2f}")
            print(f"  Valid range: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"  Outliers found: {len(outliers)}")
            print(f"  Outlier indices: {outliers.index.tolist()}")
            print(f"  Outlier values: {outliers[col].tolist()}")
    
    outlier_mask = df.index.isin(outlier_indices)
    
    return outlier_info, outlier_mask


def detect_outliers_zscore(df, columns=None, threshold=3):
    """
    Detect outliers using Z-score method
    
    Parameters:
    df: pandas DataFrame
    columns: list of numeric columns to check
    threshold: Z-score threshold (default: 3, values > |3| are outliers)
    
    Returns:
    outlier_info: Dictionary with outlier details
    outlier_mask: Boolean mask for outlier rows
    """
    import numpy as np
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_info = {}
    outlier_indices = set()
    
    print("\\n" + "="*70)
    print("OUTLIER DETECTION - Z-SCORE METHOD")
    print("="*70)
    print(f"Z-score Threshold: ±{threshold}")
    print(f"Formula: Z = (X - mean) / std")
    
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        
        if std == 0:
            print(f"\\n{col}: No variation (std=0), skipping")
            continue
        
        z_scores = abs((df[col] - mean) / std)
        outlier_mask = z_scores > threshold
        outliers = df[outlier_mask]
        
        if len(outliers) > 0:
            outlier_info[col] = {
                'count': len(outliers),
                'indices': outliers.index.tolist(),
                'values': outliers[col].tolist(),
                'z_scores': z_scores[outlier_mask].tolist(),
                'mean': mean,
                'std': std
            }
            outlier_indices.update(outliers.index.tolist())
            
            print(f"\\n{col}:")
            print(f"  Mean: {mean:.2f}, Std: {std:.2f}")
            print(f"  Outliers found: {len(outliers)}")
            print(f"  Outlier indices: {outliers.index.tolist()}")
            print(f"  Outlier values: {outliers[col].tolist()}")
            print(f"  Z-scores: {[f'{z:.2f}' for z in z_scores[outlier_mask].tolist()]}")
    
    outlier_mask = df.index.isin(outlier_indices)
    
    return outlier_info, outlier_mask


# Apply IQR method
print("\\n" + "#"*70)
print("# METHOD 1: IQR (Interquartile Range)")
print("#"*70)
iqr_info, iqr_mask = detect_outliers_iqr(df, threshold=1.5)

# Apply Z-score method
print("\\n" + "#"*70)
print("# METHOD 2: Z-SCORE")
print("#"*70)
zscore_info, zscore_mask = detect_outliers_zscore(df, threshold=3)

# Combine both methods (rows flagged by either method)
combined_mask = iqr_mask | zscore_mask

print("\\n" + "="*70)
print("COMPARISON & DECISION")
print("="*70)
print(f"Rows flagged by IQR method: {iqr_mask.sum()}")
print(f"Rows flagged by Z-score method: {zscore_mask.sum()}")
print(f"Rows flagged by either method: {combined_mask.sum()}")
print(f"Rows flagged by both methods: {(iqr_mask & zscore_mask).sum()}")

# Remove outlier rows
cleaned_df = df[~combined_mask].copy()

print("\\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Original dataset: {len(df)} rows")
print(f"Outlier rows removed: {combined_mask.sum()}")
print(f"Cleaned dataset: {len(cleaned_df)} rows")
print(f"Data retention: {len(cleaned_df)/len(df)*100:.1f}%")

print("\\nRemoved rows:")
print(df[combined_mask])

print("\\n\\nCleaned Dataset:")
print(cleaned_df)

# Alternative: Cap outliers instead of removing
print("\\n" + "="*70)
print("ALTERNATIVE: CAP OUTLIERS (WINSORIZATION)")
print("="*70)
df_capped = df.copy()
for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap values at bounds
    df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
    
print("Outliers capped to valid range (preserves all rows):")
print(df_capped)`,
    explanation: `**What are Outliers?**

Outliers are data points that significantly deviate from the overall pattern. They can be:

1. **Errors**: Data entry mistakes, measurement errors, system glitches
2. **Legitimate**: Rare but genuine extreme values (e.g., billionaire in income data)

**Why Outliers Matter:**

1. **Skew Statistics**: Mean, variance heavily affected by outliers
2. **Model Performance**: Many ML algorithms sensitive to outliers (Linear Regression, KNN, Neural Networks)
3. **Misleading Insights**: Can lead to wrong conclusions
4. **Visual Distortion**: Make plots unreadable by forcing extreme scales

**Detection Methods:**

**1. IQR (Interquartile Range) Method** ⭐ MOST COMMON

\`\`\`
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR

Outliers: values < Lower Bound OR values > Upper Bound
\`\`\`

**Advantages:**
- Not affected by outliers themselves
- Works well with skewed distributions
- Standard in data science

**Threshold Guidelines:**
- 1.5 × IQR: Standard (flags mild outliers)
- 3.0 × IQR: Conservative (only extreme outliers)

**2. Z-Score Method**

\`\`\`
Z-score = (X - mean) / std

Outliers: |Z-score| > threshold (typically 3)
\`\`\`

**Advantages:**
- Simple and interpretable
- Good for normal distributions

**Disadvantages:**
- Assumes normal distribution
- Affected by outliers (mean/std)

**3. Other Methods:**

- **Modified Z-Score**: Uses median instead of mean (more robust)
- **Isolation Forest**: ML-based approach
- **DBSCAN**: Density-based clustering
- **Domain-Specific Rules**: Age > 120, negative prices, etc.

**Handling Outliers:**

**Option 1: Remove** (What we did)
\`\`\`python
df_cleaned = df[~outlier_mask]
\`\`\`
✓ Clean data for analysis
✗ Lose information, reduce sample size

**Option 2: Cap/Winsorize**
\`\`\`python
df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
\`\`\`
✓ Preserve row count
✓ Reduce outlier impact
✗ Distort original values

**Option 3: Transform**
\`\`\`python
df[col] = np.log(df[col])  # Log transformation
df[col] = np.sqrt(df[col])  # Square root
\`\`\`
✓ Reduce skewness
✗ Changes interpretation

**Option 4: Keep with Flag**
\`\`\`python
df['is_outlier'] = outlier_mask
\`\`\`
✓ Preserve all data
✓ Model can learn to handle
✗ Requires larger dataset

**Option 5: Impute**
\`\`\`python
df.loc[outlier_mask, col] = df[col].median()
\`\`\`
✓ Replace with reasonable value
✗ Artificial data point

**Decision Framework:**

**Remove outliers when:**
- They're clearly errors (age=150, negative prices)
- Small % of data (< 5%)
- Building predictive models sensitive to outliers

**Keep outliers when:**
- They're legitimate extreme values
- Represent important segments (VIP customers)
- Large % of data (> 10%)
- Building robust models (Random Forest, XGBoost handle outliers well)

**Cap outliers when:**
- Want to preserve sample size
- Outliers are errors but removing loses too much data
- Building regression models

**Best Practices:**

1. **Understand Context**: Is 9500 purchase legitimate or error?

2. **Visualize First**: Use boxplots, scatter plots
   \`\`\`python
   df.boxplot(column=['purchase_amount'])
   \`\`\`

3. **Use Multiple Methods**: Compare IQR and Z-score

4. **Document Decisions**: Log what was removed and why

5. **Domain Validation**: Check with subject matter experts

6. **Separate Analysis**: Analyze outliers separately—they might be interesting!

7. **Test Impact**: Compare model performance with/without outliers

**Outlier-Resistant Techniques:**

Instead of removing outliers, use robust methods:
- **Median** instead of mean
- **MAD** (Median Absolute Deviation) instead of std
- **Robust Scaler** in preprocessing
- **Huber Loss** in regression
- **Tree-based models** (Random Forest, XGBoost)

**Example Workflow:**

\`\`\`python
# 1. Detect
outliers = detect_outliers_iqr(df)

# 2. Investigate
print(df[outliers])

# 3. Visualize
df.boxplot()

# 4. Decide based on domain knowledge
# If errors → remove
# If legitimate → keep or cap

# 5. Document
log.info(f"Removed {sum(outliers)} outliers from {col}")
\`\`\``
  };

  return <ProblemTemplate data={data} problemNumber={8} />;
};

export default Outliers;