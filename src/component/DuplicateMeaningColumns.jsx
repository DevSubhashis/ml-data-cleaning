import React from 'react';
import ProblemTemplate from './ProblemTemplate';

const DuplicateMeaningColumns = () => {
  const data = {
    title: 'Duplicate Meaning Columns',
    description: 'Duplicate meaning columns represent the same information in different formats or with different names. They create redundancy, inflate dimensionality, cause multicollinearity in models, and waste computational resources. Common examples include converted units, derived columns, or renamed fields.',
    originalData: [
      { customer_id: 'C001', first_name: 'John', given_name: 'John', age_years: 35, age_months: 420, birth_year: 1989, year_of_birth: 1989, price_usd: 100, price_cents: 10000 },
      { customer_id: 'C002', first_name: 'Jane', given_name: 'Jane', age_years: 28, age_months: 336, birth_year: 1996, year_of_birth: 1996, price_usd: 150, price_cents: 15000 },
      { customer_id: 'C003', first_name: 'Bob', given_name: 'Bob', age_years: 42, age_months: 504, birth_year: 1982, year_of_birth: 1982, price_usd: 200, price_cents: 20000 },
      { customer_id: 'C004', first_name: 'Alice', given_name: 'Alice', age_years: 31, age_months: 372, birth_year: 1993, year_of_birth: 1993, price_usd: 120, price_cents: 12000 },
      { customer_id: 'C005', first_name: 'Charlie', given_name: 'Charlie', age_years: 45, age_months: 540, birth_year: 1979, year_of_birth: 1979, price_usd: 175, price_cents: 17500 },
    ],
    cleanedData: [
      { customer_id: 'C001', first_name: 'John', age_years: 35, birth_year: 1989, price_usd: 100 },
      { customer_id: 'C002', first_name: 'Jane', age_years: 28, birth_year: 1996, price_usd: 150 },
      { customer_id: 'C003', first_name: 'Bob', age_years: 42, birth_year: 1982, price_usd: 200 },
      { customer_id: 'C004', first_name: 'Alice', age_years: 31, birth_year: 1993, price_usd: 120 },
      { customer_id: 'C005', first_name: 'Charlie', age_years: 45, birth_year: 1979, price_usd: 175 },
    ],
    removedColumns: ['given_name', 'age_months', 'year_of_birth', 'price_cents'],
    testDataset: `import pandas as pd
import numpy as np

# Create test dataset with duplicate meaning columns
data = {
    'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
    'first_name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
    'given_name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],  # Duplicate of first_name
    'age_years': [35, 28, 42, 31, 45],
    'age_months': [420, 336, 504, 372, 540],  # Just age_years * 12
    'birth_year': [1989, 1996, 1982, 1993, 1979],
    'year_of_birth': [1989, 1996, 1982, 1993, 1979],  # Duplicate of birth_year
    'price_usd': [100, 150, 200, 120, 175],
    'price_cents': [10000, 15000, 20000, 12000, 17500]  # Just price_usd * 100
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print(f"\\nDataset Shape: {df.shape}")

print("\\n⚠ PROBLEM: Multiple columns represent the same information:")
print("  - 'first_name' and 'given_name' are identical")
print("  - 'age_years' and 'age_months' (months = years × 12)")
print("  - 'birth_year' and 'year_of_birth' are identical")
print("  - 'price_usd' and 'price_cents' (cents = usd × 100)")

# Calculate correlations to detect duplicates
print("\\nCorrelation Matrix (perfect correlation = 1.0):")
numeric_cols = ['age_years', 'age_months', 'birth_year', 'year_of_birth', 'price_usd', 'price_cents']
print(df[numeric_cols].corr())`,
    solution: `# Solution: Detect and remove duplicate meaning columns

def detect_duplicate_columns(df, correlation_threshold=0.99):
    """
    Detect columns that represent the same information
    
    Parameters:
    df: pandas DataFrame
    correlation_threshold: float, correlation above this suggests duplicates (default: 0.99)
    
    Returns:
    duplicate_report: Dictionary with duplicate column pairs
    """
    import pandas as pd
    import numpy as np
    
    duplicate_report = {
        'exact_duplicates': [],
        'highly_correlated': [],
        'derived_columns': []
    }
    
    print("="*70)
    print("DUPLICATE MEANING COLUMN DETECTION")
    print("="*70)
    
    # Method 1: Exact duplicates (identical values)
    print("\\n1. EXACT DUPLICATES (Identical values)")
    print("-"*70)
    
    columns = df.columns.tolist()
    checked_pairs = set()
    
    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:
            pair = tuple(sorted([col1, col2]))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)
            
            # Check if columns are identical
            if df[col1].equals(df[col2]):
                duplicate_report['exact_duplicates'].append({
                    'column1': col1,
                    'column2': col2,
                    'type': 'exact_match'
                })
                print(f"  ✗ '{col1}' and '{col2}' are identical")
    
    if not duplicate_report['exact_duplicates']:
        print("  ✓ No exact duplicates found")
    
    # Method 2: High correlation (for numeric columns)
    print("\\n2. HIGHLY CORRELATED COLUMNS (Likely derived/converted)")
    print("-"*70)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Find pairs with correlation above threshold
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr_value = corr_matrix.loc[col1, col2]
                
                if corr_value >= correlation_threshold:
                    # Check for linear relationship (derived column)
                    ratio = (df[col1] / df[col2]).dropna()
                    is_constant_ratio = ratio.std() < 0.001 if len(ratio) > 0 else False
                    
                    duplicate_report['highly_correlated'].append({
                        'column1': col1,
                        'column2': col2,
                        'correlation': corr_value,
                        'constant_ratio': is_constant_ratio,
                        'ratio': ratio.mean() if is_constant_ratio else None
                    })
                    
                    print(f"  ✗ '{col1}' and '{col2}':")
                    print(f"      Correlation: {corr_value:.4f}")
                    if is_constant_ratio:
                        print(f"      Constant ratio: {ratio.mean():.2f} (one is derived from other)")
    
    if not duplicate_report['highly_correlated']:
        print("  ✓ No highly correlated columns found")
    
    # Method 3: Name similarity
    print("\\n3. SIMILAR COLUMN NAMES (Possible duplicates)")
    print("-"*70)
    
    from difflib import SequenceMatcher
    
    name_threshold = 0.7  # 70% name similarity
    
    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:
            similarity = SequenceMatcher(None, col1.lower(), col2.lower()).ratio()
            
            if similarity >= name_threshold:
                # Check if values are also similar
                are_similar = False
                if df[col1].dtype == df[col2].dtype:
                    if df[col1].dtype == 'object':
                        # For text, check equality
                        are_similar = (df[col1] == df[col2]).sum() / len(df) > 0.8
                    else:
                        # For numeric, check correlation
                        try:
                            corr = df[[col1, col2]].corr().iloc[0, 1]
                            are_similar = abs(corr) > 0.8
                        except:
                            pass
                
                if are_similar:
                    duplicate_report['derived_columns'].append({
                        'column1': col1,
                        'column2': col2,
                        'name_similarity': similarity
                    })
                    print(f"  ⚠ '{col1}' and '{col2}' (name similarity: {similarity:.2%})")
    
    if not duplicate_report['derived_columns']:
        print("  ✓ No similar named duplicates found")
    
    return duplicate_report


def remove_duplicate_columns(df, duplicate_report, keep_preference='first'):
    """
    Remove duplicate meaning columns
    
    Parameters:
    df: pandas DataFrame
    duplicate_report: Output from detect_duplicate_columns
    keep_preference: 'first', 'last', or 'shorter_name'
    
    Returns:
    cleaned_df: DataFrame with duplicates removed
    removed_columns: List of removed column names
    """
    import pandas as pd
    
    df_cleaned = df.copy()
    removed_columns = []
    
    print("\\n" + "="*70)
    print("REMOVING DUPLICATE COLUMNS")
    print("="*70)
    print(f"Keep preference: {keep_preference}")
    
    # Remove exact duplicates
    for dup in duplicate_report['exact_duplicates']:
        col1, col2 = dup['column1'], dup['column2']
        
        if keep_preference == 'first':
            to_remove = col2
        elif keep_preference == 'last':
            to_remove = col1
        else:  # shorter_name
            to_remove = col1 if len(col1) > len(col2) else col2
        
        if to_remove in df_cleaned.columns:
            df_cleaned = df_cleaned.drop(columns=[to_remove])
            removed_columns.append(to_remove)
            print(f"  ✓ Removed '{to_remove}' (duplicate of '{col1 if to_remove == col2 else col2}')")
    
    # Remove highly correlated (derived) columns
    for dup in duplicate_report['highly_correlated']:
        col1, col2 = dup['column1'], dup['column2']
        
        # Prefer keeping simpler unit (e.g., 'years' over 'months', 'usd' over 'cents')
        # Otherwise use keep_preference
        if 'month' in col2.lower() and 'year' in col1.lower():
            to_remove = col2
        elif 'cent' in col2.lower() and 'usd' in col1.lower():
            to_remove = col2
        elif 'cent' in col2.lower() and 'dollar' in col1.lower():
            to_remove = col2
        elif keep_preference == 'shorter_name':
            to_remove = col1 if len(col1) > len(col2) else col2
        else:
            to_remove = col2 if keep_preference == 'first' else col1
        
        if to_remove in df_cleaned.columns:
            df_cleaned = df_cleaned.drop(columns=[to_remove])
            removed_columns.append(to_remove)
            if dup.get('constant_ratio'):
                print(f"  ✓ Removed '{to_remove}' (derived from '{col1 if to_remove == col2 else col2}' with ratio {dup['ratio']:.2f})")
            else:
                print(f"  ✓ Removed '{to_remove}' (highly correlated with '{col1 if to_remove == col2 else col2}')")
    
    # Handle similar named columns
    for dup in duplicate_report['derived_columns']:
        col1, col2 = dup['column1'], dup['column2']
        
        if keep_preference == 'shorter_name':
            to_remove = col1 if len(col1) > len(col2) else col2
        else:
            to_remove = col2 if keep_preference == 'first' else col1
        
        if to_remove in df_cleaned.columns:
            df_cleaned = df_cleaned.drop(columns=[to_remove])
            removed_columns.append(to_remove)
            print(f"  ✓ Removed '{to_remove}' (similar to '{col1 if to_remove == col2 else col2}')")
    
    return df_cleaned, removed_columns


# Detect duplicates
duplicate_report = detect_duplicate_columns(df, correlation_threshold=0.99)

# Remove duplicates
cleaned_df, removed = remove_duplicate_columns(df, duplicate_report, keep_preference='shorter_name')

print("\\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Original columns: {len(df.columns)}")
print(f"Removed columns: {len(removed)}")
print(f"Final columns: {len(cleaned_df.columns)}")
print(f"Dimensionality reduction: {len(removed)/len(df.columns)*100:.1f}%")

print(f"\\nRemoved columns: {removed}")
print(f"Retained columns: {cleaned_df.columns.tolist()}")

print("\\n\\nCleaned Dataset:")
print(cleaned_df)

# Show impact on memory
print("\\n" + "="*70)
print("MEMORY IMPACT")
print("="*70)
original_memory = df.memory_usage(deep=True).sum() / 1024
cleaned_memory = cleaned_df.memory_usage(deep=True).sum() / 1024
print(f"Original memory: {original_memory:.2f} KB")
print(f"Cleaned memory: {cleaned_memory:.2f} KB")
print(f"Memory saved: {original_memory - cleaned_memory:.2f} KB ({(original_memory - cleaned_memory)/original_memory*100:.1f}%)")`,
    explanation: `**What are Duplicate Meaning Columns?**

Columns that represent the same information in different ways:
- **Exact duplicates**: Identical values, different names
- **Derived columns**: One calculated from another (age_months = age_years × 12)
- **Unit conversions**: Same measurement in different units (USD vs cents)
- **Renamed fields**: Same data, different column names

**Common Examples:**

**1. Unit Conversions:**
\`\`\`
weight_kg, weight_lbs
distance_km, distance_miles
price_usd, price_cents
temperature_c, temperature_f
\`\`\`

**2. Time Representations:**
\`\`\`
age_years, age_months, age_days
duration_seconds, duration_minutes
date, timestamp, epoch_time
\`\`\`

**3. Naming Variations:**
\`\`\`
first_name, given_name, fname
customer_id, cust_id, customerID
email_address, email, email_addr
\`\`\`

**4. Derived Calculations:**
\`\`\`
total, subtotal + tax
profit, revenue - cost
bmi, weight / height²
\`\`\`

**5. Database Joins:**
\`\`\`
user_id (from table A), user_id (from table B)
Same column brought in multiple times
\`\`\`

**Why They're Problematic:**

1. **Multicollinearity**: Perfect correlation breaks regression assumptions
   - Unstable coefficients
   - Inflated standard errors
   - Unreliable p-values

2. **Redundancy**: Same information counted multiple times
   - Wastes memory
   - Slows computation
   - Confuses feature importance

3. **Overfitting**: Model learns same pattern multiple times
   - Overweights certain information
   - Poor generalization

4. **Dimensionality**: Unnecessarily high feature count
   - Curse of dimensionality
   - Longer training times
   - More complex models

5. **Interpretation Issues**: Which column is "real" feature importance?

6. **Maintenance**: Updates needed in multiple places

**Detection Methods:**

**1. Exact Match:**
\`\`\`python
# Check if columns are identical
df['col1'].equals(df['col2'])

# Find all exact duplicates
duplicates = []
for i, col1 in enumerate(df.columns):
    for col2 in df.columns[i+1:]:
        if df[col1].equals(df[col2]):
            duplicates.append((col1, col2))
\`\`\`

**2. Correlation Analysis:**
\`\`\`python
# Calculate correlation matrix
corr_matrix = df.corr().abs()

# Find high correlations (> 0.99)
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.99:
            high_corr.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))
\`\`\`

**3. Constant Ratio Check:**
\`\`\`python
# Check if col2 = col1 × constant
ratio = df['col2'] / df['col1']
is_constant = ratio.std() < 0.001

if is_constant:
    print(f"col2 = col1 × {ratio.mean():.2f}")
    # col2 is derived from col1
\`\`\`

**4. Name Similarity:**
\`\`\`python
from difflib import SequenceMatcher

def name_similarity(name1, name2):
    return SequenceMatcher(None, name1, name2).ratio()

# Check all column pairs
for col1 in df.columns:
    for col2 in df.columns:
        if col1 != col2:
            sim = name_similarity(col1, col2)
            if sim > 0.7:  # 70% similar
                print(f"{col1} ≈ {col2}: {sim:.2%}")
\`\`\`

**5. Hash Comparison:**
\`\`\`python
import hashlib

def column_hash(series):
    return hashlib.md5(series.to_string().encode()).hexdigest()

hashes = {col: column_hash(df[col]) for col in df.columns}
\`\`\`

**Removal Strategies:**

**Strategy 1: Keep Shorter Name**
\`\`\`python
# Keep 'age' over 'age_years'
# Keep 'id' over 'customer_id'
to_keep = col1 if len(col1) < len(col2) else col2
\`\`\`

**Strategy 2: Keep Standard Unit**
\`\`\`python
# Keep USD over cents
# Keep years over months
# Keep kg over lbs (if metric standard)

preferences = {
    'usd': ['cents', 'dollars'],
    'years': ['months', 'days'],
    'kg': ['lbs', 'pounds']
}
\`\`\`

**Strategy 3: Keep First Occurrence**
\`\`\`python
# In order of columns, keep first seen
columns_to_keep = []
seen_values = set()

for col in df.columns:
    col_hash = column_hash(df[col])
    if col_hash not in seen_values:
        columns_to_keep.append(col)
        seen_values.add(col_hash)
\`\`\`

**Strategy 4: Domain Knowledge**
\`\`\`python
# Keep columns that make sense for your domain
# E.g., in finance, keep 'price_usd' not 'price_cents'
# In medicine, keep 'age_years' not 'age_days'
\`\`\`

**Best Practices:**

1. **Check Before Feature Engineering:**
   Detect duplicates before creating new features

2. **Document Removal:**
   \`\`\`python
   removed_log = {
       'removed_columns': ['age_months', 'price_cents'],
       'kept_columns': ['age_years', 'price_usd'],
       'reason': 'derived columns with constant ratio'
   }
   \`\`\`

3. **Visual Inspection:**
   \`\`\`python
   import seaborn as sns
   import matplotlib.pyplot as plt
   
   # Heatmap of correlations
   sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
   plt.show()
   \`\`\`

4. **Automated Checks in Pipeline:**
   \`\`\`python
   def validate_no_duplicates(df, threshold=0.99):
       corr = df.corr().abs()
       duplicates = (corr > threshold) & (corr < 1.0)
       if duplicates.any().any():
           raise ValueError("Duplicate meaning columns detected!")
   \`\`\`

5. **Keep One, Drop Others:**
   Never keep both correlated columns in model

**Handling Edge Cases:**

**Case 1: Partially Duplicated**
\`\`\`python
# 95% identical values
# Decision: Investigate why 5% differ
# May indicate data quality issue
\`\`\`

**Case 2: Different Scales, Same Information**
\`\`\`python
# price_normalized (0-1) vs price (dollars)
# Keep original, drop derived
\`\`\`

**Case 3: Legitimate Similar Columns**
\`\`\`python
# 'order_date' vs 'ship_date' - both dates but different meaning
# Keep both!
# Check: are they highly correlated due to business process?
\`\`\`

**Verification After Removal:**

\`\`\`python
# Check no information lost
def verify_removal(original_df, cleaned_df, removed_cols):
    for col in removed_cols:
        # Can we reconstruct from remaining columns?
        # E.g., price_cents = price_usd * 100
        pass
    
    # Check model performance not affected
    # Train model with both versions, compare scores
\`\`\`

**Multicollinearity Detection (VIF):**

\`\`\`python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) 
                       for i in range(len(df.columns))]
    return vif_data

# VIF > 10 indicates high multicollinearity
vif = calculate_vif(df[numeric_columns])
print(vif[vif['VIF'] > 10])
\`\`\`

**Real-World Impact:**

\`\`\`
E-commerce Dataset:
- Original: 50 columns
- Found duplicates:
  * price_usd, price_cents (ratio: 100)
  * weight_kg, weight_lbs (ratio: 2.205)
  * timestamp, date, epoch_time (identical)
  * customer_id, cust_id (identical)
  
After removal:
- 42 columns (16% reduction)
- Memory: 1.2GB → 1.0GB (17% saved)
- Training time: 45min → 38min (16% faster)
- Model accuracy: Same (no information lost)
\`\`\`

**Prevention:**

1. **Data Dictionary**: Maintain clear column definitions
2. **Naming Conventions**: Consistent naming prevents confusion
3. **ETL Validation**: Check for duplicates during import
4. **Code Review**: Review feature engineering for redundancy`
  };

  return <ProblemTemplate data={data} problemNumber={17} />;
};

export default DuplicateMeaningColumns;