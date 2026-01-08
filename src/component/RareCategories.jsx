import React from 'react';
import ProblemTemplate from './ProblemTemplate';

const RareCategories = () => {
  const data = {
    title: 'Rare Categories',
    description: 'Rare categories are values that appear very infrequently in categorical columns. They can cause overfitting in models, create too many dummy variables in encoding, and provide unreliable statistics. These low-frequency categories should be identified and either grouped together or removed.',
    originalData: [
      { customer_id: 'C001', country: 'USA', product_category: 'Electronics', payment_method: 'Credit Card', subscription: 'Premium' },
      { customer_id: 'C002', country: 'USA', product_category: 'Electronics', payment_method: 'Credit Card', subscription: 'Basic' },
      { customer_id: 'C003', country: 'UK', product_category: 'Clothing', payment_method: 'PayPal', subscription: 'Premium' },
      { customer_id: 'C004', country: 'USA', product_category: 'Electronics', payment_method: 'Credit Card', subscription: 'Basic' },
      { customer_id: 'C005', country: 'Canada', product_category: 'Books', payment_method: 'Debit Card', subscription: 'Premium' },
      { customer_id: 'C006', country: 'USA', product_category: 'Electronics', payment_method: 'Credit Card', subscription: 'Basic' },
      { customer_id: 'C007', country: 'France', product_category: 'Home Decor', payment_method: 'Bank Transfer', subscription: 'Enterprise' },
      { customer_id: 'C008', country: 'USA', product_category: 'Electronics', payment_method: 'Credit Card', subscription: 'Premium' },
      { customer_id: 'C009', country: 'Germany', product_category: 'Sports', payment_method: 'Cryptocurrency', subscription: 'VIP' },
      { customer_id: 'C010', country: 'USA', product_category: 'Electronics', payment_method: 'Credit Card', subscription: 'Basic' },
      { customer_id: 'C011', country: 'UK', product_category: 'Clothing', payment_method: 'PayPal', subscription: 'Premium' },
      { customer_id: 'C012', country: 'USA', product_category: 'Electronics', payment_method: 'Credit Card', subscription: 'Basic' },
      { customer_id: 'C013', country: 'Japan', product_category: 'Gaming', payment_method: 'Gift Card', subscription: 'Lifetime' },
      { customer_id: 'C014', country: 'USA', product_category: 'Electronics', payment_method: 'Credit Card', subscription: 'Premium' },
      { customer_id: 'C015', country: 'USA', product_category: 'Electronics', payment_method: 'Credit Card', subscription: 'Basic' },
    ],
    cleanedData: [
      { customer_id: 'C001', country: 'USA', product_category: 'Electronics', payment_method: 'Credit Card', subscription: 'Premium' },
      { customer_id: 'C002', country: 'USA', product_category: 'Electronics', payment_method: 'Credit Card', subscription: 'Basic' },
      { customer_id: 'C003', country: 'UK', product_category: 'Clothing', payment_method: 'PayPal', subscription: 'Premium' },
      { customer_id: 'C004', country: 'USA', product_category: 'Electronics', payment_method: 'Credit Card', subscription: 'Basic' },
      { customer_id: 'C005', country: 'Other', product_category: 'Other', payment_method: 'Other', subscription: 'Premium' },
      { customer_id: 'C006', country: 'USA', product_category: 'Electronics', payment_method: 'Credit Card', subscription: 'Basic' },
      { customer_id: 'C007', country: 'Other', product_category: 'Other', payment_method: 'Other', subscription: 'Other' },
      { customer_id: 'C008', country: 'USA', product_category: 'Electronics', payment_method: 'Credit Card', subscription: 'Premium' },
      { customer_id: 'C009', country: 'Other', product_category: 'Other', payment_method: 'Other', subscription: 'Other' },
      { customer_id: 'C010', country: 'USA', product_category: 'Electronics', payment_method: 'Credit Card', subscription: 'Basic' },
      { customer_id: 'C011', country: 'UK', product_category: 'Clothing', payment_method: 'PayPal', subscription: 'Premium' },
      { customer_id: 'C012', country: 'USA', product_category: 'Electronics', payment_method: 'Credit Card', subscription: 'Basic' },
      { customer_id: 'C013', country: 'Other', product_category: 'Other', payment_method: 'Other', subscription: 'Other' },
      { customer_id: 'C014', country: 'USA', product_category: 'Electronics', payment_method: 'Credit Card', subscription: 'Premium' },
      { customer_id: 'C015', country: 'USA', product_category: 'Electronics', payment_method: 'Credit Card', subscription: 'Basic' },
    ],
    removedColumns: [],
    testDataset: `import pandas as pd
import numpy as np

# Create test dataset with rare categories
data = {
    'customer_id': [f'C{str(i).zfill(3)}' for i in range(1, 16)],
    'country': ['USA']*9 + ['UK']*2 + ['Canada', 'France', 'Germany', 'Japan'],
    'product_category': ['Electronics']*9 + ['Clothing']*2 + ['Books', 'Home Decor', 'Sports', 'Gaming'],
    'payment_method': ['Credit Card']*9 + ['PayPal']*2 + ['Debit Card', 'Bank Transfer', 'Cryptocurrency', 'Gift Card'],
    'subscription': ['Basic']*5 + ['Premium']*5 + ['Enterprise', 'VIP', 'Lifetime'] + ['Premium', 'Basic']
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print(f"\\nDataset Shape: {df.shape}")

# Show frequency distributions
print("\\n" + "="*70)
print("FREQUENCY DISTRIBUTIONS")
print("="*70)

for col in ['country', 'product_category', 'payment_method', 'subscription']:
    print(f"\\n{col}:")
    print(df[col].value_counts())
    print(f"Unique values: {df[col].nunique()}")`,
    solution: `# Solution: Handle rare categories by grouping or removing

def handle_rare_categories(df, columns=None, threshold=0.05, strategy='group', 
                           group_name='Other'):
    """
    Identify and handle rare categories in categorical columns
    
    Parameters:
    df: pandas DataFrame
    columns: list of categorical columns to check (None = all object columns)
    threshold: float, minimum frequency threshold (default: 5%)
             Categories with frequency < threshold are considered rare
    strategy: str, how to handle rare categories
             - 'group': Group rare categories into a single category
             - 'remove': Remove rows with rare categories
             - 'flag': Keep but add a flag column
    group_name: str, name for grouped rare categories (default: 'Other')
    
    Returns:
    cleaned_df: DataFrame with rare categories handled
    rare_report: Dictionary with details about rare categories
    """
    import pandas as pd
    
    df_cleaned = df.copy()
    rare_report = {}
    
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    
    print("="*70)
    print("RARE CATEGORY DETECTION AND HANDLING")
    print("="*70)
    print(f"Threshold: {threshold*100}% (categories below this are considered rare)")
    print(f"Strategy: {strategy}")
    
    total_rows = len(df)
    
    for col in columns:
        if col not in df.columns:
            print(f"\\n⚠ Column '{col}' not found, skipping")
            continue
        
        print(f"\\n{col}:")
        
        # Calculate frequency for each category
        value_counts = df[col].value_counts()
        frequencies = value_counts / total_rows
        
        # Identify rare categories
        rare_mask = frequencies < threshold
        rare_categories = frequencies[rare_mask].index.tolist()
        common_categories = frequencies[~rare_mask].index.tolist()
        
        if len(rare_categories) > 0:
            print(f"  Total categories: {df[col].nunique()}")
            print(f"  Common categories: {len(common_categories)}")
            print(f"  Rare categories: {len(rare_categories)}")
            
            print(f"\\n  Rare categories found:")
            for cat in rare_categories:
                count = value_counts[cat]
                freq = frequencies[cat]
                print(f"    '{cat}': {count} occurrences ({freq*100:.1f}%)")
            
            # Handle based on strategy
            if strategy == 'group':
                # Replace rare categories with group_name
                rare_mask_rows = df[col].isin(rare_categories)
                df_cleaned.loc[rare_mask_rows, col] = group_name
                print(f"\\n  ✓ Grouped {rare_mask_rows.sum()} rows into '{group_name}'")
                
            elif strategy == 'remove':
                # Remove rows with rare categories
                rare_mask_rows = df[col].isin(rare_categories)
                df_cleaned = df_cleaned[~rare_mask_rows]
                print(f"\\n  ✓ Removed {rare_mask_rows.sum()} rows with rare categories")
                
            elif strategy == 'flag':
                # Add flag column
                flag_col = f"{col}_is_rare"
                df_cleaned[flag_col] = df_cleaned[col].isin(rare_categories)
                print(f"\\n  ✓ Added flag column '{flag_col}'")
            
            rare_report[col] = {
                'total_categories': df[col].nunique(),
                'common_categories': common_categories,
                'rare_categories': rare_categories,
                'rare_category_counts': {cat: int(value_counts[cat]) for cat in rare_categories},
                'rows_affected': int((df[col].isin(rare_categories)).sum())
            }
        else:
            print(f"  ✓ No rare categories found")
    
    return df_cleaned, rare_report


# Apply with grouping strategy (most common)
print("\\n" + "#"*70)
print("# STRATEGY 1: GROUP RARE CATEGORIES")
print("#"*70)

cleaned_df_group, report_group = handle_rare_categories(
    df, 
    columns=['country', 'product_category', 'payment_method', 'subscription'],
    threshold=0.10,  # 10% threshold
    strategy='group',
    group_name='Other'
)

print("\\n" + "="*70)
print("RARE CATEGORY REPORT")
print("="*70)
for col, info in report_group.items():
    print(f"\\n{col}:")
    print(f"  Original categories: {info['total_categories']}")
    print(f"  After grouping: {cleaned_df_group[col].nunique()}")
    print(f"  Reduction: {info['total_categories'] - cleaned_df_group[col].nunique()} categories")
    print(f"  Rows affected: {info['rows_affected']}")

print("\\n" + "="*70)
print("BEFORE vs AFTER VALUE COUNTS")
print("="*70)

for col in ['country', 'product_category', 'payment_method', 'subscription']:
    if col in report_group:
        print(f"\\n{col}:")
        print("BEFORE:")
        print(df[col].value_counts())
        print("\\nAFTER:")
        print(cleaned_df_group[col].value_counts())

print("\\n\\nCleaned Dataset (Grouped Strategy):")
print(cleaned_df_group)

# Demonstrate alternative strategy: Remove
print("\\n" + "#"*70)
print("# STRATEGY 2: REMOVE RARE CATEGORIES")
print("#"*70)

cleaned_df_remove, report_remove = handle_rare_categories(
    df,
    columns=['country', 'product_category', 'payment_method', 'subscription'],
    threshold=0.10,
    strategy='remove'
)

print("\\nDataset size after removal:")
print(f"  Original: {len(df)} rows")
print(f"  After removal: {len(cleaned_df_remove)} rows")
print(f"  Rows removed: {len(df) - len(cleaned_df_remove)}")
print(f"  Data retention: {len(cleaned_df_remove)/len(df)*100:.1f}%")`,
    explanation: `**What are Rare Categories?**

Rare categories are values in categorical columns that appear infrequently in the dataset. They have low frequency/support and can cause problems in analysis and modeling.

**Examples:**
- Country: 99% USA, 1% distributed across 50 other countries
- Product: 95% common items, 5% one-off special products
- Payment: 90% credit card, 10% split across 15 rare methods

**Why Rare Categories are Problematic:**

1. **Overfitting Risk**: Models memorize rare patterns instead of learning general ones
2. **Unreliable Statistics**: Can't trust mean/patterns from 1-2 samples
3. **High Dimensionality**: One-hot encoding creates too many sparse columns
4. **Poor Generalization**: Rare categories unlikely to appear in new data
5. **Computational Cost**: Extra features slow training/inference
6. **Test/Train Split Issues**: Rare categories may only appear in one split

**When are Categories "Rare"?**

**Frequency-Based Thresholds:**
- **< 1%**: Almost always rare
- **< 5%**: Usually rare, handle with care
- **< 10%**: Consider rare in smaller datasets
- **< 50 samples**: Absolute minimum for statistical reliability

**Context Matters:**
- **1% in 1M rows** = 10,000 samples (maybe keep)
- **1% in 100 rows** = 1 sample (definitely rare)

**Detection Methods:**

\`\`\`python
# Method 1: Frequency distribution
df['category'].value_counts()
df['category'].value_counts(normalize=True)  # As percentages

# Method 2: Frequency threshold
threshold = 0.05  # 5%
freq = df['category'].value_counts(normalize=True)
rare = freq[freq < threshold].index.tolist()

# Method 3: Absolute count threshold
min_count = 50
counts = df['category'].value_counts()
rare = counts[counts < min_count].index.tolist()

# Method 4: Cumulative frequency
# Keep categories that make up 95% of data
freq_sorted = df['category'].value_counts(normalize=True).sort_values(ascending=False)
cumsum = freq_sorted.cumsum()
keep = cumsum[cumsum <= 0.95].index.tolist()
\`\`\`

**Handling Strategies:**

**Strategy 1: Group into "Other"** ⭐ MOST COMMON

\`\`\`python
# Replace rare categories with 'Other'
rare_categories = ['Category_A', 'Category_B', ...]
df['category'] = df['category'].replace(rare_categories, 'Other')
\`\`\`

**Pros:**
✓ Preserves row count
✓ Reduces dimensionality
✓ Clear semantics ("Other" = infrequent)

**Cons:**
✗ Loses granular information
✗ "Other" becomes catch-all

**Strategy 2: Remove Rows**

\`\`\`python
# Remove rows with rare categories
df = df[~df['category'].isin(rare_categories)]
\`\`\`

**Pros:**
✓ Clean dataset with only common categories
✓ No "Other" ambiguity

**Cons:**
✗ Loses data
✗ Can reduce sample size significantly

**Strategy 3: Keep with Flag**

\`\`\`python
# Add is_rare flag column
df['category_is_rare'] = df['category'].isin(rare_categories)
\`\`\`

**Pros:**
✓ Preserves all information
✓ Model can learn rare category behavior

**Cons:**
✗ Still have high dimensionality
✗ Doesn't solve overfitting

**Strategy 4: Hierarchical Grouping**

\`\`\`python
# Group by meaningful hierarchy
category_mapping = {
    'iPhone': 'Electronics',
    'Galaxy': 'Electronics',
    'Rare_Phone_Model': 'Electronics'
}
df['category_grouped'] = df['category'].map(category_mapping)
\`\`\`

**Pros:**
✓ Semantically meaningful
✓ Preserves domain structure

**Cons:**
✗ Requires domain knowledge
✗ Manual effort

**Best Practices:**

1. **Set Context-Appropriate Thresholds:**
   \`\`\`python
   # For large datasets
   threshold = 0.01  # 1%
   
   # For small datasets
   threshold = 0.05  # 5%
   
   # Or use absolute counts
   min_samples = 50
   \`\`\`

2. **Analyze Before/After:**
   \`\`\`python
   print("Before:", df['category'].nunique())
   # ... handle rare categories ...
   print("After:", df['category'].nunique())
   \`\`\`

3. **Document Groupings:**
   Keep track of what went into "Other"
   \`\`\`python
   rare_categories_log = {
       'country': ['Monaco', 'Liechtenstein', ...],
       'payment': ['Bitcoin', 'Check', ...]
   }
   \`\`\`

4. **Consider Domain Importance:**
   Some rare categories matter:
   - Rare disease in medical data
   - Fraud cases in transaction data
   - VIP customers in business data

5. **Test Both Strategies:**
   Compare model performance with grouping vs removal

6. **Use in Combination:**
   Group moderately rare (1-5%), remove extremely rare (< 1%)

**Decision Framework:**

**Group when:**
- Rare categories < 10% of data
- Need to preserve row count
- Categories are semantically similar
- Building classification models

**Remove when:**
- Rare categories < 1% of data
- Have plenty of data overall
- Categories are truly outliers
- Doing statistical analysis

**Keep when:**
- Rare categories are important (fraud, disease)
- Large dataset (rare = thousands of samples)
- Doing exploratory analysis
- Need complete records

**Impact on Machine Learning:**

**One-Hot Encoding Problem:**
\`\`\`
Before handling:
50 categories → 50 binary columns (49 sparse)

After grouping rare:
5 common + 1 "Other" → 6 binary columns
88% dimensionality reduction!
\`\`\`

**Model Performance:**
- **Before**: Overfits to rare patterns, poor generalization
- **After**: Learns from common patterns, better generalization

**Advanced Techniques:**

**1. Frequency Encoding:**
\`\`\`python
# Replace category with its frequency
freq_map = df['category'].value_counts(normalize=True).to_dict()
df['category_freq'] = df['category'].map(freq_map)
# No need to handle rare categories separately
\`\`\`

**2. Target Encoding:**
\`\`\`python
# Replace category with mean of target variable
target_map = df.groupby('category')['target'].mean().to_dict()
df['category_encoded'] = df['category'].map(target_map)
\`\`\`

**3. Smooth Encoding:**
\`\`\`python
# Add smoothing for rare categories
def smooth_encode(cat, overall_mean, smoothing=10):
    cat_mean = df[df['category']==cat]['target'].mean()
    cat_count = (df['category']==cat).sum()
    return (cat_mean * cat_count + overall_mean * smoothing) / (cat_count + smoothing)
\`\`\`

**Real-World Example:**

\`\`\`
E-commerce Dataset:
- 1M transactions
- 5,000 product categories
- Top 100 categories = 90% of sales
- Bottom 4,900 categories = 10% of sales

Solution:
1. Keep top 100 (90% coverage)
2. Group next 400 into "Other_Frequent" (8%)
3. Group bottom 4,500 into "Other_Rare" (2%)

Result:
- 5,000 → 102 categories (98% reduction)
- Preserves 90% granularity
- Handles rare categories gracefully
\`\`\``
  };

  return <ProblemTemplate data={data} problemNumber={14} />;
};

export default RareCategories;