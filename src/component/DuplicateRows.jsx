import React from 'react';
import ProblemTemplate from './ProblemTemplate';

const DuplicateRows = () => {
  const data = {
    title: 'Duplicate Rows',
    description: 'Duplicate rows occur when the same record appears multiple times in the dataset. This can happen due to data collection errors, system bugs, or merging datasets. Duplicates can skew analysis and create bias in machine learning models.',
    originalData: [
      { order_id: 1001, customer: 'Alice', product: 'Laptop', quantity: 1, price: 1200, date: '2024-01-15' },
      { order_id: 1002, customer: 'Bob', product: 'Mouse', quantity: 2, price: 25, date: '2024-01-16' },
      { order_id: 1003, customer: 'Charlie', product: 'Keyboard', quantity: 1, price: 75, date: '2024-01-17' },
      { order_id: 1002, customer: 'Bob', product: 'Mouse', quantity: 2, price: 25, date: '2024-01-16' },
      { order_id: 1004, customer: 'Diana', product: 'Monitor', quantity: 1, price: 350, date: '2024-01-18' },
      { order_id: 1005, customer: 'Eve', product: 'Webcam', quantity: 1, price: 80, date: '2024-01-19' },
      { order_id: 1003, customer: 'Charlie', product: 'Keyboard', quantity: 1, price: 75, date: '2024-01-17' },
      { order_id: 1006, customer: 'Frank', product: 'Headset', quantity: 1, price: 120, date: '2024-01-20' },
      { order_id: 1005, customer: 'Eve', product: 'Webcam', quantity: 1, price: 80, date: '2024-01-19' },
    ],
    cleanedData: [
      { order_id: 1001, customer: 'Alice', product: 'Laptop', quantity: 1, price: 1200, date: '2024-01-15' },
      { order_id: 1002, customer: 'Bob', product: 'Mouse', quantity: 2, price: 25, date: '2024-01-16' },
      { order_id: 1003, customer: 'Charlie', product: 'Keyboard', quantity: 1, price: 75, date: '2024-01-17' },
      { order_id: 1004, customer: 'Diana', product: 'Monitor', quantity: 1, price: 350, date: '2024-01-18' },
      { order_id: 1005, customer: 'Eve', product: 'Webcam', quantity: 1, price: 80, date: '2024-01-19' },
      { order_id: 1006, customer: 'Frank', product: 'Headset', quantity: 1, price: 120, date: '2024-01-20' },
    ],
    removedColumns: [],
    testDataset: `import pandas as pd
import numpy as np

# Create test dataset with duplicate rows
data = {
    'order_id': [1001, 1002, 1003, 1002, 1004, 1005, 1003, 1006, 1005],
    'customer': ['Alice', 'Bob', 'Charlie', 'Bob', 'Diana', 'Eve', 'Charlie', 'Frank', 'Eve'],
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Mouse', 'Monitor', 'Webcam', 'Keyboard', 'Headset', 'Webcam'],
    'quantity': [1, 2, 1, 2, 1, 1, 1, 1, 1],
    'price': [1200, 25, 75, 25, 350, 80, 75, 120, 80],
    'date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-16', 
             '2024-01-18', '2024-01-19', '2024-01-17', '2024-01-20', '2024-01-19']
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print(f"\\nDataset Shape: {df.shape}")
print(f"Total Rows: {len(df)}")
print(f"Duplicate Rows: {df.duplicated().sum()}")`,
    solution: `# Solution: Identify and remove duplicate rows

def remove_duplicate_rows(df, subset=None, keep='first'):
    """
    Identify and remove duplicate rows from the dataset
    
    Parameters:
    df: pandas DataFrame
    subset: list of column names to consider for identifying duplicates
            None means all columns (default)
    keep: {'first', 'last', False}
          - 'first': Keep first occurrence, remove subsequent duplicates
          - 'last': Keep last occurrence, remove previous duplicates
          - False: Remove all duplicates (keep none)
    
    Returns:
    cleaned_df: DataFrame without duplicates
    duplicate_info: Dictionary with duplicate statistics
    """
    
    print("="*70)
    print("DUPLICATE DETECTION ANALYSIS")
    print("="*70)
    
    # Find duplicates
    duplicate_mask = df.duplicated(subset=subset, keep=False)
    duplicate_rows = df[duplicate_mask]
    
    print(f"\\nTotal Rows: {len(df)}")
    print(f"Duplicate Rows Found: {duplicate_mask.sum()}")
    print(f"Unique Rows: {len(df) - duplicate_mask.sum()}")
    print(f"Duplicate Sets: {df.duplicated(subset=subset).sum()}")
    
    if duplicate_mask.sum() > 0:
        print(f"\\n{'Column':<20} {'Has Duplicates'}")
        print("-" * 40)
        
        # Show which rows are duplicated
        print(f"\\nDuplicate Row Indices: {df[df.duplicated(subset=subset, keep=False)].index.tolist()}")
        
        print(f"\\nDuplicate Rows Preview:")
        print(duplicate_rows.sort_values(by=df.columns.tolist()[:2]))
    
    # Remove duplicates
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    duplicate_info = {
        'original_rows': len(df),
        'duplicate_count': duplicate_mask.sum(),
        'removed_count': len(df) - len(cleaned_df),
        'final_rows': len(cleaned_df)
    }
    
    return cleaned_df, duplicate_info

# Apply the solution
cleaned_df, info = remove_duplicate_rows(df, keep='first')

print("\\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Original Dataset: {info['original_rows']} rows")
print(f"Duplicates Found: {info['duplicate_count']} rows")
print(f"Rows Removed: {info['removed_count']}")
print(f"Final Dataset: {info['final_rows']} rows")
print(f"Data Reduction: {(info['removed_count']/info['original_rows']*100):.1f}%")

print("\\n\\nCleaned Dataset:")
print(cleaned_df)

# Alternative: Remove duplicates based on specific columns only
print("\\n" + "="*70)
print("ALTERNATIVE: Remove duplicates based on 'order_id' only")
print("="*70)
cleaned_df_subset = df.drop_duplicates(subset=['order_id'], keep='first')
print(f"Rows after removing duplicates by order_id: {len(cleaned_df_subset)}")`,
    explanation: `**Why Remove Duplicate Rows?**

1. **Bias in Analysis**: Duplicates artificially inflate certain patterns
2. **Model Training Issues**: ML models may overfit to duplicated samples
3. **Statistical Integrity**: Skews distributions, means, and correlations
4. **Data Quality**: Often indicates upstream data collection problems

**Types of Duplicates:**

**1. Exact Duplicates:**
- All column values are identical
- Usually caused by system errors or data loading issues
- Safe to remove in most cases

**2. Partial Duplicates:**
- Key columns match, but other columns differ
- May indicate legitimate repeated events or data quality issues
- Requires domain knowledge to decide

**Understanding the 'keep' Parameter:**

\`\`\`python
# keep='first' (default): Keep first occurrence
[A, B, A] → [A, B]

# keep='last': Keep last occurrence  
[A, B, A] → [B, A]

# keep=False: Remove all duplicates
[A, B, A] → [B]
\`\`\`

**Using subset Parameter:**

Sometimes you want to check duplicates based on specific columns only:

\`\`\`python
# Remove duplicates based on order_id only
df.drop_duplicates(subset=['order_id'], keep='first')

# Multiple columns
df.drop_duplicates(subset=['customer', 'date'], keep='first')
\`\`\`

**When NOT to Remove Duplicates:**

1. **Time Series Data**: Same values at different timestamps are valid
2. **Legitimate Repeats**: Customer making multiple identical purchases
3. **Survey Data**: Multiple responses from same person might be valid
4. **Joined Tables**: Duplicates from one-to-many relationships

**Best Practices:**

1. **Investigate First**: Understand WHY duplicates exist before removing
2. **Log Removals**: Keep track of how many duplicates were found
3. **Check Key Columns**: Use subset parameter for business keys (IDs, timestamps)
4. **Visual Inspection**: Review duplicate rows before deletion
5. **Backup Data**: Keep original dataset before cleaning

**Advanced Duplicate Detection:**

\`\`\`python
# Find all duplicate groups
duplicates = df[df.duplicated(keep=False)]
duplicates.sort_values(by=['order_id'])

# Count occurrences of each duplicate
df.groupby(df.columns.tolist()).size().sort_values(ascending=False)

# Find near-duplicates (fuzzy matching)
from fuzzywuzzy import fuzz
# Compare string similarity for text columns
\`\`\`

**Common Causes of Duplicates:**
- Database joins without proper constraints
- Multiple data sources merged incorrectly
- ETL pipeline errors
- User interface allowing duplicate submissions
- Copy-paste errors in manual data entry`
  };

  return <ProblemTemplate data={data} problemNumber={4} />;
};

export default DuplicateRows;