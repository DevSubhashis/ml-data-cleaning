import React from 'react';
import ProblemTemplate from './ProblemTemplate';

const VeryFewUniqueValues = () => {
  const data = {
    title: 'Very Few Unique Values',
    description: 'Columns with very few unique values relative to the total number of rows may not provide meaningful information for analysis. These low-cardinality features can sometimes be removed or need special handling.',
    originalData: [
      { product_id: 1, name: 'Laptop Pro', category: 'Electronics', brand: 'TechCorp', color: 'Silver', stock: 45, rating: 4.5 },
      { product_id: 2, name: 'Gaming Mouse', category: 'Electronics', brand: 'TechCorp', color: 'Black', stock: 120, rating: 4.7 },
      { product_id: 3, name: 'Keyboard RGB', category: 'Electronics', brand: 'TechCorp', color: 'Black', stock: 89, rating: 4.3 },
      { product_id: 4, name: 'USB Cable', category: 'Electronics', brand: 'TechCorp', color: 'Black', stock: 200, rating: 4.1 },
      { product_id: 5, name: 'Monitor 27"', category: 'Electronics', brand: 'TechCorp', color: 'Black', stock: 34, rating: 4.6 },
      { product_id: 6, name: 'Webcam HD', category: 'Electronics', brand: 'TechCorp', color: 'Black', stock: 67, rating: 4.2 },
      { product_id: 7, name: 'Headphones', category: 'Electronics', brand: 'TechCorp', color: 'Black', stock: 95, rating: 4.4 },
      { product_id: 8, name: 'Mouse Pad', category: 'Electronics', brand: 'TechCorp', color: 'Black', stock: 150, rating: 4.0 },
    ],
    cleanedData: [
      { product_id: 1, name: 'Laptop Pro', color: 'Silver', stock: 45, rating: 4.5 },
      { product_id: 2, name: 'Gaming Mouse', color: 'Black', stock: 120, rating: 4.7 },
      { product_id: 3, name: 'Keyboard RGB', color: 'Black', stock: 89, rating: 4.3 },
      { product_id: 4, name: 'USB Cable', color: 'Black', stock: 200, rating: 4.1 },
      { product_id: 5, name: 'Monitor 27"', color: 'Black', stock: 34, rating: 4.6 },
      { product_id: 6, name: 'Webcam HD', color: 'Black', stock: 67, rating: 4.2 },
      { product_id: 7, name: 'Headphones', color: 'Black', stock: 95, rating: 4.4 },
      { product_id: 8, name: 'Mouse Pad', color: 'Black', stock: 150, rating: 4.0 },
    ],
    removedColumns: ['category', 'brand'],
    testDataset: `import pandas as pd
import numpy as np

# Create test dataset with very few unique values
data = {
    'product_id': range(1, 9),
    'name': ['Laptop Pro', 'Gaming Mouse', 'Keyboard RGB', 'USB Cable', 
             'Monitor 27"', 'Webcam HD', 'Headphones', 'Mouse Pad'],
    'category': ['Electronics'] * 8,  # Only 1 unique value (12.5% cardinality)
    'brand': ['TechCorp'] * 8,  # Only 1 unique value (12.5% cardinality)
    'color': ['Silver', 'Black', 'Black', 'Black', 'Black', 'Black', 'Black', 'Black'],  # 2 unique (25%)
    'stock': [45, 120, 89, 200, 34, 67, 95, 150],
    'rating': [4.5, 4.7, 4.3, 4.1, 4.6, 4.2, 4.4, 4.0]
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print("\\nDataset Shape:", df.shape)
print("\\nUnique Values per Column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")`,
    solution: `# Solution: Identify and remove columns with very few unique values

def remove_low_cardinality_columns(df, threshold=0.05):
    """
    Identify and remove columns with very few unique values
    
    Parameters:
    df: pandas DataFrame
    threshold: float, minimum ratio of unique values to total rows (default: 5%)
               Columns with unique_values/total_rows < threshold will be removed
    
    Returns:
    cleaned_df: DataFrame without low-cardinality columns
    removed_cols: Dictionary with removed columns and their stats
    """
    removed_cols = {}
    cols_to_remove = []
    
    print(f"\\nAnalyzing columns with cardinality threshold: {threshold*100}%")
    print("-" * 60)
    
    for col in df.columns:
        unique_count = df[col].nunique()
        total_count = len(df)
        cardinality_ratio = unique_count / total_count
        
        print(f"{col}:")
        print(f"  Unique values: {unique_count}/{total_count}")
        print(f"  Cardinality ratio: {cardinality_ratio:.2%}")
        
        if cardinality_ratio < threshold:
            cols_to_remove.append(col)
            removed_cols[col] = {
                'unique_count': unique_count,
                'cardinality_ratio': cardinality_ratio
            }
            print(f"  ❌ REMOVED (below {threshold*100}% threshold)")
        else:
            print(f"  ✓ KEPT")
        print()
    
    # Remove low-cardinality columns
    cleaned_df = df.drop(columns=cols_to_remove)
    
    return cleaned_df, removed_cols

# Apply the solution
cleaned_df, removed = remove_low_cardinality_columns(df, threshold=0.05)

print("\\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {cleaned_df.shape}")
print(f"\\nRemoved {len(removed)} column(s):")
for col, stats in removed.items:
    print(f"  - {col}: {stats['unique_count']} unique ({stats['cardinality_ratio']:.2%})")

print("\\n\\nCleaned Dataset:")
print(cleaned_df)`,
    explanation: `**Why Remove Low-Cardinality Columns?**

1. **Limited Information**: Columns with very few unique values provide minimal discriminatory power
2. **Overfitting Risk**: In small datasets, these features can cause models to memorize rather than generalize
3. **Noise Reduction**: Removes features that may represent data collection artifacts rather than meaningful patterns

**Cardinality Ratio Formula:**
\`\`\`
Cardinality Ratio = (Number of Unique Values) / (Total Number of Rows)
\`\`\`

**Common Thresholds:**
- **< 1%**: Almost certainly remove (except single-value columns)
- **1-5%**: Consider removing, especially in large datasets
- **5-10%**: Evaluate based on domain knowledge
- **> 10%**: Usually keep

**When to Keep Low-Cardinality Columns:**
- Binary/Boolean features (Yes/No, True/False) - these are often meaningful
- Categorical features with business importance (e.g., VIP status, region codes)
- In small datasets where even limited variation matters
- When the column is a known predictor in your domain

**Best Practice:**
Instead of automatic removal, consider:
1. One-hot encoding for meaningful low-cardinality categoricals
2. Feature engineering to combine with other features
3. Domain expert consultation before removal`
  };

  return <ProblemTemplate data={data} problemNumber={2} />;
};

export default VeryFewUniqueValues;