// ============= Problem Components =============

import React from 'react';
import ProblemTemplate from './ProblemTemplate';


const SingleValueColumn = () => {
  const data = {
    title: 'Single-value Columns',
    description: 'Columns that contain only one unique value provide no information for analysis or modeling. They should be identified and removed.',
    originalData: [
      { customer_id: 1, name: 'Alice', country: 'USA', age: 25, status: 'active', purchase_amount: 100 },
      { customer_id: 2, name: 'Bob', country: 'USA', age: 30, status: 'active', purchase_amount: 150 },
      { customer_id: 3, name: 'Charlie', country: 'USA', age: 35, status: 'active', purchase_amount: 200 },
      { customer_id: 4, name: 'David', country: 'USA', age: 28, status: 'active', purchase_amount: 120 },
      { customer_id: 5, name: 'Eve', country: 'USA', age: 42, status: 'active', purchase_amount: 180 },
    ],
    cleanedData: [
      { customer_id: 1, name: 'Alice', age: 25, purchase_amount: 100 },
      { customer_id: 2, name: 'Bob', age: 30, purchase_amount: 150 },
      { customer_id: 3, name: 'Charlie', age: 35, purchase_amount: 200 },
      { customer_id: 4, name: 'David', age: 28, purchase_amount: 120 },
      { customer_id: 5, name: 'Eve', age: 42, purchase_amount: 180 },
    ],
    removedColumns: ['country', 'status'],
    testDataset: `import pandas as pd
import numpy as np

# Create test dataset with single-value columns
data = {
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'country': ['USA', 'USA', 'USA', 'USA', 'USA'],  # Single value
    'age': [25, 30, 35, 28, 42],
    'status': ['active', 'active', 'active', 'active', 'active'],  # Single value
    'purchase_amount': [100, 150, 200, 120, 180]
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print("\\nDataset Shape:", df.shape)`,
    solution: `# Solution: Identify and remove single-value columns

def remove_single_value_columns(df):
    """
    Identify and remove columns with only one unique value
    
    Parameters:
    df: pandas DataFrame
    
    Returns:
    cleaned_df: DataFrame without single-value columns
    removed_cols: List of removed column names
    """
    # Identify single-value columns
    single_value_cols = [col for col in df.columns 
                         if df[col].nunique() <= 1]
    
    print(f"\\nSingle-value columns found: {single_value_cols}")
    
    # Remove single-value columns
    cleaned_df = df.drop(columns=single_value_cols)
    
    return cleaned_df, single_value_cols

# Apply the solution
cleaned_df, removed = remove_single_value_columns(df)

print("\\nCleaned Dataset:")
print(cleaned_df)
print("\\nCleaned Dataset Shape:", cleaned_df.shape)
print(f"\\nRemoved columns: {removed}")

# Verification
print("\\n--- Verification ---")
for col in df.columns:
    unique_count = df[col].nunique()
    print(f"{col}: {unique_count} unique values")`,
    explanation: `**Why Remove Single-Value Columns?**

1. **No Information Gain**: Columns with a single value don't help differentiate between rows
2. **Computational Efficiency**: Reduces memory usage and processing time
3. **Model Performance**: Prevents unnecessary features in ML models

**When to Keep Them:**
- If they represent important metadata (e.g., dataset version)
- For documentation purposes in reports

**Best Practice:**
Always log which columns were removed for transparency and reproducibility.`
  };

  return <ProblemTemplate data={data} problemNumber={1} />;
};

export default SingleValueColumn;