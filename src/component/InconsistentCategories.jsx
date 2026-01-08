import React from 'react';
import ProblemTemplate from './ProblemTemplate';

const InconsistentCategories = () => {
  const data = {
    title: 'Inconsistent Categories',
    description: 'Categorical data often suffers from inconsistencies like variations in spelling, capitalization, abbreviations, and typos. These inconsistencies cause the same category to be treated as different values, inflating cardinality and causing analysis errors.',
    originalData: [
      { order_id: 'O001', product: 'Laptop', category: 'Electronics', status: 'Delivered', payment: 'Credit Card' },
      { order_id: 'O002', product: 'Mouse', category: 'electronics', status: 'delivered', payment: 'credit card' },
      { order_id: 'O003', product: 'Desk', category: 'Furniture', status: 'Shipped', payment: 'PayPal' },
      { order_id: 'O004', product: 'Chair', category: 'furniture', status: 'DELIVERED', payment: 'Paypal' },
      { order_id: 'O005', product: 'Monitor', category: 'Electronics ', status: 'In Transit', payment: 'Credit card' },
      { order_id: 'O006', product: 'Keyboard', category: 'Electrnics', status: 'delivered', payment: 'Debit Card' },
      { order_id: 'O007', product: 'Lamp', category: 'Furnitur', status: 'shiped', payment: 'PayPal' },
      { order_id: 'O008', product: 'Cable', category: 'ELECTRONICS', status: 'Delivered', payment: 'CC' },
      { order_id: 'O009', product: 'Bookshelf', category: 'Furniture', status: 'in transit', payment: 'Debit card' },
      { order_id: 'O010', product: 'Webcam', category: 'Electronics', status: 'Delivered ', payment: 'Credit Card' },
    ],
    cleanedData: [
      { order_id: 'O001', product: 'Laptop', category: 'Electronics', status: 'Delivered', payment: 'Credit Card' },
      { order_id: 'O002', product: 'Mouse', category: 'Electronics', status: 'Delivered', payment: 'Credit Card' },
      { order_id: 'O003', product: 'Desk', category: 'Furniture', status: 'Shipped', payment: 'PayPal' },
      { order_id: 'O004', product: 'Chair', category: 'Furniture', status: 'Delivered', payment: 'PayPal' },
      { order_id: 'O005', product: 'Monitor', category: 'Electronics', status: 'In Transit', payment: 'Credit Card' },
      { order_id: 'O006', product: 'Keyboard', category: 'Electronics', status: 'Delivered', payment: 'Debit Card' },
      { order_id: 'O007', product: 'Lamp', category: 'Furniture', status: 'Shipped', payment: 'PayPal' },
      { order_id: 'O008', product: 'Cable', category: 'Electronics', status: 'Delivered', payment: 'Credit Card' },
      { order_id: 'O009', product: 'Bookshelf', category: 'Furniture', status: 'In Transit', payment: 'Debit Card' },
      { order_id: 'O010', product: 'Webcam', category: 'Electronics', status: 'Delivered', payment: 'Credit Card' },
    ],
    removedColumns: [],
    testDataset: `import pandas as pd
import numpy as np

# Create test dataset with inconsistent categories
data = {
    'order_id': ['O001', 'O002', 'O003', 'O004', 'O005', 'O006', 'O007', 'O008', 'O009', 'O010'],
    'product': ['Laptop', 'Mouse', 'Desk', 'Chair', 'Monitor', 'Keyboard', 'Lamp', 'Cable', 'Bookshelf', 'Webcam'],
    'category': ['Electronics', 'electronics', 'Furniture', 'furniture', 'Electronics ', 
                 'Electrnics', 'Furnitur', 'ELECTRONICS', 'Furniture', 'Electronics'],
    'status': ['Delivered', 'delivered', 'Shipped', 'DELIVERED', 'In Transit', 
               'delivered', 'shiped', 'Delivered', 'in transit', 'Delivered '],
    'payment': ['Credit Card', 'credit card', 'PayPal', 'Paypal', 'Credit card',
                'Debit Card', 'PayPal', 'CC', 'Debit card', 'Credit Card']
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print(f"\\nDataset Shape: {df.shape}")

# Show the problem: inflated unique values
print("\\nUnique values per categorical column:")
for col in ['category', 'status', 'payment']:
    print(f"\\n{col}: {df[col].nunique()} unique values")
    print(f"  Values: {sorted(df[col].unique())}")`,
    solution: `# Solution: Standardize inconsistent categories

def standardize_categories(df, column_mappings=None):
    """
    Standardize categorical columns by fixing common inconsistencies
    
    Parameters:
    df: pandas DataFrame
    column_mappings: dict of {column_name: {old_value: new_value}}
                    If None, automatic standardization is applied
    
    Returns:
    cleaned_df: DataFrame with standardized categories
    changes_made: Dictionary tracking all changes
    """
    import pandas as pd
    
    df_cleaned = df.copy()
    changes_made = {}
    
    print("="*70)
    print("CATEGORY STANDARDIZATION")
    print("="*70)
    
    # Get categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Automatic standardization steps
    for col in categorical_cols:
        original_values = df[col].copy()
        changes_in_col = []
        
        print(f"\\n{col}:")
        print(f"  Original unique values: {df[col].nunique()}")
        print(f"  Values: {sorted(df[col].unique())}")
        
        # Step 1: Strip leading/trailing whitespace
        df_cleaned[col] = df_cleaned[col].str.strip()
        
        # Step 2: Standardize case (title case for most, but can be customized)
        # Check if column has specific mapping, else use title case
        if column_mappings and col in column_mappings:
            # Apply custom mapping
            df_cleaned[col] = df_cleaned[col].replace(column_mappings[col])
            print(f"  Applied custom mapping")
        else:
            # Auto-standardize: title case
            df_cleaned[col] = df_cleaned[col].str.title()
        
        # Step 3: Fix common typos/abbreviations (can be customized per column)
        # This should ideally be domain-specific
        common_fixes = {
            'Electrnics': 'Electronics',
            'Furnitur': 'Furniture',
            'shiped': 'Shipped',
            'CC': 'Credit Card',
            'Cc': 'Credit Card'
        }
        df_cleaned[col] = df_cleaned[col].replace(common_fixes)
        
        # Track changes
        changed_mask = original_values != df_cleaned[col]
        if changed_mask.any():
            for idx in df[changed_mask].index:
                changes_in_col.append({
                    'row': idx,
                    'from': original_values[idx],
                    'to': df_cleaned[col][idx]
                })
        
        if changes_in_col:
            changes_made[col] = changes_in_col
        
        print(f"  Cleaned unique values: {df_cleaned[col].nunique()}")
        print(f"  Values: {sorted(df_cleaned[col].unique())}")
        print(f"  Changes made: {len(changes_in_col)}")
    
    return df_cleaned, changes_made


def find_similar_categories(df, column, threshold=0.8):
    """
    Find potentially similar categories using fuzzy string matching
    Useful for identifying typos and variations
    
    Parameters:
    df: pandas DataFrame
    column: column name to analyze
    threshold: similarity threshold (0-1), default 0.8
    
    Returns:
    similar_pairs: List of tuples with similar category pairs
    """
    from difflib import SequenceMatcher
    
    unique_vals = df[column].unique()
    similar_pairs = []
    
    print(f"\\n{'='*70}")
    print(f"SIMILARITY ANALYSIS: {column}")
    print(f"{'='*70}")
    
    for i, val1 in enumerate(unique_vals):
        for val2 in unique_vals[i+1:]:
            # Calculate similarity
            similarity = SequenceMatcher(None, val1.lower(), val2.lower()).ratio()
            
            if similarity >= threshold:
                similar_pairs.append((val1, val2, similarity))
                print(f"  '{val1}' ≈ '{val2}' (similarity: {similarity:.2%})")
    
    if not similar_pairs:
        print(f"  No similar categories found (threshold: {threshold:.0%})")
    
    return similar_pairs


# Apply automatic standardization
cleaned_df, changes = standardize_categories(df)

print("\\n" + "="*70)
print("DETAILED CHANGES")
print("="*70)
for col, col_changes in changes.items():
    print(f"\\n{col}: {len(col_changes)} changes")
    for change in col_changes[:5]:  # Show first 5
        print(f"  Row {change['row']}: '{change['from']}' → '{change['to']}'")
    if len(col_changes) > 5:
        print(f"  ... and {len(col_changes) - 5} more")

print("\\n" + "="*70)
print("BEFORE vs AFTER COMPARISON")
print("="*70)
for col in ['category', 'status', 'payment']:
    print(f"\\n{col}:")
    print(f"  Before: {df[col].nunique()} unique values")
    print(f"  After:  {cleaned_df[col].nunique()} unique values")
    print(f"  Reduction: {df[col].nunique() - cleaned_df[col].nunique()} categories")

print("\\n\\nCleaned Dataset:")
print(cleaned_df)

# Show value distributions
print("\\n" + "="*70)
print("VALUE DISTRIBUTIONS AFTER CLEANING")
print("="*70)
for col in ['category', 'status', 'payment']:
    print(f"\\n{col}:")
    print(cleaned_df[col].value_counts().to_string())

# Demonstrate fuzzy matching for remaining potential issues
print("\\n" + "="*70)
print("FUZZY MATCHING CHECK")
print("="*70)
for col in ['category', 'status', 'payment']:
    find_similar_categories(cleaned_df, col, threshold=0.8)`,
    explanation: `**Why Inconsistent Categories are Problematic?**

1. **Inflated Cardinality**: "Electronics" and "electronics" treated as different
2. **Wrong Counts**: Aggregations split across variations
3. **Encoding Issues**: More dummy variables than necessary in ML
4. **Analysis Errors**: Grouping and filtering returns incomplete results
5. **Visualization Mess**: Charts show duplicate categories

**Common Inconsistency Types:**

**1. Case Variations:**
\`\`\`
'Electronics', 'electronics', 'ELECTRONICS', 'ElEcTrOnIcS'
\`\`\`

**2. Whitespace Issues:**
\`\`\`
'Electronics', 'Electronics ', ' Electronics', '  Electronics  '
\`\`\`

**3. Spelling Errors/Typos:**
\`\`\`
'Electronics', 'Electrnics', 'Electroincs', 'Elctronics'
\`\`\`

**4. Abbreviations:**
\`\`\`
'Credit Card', 'CC', 'C.C.', 'Cr Card', 'CrCard'
\`\`\`

**5. Plural vs Singular:**
\`\`\`
'Product', 'Products'
'Category', 'Categories'
\`\`\`

**6. Special Characters:**
\`\`\`
'E-Commerce', 'E Commerce', 'Ecommerce', 'E_Commerce'
\`\`\`

**Detection Strategy:**

\`\`\`python
# 1. Check unique values
df['category'].unique()

# 2. Check value counts
df['category'].value_counts()

# 3. Case-insensitive unique count
df['category'].str.lower().nunique()

# 4. After stripping whitespace
df['category'].str.strip().nunique()
\`\`\`

**Standardization Approaches:**

**Approach 1: Manual Mapping** (Most Accurate)
\`\`\`python
mapping = {
    'electronics': 'Electronics',
    'ELECTRONICS': 'Electronics',
    'Electrnics': 'Electronics',
    'furniture': 'Furniture',
    'Furnitur': 'Furniture'
}
df['category'] = df['category'].replace(mapping)
\`\`\`

**Approach 2: Automatic Case Standardization**
\`\`\`python
# Title Case (Recommended for names)
df['category'] = df['category'].str.title()

# Upper Case
df['category'] = df['category'].str.upper()

# Lower Case
df['category'] = df['category'].str.lower()
\`\`\`

**Approach 3: Fuzzy Matching** (For Typos)
\`\`\`python
from fuzzywuzzy import process

def fuzzy_standardize(value, choices, threshold=80):
    match, score = process.extractOne(value, choices)
    return match if score >= threshold else value

standard_categories = ['Electronics', 'Furniture', 'Clothing']
df['category'] = df['category'].apply(
    lambda x: fuzzy_standardize(x, standard_categories)
)
\`\`\`

**Approach 4: Regular Expressions** (For Patterns)
\`\`\`python
# Remove special characters
df['category'] = df['category'].str.replace(r'[^a-zA-Z0-9\\s]', '', regex=True)

# Standardize separators
df['category'] = df['category'].str.replace(r'[-_]', ' ', regex=True)
\`\`\`

**Best Practices:**

1. **Clean at Import:**
   \`\`\`python
   df = pd.read_csv('data.csv', converters={
       'category': lambda x: x.strip().title()
   })
   \`\`\`

2. **Always Strip First:**
   \`\`\`python
   df['category'] = df['category'].str.strip()
   \`\`\`

3. **Use Master List:**
   Keep a canonical list of valid categories
   \`\`\`python
   valid_categories = ['Electronics', 'Furniture', 'Clothing']
   \`\`\`

4. **Document Mappings:**
   Save your standardization rules
   \`\`\`python
   with open('category_mappings.json', 'w') as f:
       json.dump(mapping, f)
   \`\`\`

5. **Validate After Cleaning:**
   \`\`\`python
   assert df['category'].isin(valid_categories).all()
   \`\`\`

6. **Review Unique Values:**
   Always check before and after
   \`\`\`python
   print("Before:", df['category'].nunique())
   # ... clean ...
   print("After:", df['category'].nunique())
   \`\`\`

**Tools for Detection:**

**1. Value Counts:**
\`\`\`python
df['category'].value_counts()
# Shows if 'Electronics' appears 5 times and 'electronics' 3 times
\`\`\`

**2. Fuzzy String Matching:**
\`\`\`python
from fuzzywuzzy import fuzz
fuzz.ratio('Electronics', 'Electrnics')  # Returns 90
\`\`\`

**3. Edit Distance:**
\`\`\`python
from Levenshtein import distance
distance('Electronics', 'Electrnics')  # Returns 1
\`\`\`

**Common Mistakes:**

❌ **Only fixing case**: Missing typos and whitespace
❌ **Over-automation**: Merging genuinely different categories
❌ **No validation**: Not checking if standardization worked
❌ **Losing information**: Not documenting original values

**Domain-Specific Considerations:**

- **Product Categories**: Usually title case
- **Country Codes**: Usually uppercase (US, UK, IN)
- **Email Domains**: Usually lowercase
- **Names**: Title case, but watch for "McDonald" vs "Mcdonald"
- **Addresses**: Complex rules for abbreviations (St., Street, Str)

**Advanced: Automated Pipeline:**

\`\`\`python
def clean_categories(df, col, standard_values=None):
    # Step 1: Basic cleaning
    df[col] = df[col].str.strip().str.title()
    
    # Step 2: If standard list provided, fuzzy match
    if standard_values:
        df[col] = df[col].apply(
            lambda x: fuzzy_match(x, standard_values)
        )
    
    # Step 3: Log changes
    changes = df[col].value_counts()
    
    return df, changes
\`\`\`

**When to Use Each Method:**

- **Small dataset (< 1000 rows)**: Manual mapping
- **Clear patterns**: Regex and case standardization
- **Typos present**: Fuzzy matching
- **Large dataset**: Automated pipeline with validation`
  };

  return <ProblemTemplate data={data} problemNumber={9} />;
};

export default InconsistentCategories;