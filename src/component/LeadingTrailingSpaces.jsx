import React from 'react';
import ProblemTemplate from './ProblemTemplate';

const LeadingTrailingSpaces = () => {
  const data = {
    title: 'Leading / Trailing Spaces',
    description: 'Leading and trailing spaces in text data create duplicate-like values that appear identical but are treated as different by the system. This causes issues in matching, grouping, and analysis. These invisible characters are hard to spot visually but critically important to remove.',
    originalData: [
      { customer_id: 'C001', first_name: 'John ', last_name: 'Smith', city: 'New York', country: 'USA', product: 'Laptop' },
      { customer_id: 'C002', first_name: ' Jane', last_name: 'Doe ', city: ' Boston', country: 'USA ', product: 'Mouse ' },
      { customer_id: 'C003', first_name: 'Bob', last_name: '  Wilson', city: 'Chicago  ', country: 'USA', product: 'Keyboard' },
      { customer_id: 'C004', first_name: 'Alice  ', last_name: 'Brown', city: 'Seattle', country: '  USA', product: '  Monitor' },
      { customer_id: 'C005', first_name: 'Charlie', last_name: ' Davis ', city: ' Denver ', country: 'USA', product: 'Webcam' },
      { customer_id: 'C006', first_name: '  Diana', last_name: 'Miller', city: 'Austin', country: 'USA  ', product: ' Cable ' },
      { customer_id: 'C007', first_name: 'Ethan', last_name: 'Garcia  ', city: '  Miami', country: 'USA', product: 'Headset' },
      { customer_id: 'C008', first_name: ' Fiona ', last_name: ' Martinez', city: 'Phoenix', country: ' USA ', product: 'Speaker' },
    ],
    cleanedData: [
      { customer_id: 'C001', first_name: 'John', last_name: 'Smith', city: 'New York', country: 'USA', product: 'Laptop' },
      { customer_id: 'C002', first_name: 'Jane', last_name: 'Doe', city: 'Boston', country: 'USA', product: 'Mouse' },
      { customer_id: 'C003', first_name: 'Bob', last_name: 'Wilson', city: 'Chicago', country: 'USA', product: 'Keyboard' },
      { customer_id: 'C004', first_name: 'Alice', last_name: 'Brown', city: 'Seattle', country: 'USA', product: 'Monitor' },
      { customer_id: 'C005', first_name: 'Charlie', last_name: 'Davis', city: 'Denver', country: 'USA', product: 'Webcam' },
      { customer_id: 'C006', first_name: 'Diana', last_name: 'Miller', city: 'Austin', country: 'USA', product: 'Cable' },
      { customer_id: 'C007', first_name: 'Ethan', last_name: 'Garcia', city: 'Miami', country: 'USA', product: 'Headset' },
      { customer_id: 'C008', first_name: 'Fiona', last_name: 'Martinez', city: 'Phoenix', country: 'USA', product: 'Speaker' },
    ],
    removedColumns: [],
    testDataset: `import pandas as pd
import numpy as np

# Create test dataset with leading/trailing spaces
data = {
    'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005', 'C006', 'C007', 'C008'],
    'first_name': ['John ', ' Jane', 'Bob', 'Alice  ', 'Charlie', '  Diana', 'Ethan', ' Fiona '],
    'last_name': ['Smith', 'Doe ', '  Wilson', 'Brown', ' Davis ', 'Miller', 'Garcia  ', ' Martinez'],
    'city': ['New York', ' Boston', 'Chicago  ', 'Seattle', ' Denver ', 'Austin', '  Miami', 'Phoenix'],
    'country': ['USA', 'USA ', 'USA', '  USA', 'USA', 'USA  ', 'USA', ' USA '],
    'product': ['Laptop', 'Mouse ', 'Keyboard', '  Monitor', 'Webcam', ' Cable ', 'Headset', 'Speaker']
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print(f"\\nDataset Shape: {df.shape}")

# Demonstrate the problem
print("\\n" + "="*70)
print("DEMONSTRATING THE PROBLEM")
print("="*70)

print("\\nUnique countries (looks like 1, but...):")
print(df['country'].unique())
print(f"Count: {df['country'].nunique()} unique values")

print("\\nValue counts for 'country':")
print(df['country'].value_counts())

print("\\nNotice: 'USA', 'USA ', '  USA', ' USA ' are all treated as different!")

# Show string lengths
print("\\n" + "="*70)
print("STRING LENGTHS (reveals hidden spaces)")
print("="*70)
for col in ['first_name', 'last_name', 'city', 'country']:
    print(f"\\n{col}:")
    for idx, val in enumerate(df[col]):
        print(f"  Row {idx}: '{val}' (length: {len(val)})") `,
    solution: `# Solution: Remove leading and trailing spaces from text columns

def remove_leading_trailing_spaces(df, columns=None, also_reduce_internal=False):
    """
    Remove leading and trailing whitespace from text columns
    
    Parameters:
    df: pandas DataFrame
    columns: list of column names to clean (None = all object columns)
    also_reduce_internal: bool, if True, reduce multiple internal spaces to single space
    
    Returns:
    cleaned_df: DataFrame with spaces removed
    cleaning_report: Dictionary with statistics about cleaning
    """
    import pandas as pd
    
    df_cleaned = df.copy()
    cleaning_report = {
        'columns_cleaned': [],
        'total_values_modified': 0,
        'details': {}
    }
    
    # If no columns specified, clean all object (string) columns
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    
    print("="*70)
    print("WHITESPACE CLEANING")
    print("="*70)
    
    for col in columns:
        if col not in df.columns:
            print(f"\\n⚠ Warning: Column '{col}' not found")
            continue
        
        print(f"\\n{col}:")
        
        # Track changes
        original_values = df[col].copy()
        
        # Step 1: Strip leading/trailing spaces
        df_cleaned[col] = df_cleaned[col].str.strip()
        
        # Step 2: Optionally reduce internal multiple spaces to single
        if also_reduce_internal:
            df_cleaned[col] = df_cleaned[col].str.replace(r'\\s+', ' ', regex=True)
            print("  ✓ Stripped leading/trailing spaces + reduced internal spaces")
        else:
            print("  ✓ Stripped leading/trailing spaces")
        
        # Count modifications
        modified_mask = original_values != df_cleaned[col]
        modified_count = modified_mask.sum()
        
        if modified_count > 0:
            cleaning_report['columns_cleaned'].append(col)
            cleaning_report['total_values_modified'] += modified_count
            cleaning_report['details'][col] = {
                'modified_count': modified_count,
                'modified_rows': df[modified_mask].index.tolist()
            }
            
            print(f"  Modified: {modified_count} values")
            print(f"  Rows: {df[modified_mask].index.tolist()}")
            
            # Show some examples
            examples = []
            for idx in df[modified_mask].index[:3]:  # Show first 3
                examples.append({
                    'row': idx,
                    'before': f"'{original_values[idx]}'",
                    'after': f"'{df_cleaned[col][idx]}'"
                })
            
            if examples:
                print("  Examples:")
                for ex in examples:
                    print(f"    Row {ex['row']}: {ex['before']} → {ex['after']}")
        else:
            print(f"  No modifications needed")
    
    return df_cleaned, cleaning_report


# Apply the solution
cleaned_df, report = remove_leading_trailing_spaces(df, also_reduce_internal=True)

print("\\n" + "="*70)
print("CLEANING SUMMARY")
print("="*70)
print(f"Columns cleaned: {len(report['columns_cleaned'])}")
print(f"Total values modified: {report['total_values_modified']}")

print("\\nModifications by column:")
for col, details in report['details'].items():
    print(f"  {col}: {details['modified_count']} values modified")

print("\\n" + "="*70)
print("VERIFICATION - UNIQUE VALUES")
print("="*70)

print("\\nBEFORE cleaning:")
print(f"  'country' unique values: {df['country'].nunique()}")
print(f"  Values: {df['country'].unique().tolist()}")

print("\\nAFTER cleaning:")
print(f"  'country' unique values: {cleaned_df['country'].nunique()}")
print(f"  Values: {cleaned_df['country'].unique().tolist()}")
print("  ✓ Now correctly shows 1 unique value!")

print("\\n" + "="*70)
print("VALUE COUNTS COMPARISON")
print("="*70)

print("\\nBEFORE:")
print(df['country'].value_counts())

print("\\nAFTER:")
print(cleaned_df['country'].value_counts())

print("\\n\\nCleaned Dataset:")
print(cleaned_df)

# Additional verification: Check string lengths
print("\\n" + "="*70)
print("STRING LENGTH VERIFICATION")
print("="*70)
print("\\nSample column 'first_name' lengths:")
print("BEFORE:", df['first_name'].str.len().tolist())
print("AFTER: ", cleaned_df['first_name'].str.len().tolist())

# Advanced: Detect other whitespace characters
print("\\n" + "="*70)
print("ADVANCED: CHECK FOR OTHER WHITESPACE")
print("="*70)

def check_special_whitespace(df, col):
    """Check for tabs, newlines, and other whitespace"""
    has_tab = df[col].str.contains('\\t', na=False).any()
    has_newline = df[col].str.contains('\\n', na=False).any()
    has_carriage = df[col].str.contains('\\r', na=False).any()
    
    print(f"\\n{col}:")
    print(f"  Contains tabs (\\t): {has_tab}")
    print(f"  Contains newlines (\\n): {has_newline}")
    print(f"  Contains carriage returns (\\r): {has_carriage}")
    
    if has_tab or has_newline or has_carriage:
        print("  ⚠ Special whitespace detected! Use regex cleaning.")

for col in ['first_name', 'last_name', 'city', 'country']:
    check_special_whitespace(cleaned_df, col)`,
    explanation: `**Why Leading/Trailing Spaces are Problematic?**

1. **Duplicate Values**: 'USA' ≠ 'USA ' ≠ ' USA' - all treated as different
2. **Matching Fails**: JOIN operations and lookups miss matches
3. **Inflated Cardinality**: Appears to have more unique values than reality
4. **Grouping Issues**: Same category split into multiple groups
5. **Visual Confusion**: Invisible to human eye, hard to debug
6. **Sorting Problems**: ' USA' sorts before 'USA' unexpectedly

**Where They Come From:**

1. **User Input**: Copy-paste from documents, accidental spaces
2. **Data Entry**: Users hitting spacebar before/after text
3. **Excel/CSV Export**: Cell padding in spreadsheets
4. **Database Exports**: VARCHAR fields with space padding
5. **API Responses**: JSON with extra whitespace
6. **Web Forms**: Not trimmed on frontend validation
7. **ETL Pipelines**: String concatenation adding spaces

**Types of Whitespace Issues:**

**1. Leading Spaces:**
\`\`\`
' John' instead of 'John'
'  USA' instead of 'USA'
\`\`\`

**2. Trailing Spaces:**
\`\`\`
'John ' instead of 'John'
'USA  ' instead of 'USA'
\`\`\`

**3. Both:**
\`\`\`
' John ' instead of 'John'
'  USA  ' instead of 'USA'
\`\`\`

**4. Internal Multiple Spaces:**
\`\`\`
'New  York' instead of 'New York'
'John    Doe' instead of 'John Doe'
\`\`\`

**5. Special Whitespace:**
\`\`\`
'John\\t' (tab)
'USA\\n' (newline)
'Name\\r' (carriage return)
\`\`\`

**Detection Methods:**

\`\`\`python
# Method 1: Check unique values
df['column'].unique()  # Shows spaces if you look carefully

# Method 2: Value counts
df['column'].value_counts()  # Shows duplicates with spaces

# Method 3: String length check
df['column'].str.len()  # Different lengths reveal spaces

# Method 4: Compare with stripped version
(df['column'] != df['column'].str.strip()).sum()  # Count values with spaces

# Method 5: Visual representation
for val in df['column'].unique():
    print(f"'{val}' (len: {len(val)})")
\`\`\`

**Cleaning Solutions:**

**Basic Cleaning:**
\`\`\`python
# Strip leading and trailing spaces
df['column'] = df['column'].str.strip()

# Apply to all text columns
text_cols = df.select_dtypes(include=['object']).columns
for col in text_cols:
    df[col] = df[col].str.strip()
\`\`\`

**Clean at Import (BEST):**
\`\`\`python
# Using converters
df = pd.read_csv('data.csv', converters={
    col: lambda x: x.strip() if isinstance(x, str) else x 
    for col in text_columns
})

# Or apply to all columns after import
df = pd.read_csv('data.csv')
df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
\`\`\`

**Advanced Cleaning:**
\`\`\`python
# Remove all types of whitespace (tabs, newlines, etc.)
df['column'] = df['column'].str.replace(r'^\\s+|\\s+$', '', regex=True)

# Reduce multiple internal spaces to single space
df['column'] = df['column'].str.replace(r'\\s+', ' ', regex=True)

# Remove ALL whitespace (not recommended for text with spaces)
df['column'] = df['column'].str.replace(r'\\s+', '', regex=True)
\`\`\`

**Handling Special Characters:**
\`\`\`python
# Remove tabs, newlines, carriage returns
df['column'] = df['column'].str.replace(r'[\\t\\n\\r]', '', regex=True)

# Replace with space first, then strip
df['column'] = df['column'].str.replace(r'[\\t\\n\\r]', ' ', regex=True).str.strip()
\`\`\`

**Best Practices:**

1. **Clean Immediately:**
   Strip spaces as soon as data is loaded

2. **Apply to All Text Columns:**
   Don't assume only specific columns have the issue

3. **Verify After Cleaning:**
   \`\`\`python
   # Check if any spaces remain
   assert not df['column'].str.contains(r'^\\s|\\s$', regex=True).any()
   \`\`\`

4. **Document the Cleaning:**
   Log how many values were modified

5. **Consider Internal Spaces:**
   Decide if multiple internal spaces should be reduced

6. **Handle Special Whitespace:**
   Check for tabs, newlines beyond regular spaces

**Common Mistakes:**

❌ **Only cleaning visible columns**: Missing columns you don't look at
❌ **Not cleaning at import**: Cleaning later means issues propagate
❌ **Assuming no internal issues**: Multiple internal spaces also problematic
❌ **Not verifying**: Spaces can sneak back in during processing

**Real-World Impact Example:**

\`\`\`python
# BEFORE cleaning:
df.groupby('country')['sales'].sum()
# Result:
# 'USA'   → $500,000
# 'USA '  → $200,000  # Lost in separate group!
# ' USA'  → $150,000  # Another separate group!

# AFTER cleaning:
df['country'] = df['country'].str.strip()
df.groupby('country')['sales'].sum()
# Result:
# 'USA' → $850,000  # ✓ All sales correctly grouped!
\`\`\`

**Validation Techniques:**

\`\`\`python
# 1. Check unique value reduction
before = df['column'].nunique()
df['column'] = df['column'].str.strip()
after = df['column'].nunique()
print(f"Reduced from {before} to {after} unique values")

# 2. Find values that still have issues
suspicious = df[df['column'].str.contains(r'^\\s|\\s$', regex=True, na=False)]
if len(suspicious) > 0:
    print("⚠ Still have spacing issues:")
    print(suspicious)

# 3. String length consistency
# Same values should have same length after cleaning
df.groupby('column')['column'].apply(lambda x: x.str.len().nunique())
# Should all be 1
\`\`\`

**Preventive Measures:**

**1. Database Level:**
\`\`\`sql
-- Add TRIM in SQL queries
SELECT TRIM(column_name) FROM table
\`\`\`

**2. Application Level:**
\`\`\`javascript
// JavaScript form validation
input.value = input.value.trim();
\`\`\`

**3. ETL Pipeline:**
\`\`\`python
# Add trimming step in data pipeline
df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
\`\`\`

**Related Problems:**
- Problem #5: Whitespace-only strings (' ', '  ')
- Problem #9: Inconsistent categories (can be caused by spaces)
- Problem #16: Corrupted text (may include special whitespace)`
  };

  return <ProblemTemplate data={data} problemNumber={11} />;
};

export default LeadingTrailingSpaces;