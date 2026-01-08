import React from 'react';
import ProblemTemplate from './ProblemTemplate';

const MissingSpaces = () => {
  const data = {
    title: 'Missing " " (Spaces)',
    description: 'Sometimes missing values are represented as whitespace strings (single or multiple spaces) instead of proper null/NaN values. These are hard to detect visually but can cause serious issues in analysis and modeling.',
    originalData: [
      { employee_id: 'E001', name: 'John Doe', department: 'Engineering', salary: 75000, email: 'john@company.com', phone: '555-0101' },
      { employee_id: 'E002', name: 'Jane Smith', department: '  ', salary: 68000, email: 'jane@company.com', phone: '555-0102' },
      { employee_id: 'E003', name: 'Bob Johnson', department: 'Marketing', salary: 62000, email: ' ', phone: '555-0103' },
      { employee_id: 'E004', name: 'Alice Williams', department: 'Sales', salary: 71000, email: 'alice@company.com', phone: '   ' },
      { employee_id: 'E005', name: 'Charlie Brown', department: ' ', salary: 0, email: 'charlie@company.com', phone: '555-0105' },
      { employee_id: 'E006', name: 'Diana Prince', department: 'HR', salary: 65000, email: 'diana@company.com', phone: '555-0106' },
      { employee_id: 'E007', name: 'Ethan Hunt', department: 'Engineering', salary: 78000, email: '  ', phone: '555-0107' },
      { employee_id: 'E008', name: 'Fiona Green', department: 'Sales', salary: 69000, email: 'fiona@company.com', phone: '555-0108' },
    ],
    cleanedData: [
      { employee_id: 'E001', name: 'John Doe', department: 'Engineering', salary: 75000, email: 'john@company.com', phone: '555-0101' },
      { employee_id: 'E002', name: 'Jane Smith', department: 'NaN', salary: 68000, email: 'jane@company.com', phone: '555-0102' },
      { employee_id: 'E003', name: 'Bob Johnson', department: 'Marketing', salary: 62000, email: 'NaN', phone: '555-0103' },
      { employee_id: 'E004', name: 'Alice Williams', department: 'Sales', salary: 71000, email: 'alice@company.com', phone: 'NaN' },
      { employee_id: 'E005', name: 'Charlie Brown', department: 'NaN', salary: 'NaN', email: 'charlie@company.com', phone: '555-0105' },
      { employee_id: 'E006', name: 'Diana Prince', department: 'HR', salary: 65000, email: 'diana@company.com', phone: '555-0106' },
      { employee_id: 'E007', name: 'Ethan Hunt', department: 'Engineering', salary: 78000, email: 'NaN', phone: '555-0107' },
      { employee_id: 'E008', name: 'Fiona Green', department: 'Sales', salary: 69000, email: 'fiona@company.com', phone: '555-0108' },
    ],
    removedColumns: [],
    testDataset: `import pandas as pd
import numpy as np

# Create test dataset with whitespace as missing values
data = {
    'employee_id': ['E001', 'E002', 'E003', 'E004', 'E005', 'E006', 'E007', 'E008'],
    'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Williams', 
             'Charlie Brown', 'Diana Prince', 'Ethan Hunt', 'Fiona Green'],
    'department': ['Engineering', '  ', 'Marketing', 'Sales', ' ', 'HR', 'Engineering', 'Sales'],
    'salary': [75000, 68000, 62000, 71000, '  ', 65000, 78000, 69000],  # Note: whitespace in numeric column
    'email': ['john@company.com', 'jane@company.com', ' ', 'alice@company.com',
              'charlie@company.com', 'diana@company.com', '  ', 'fiona@company.com'],
    'phone': ['555-0101', '555-0102', '555-0103', '   ', '555-0105', '555-0106', '555-0107', '555-0108']
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print(f"\\nDataset Shape: {df.shape}")
print("\\nMissing values (standard detection):")
print(df.isnull().sum())
print("\\nNote: Whitespace strings are NOT detected as missing!")`,
    solution: `# Solution: Identify and replace whitespace strings with proper NaN values

def replace_whitespace_with_nan(df, columns=None):
    """
    Replace whitespace strings (spaces, tabs, etc.) with NaN values
    
    Parameters:
    df: pandas DataFrame
    columns: list of column names to check (None = check all columns)
    
    Returns:
    cleaned_df: DataFrame with whitespace replaced by NaN
    replacement_info: Dictionary with statistics about replacements
    """
    import pandas as pd
    import numpy as np
    
    df_cleaned = df.copy()
    
    if columns is None:
        columns = df.columns
    
    replacement_info = {}
    total_replacements = 0
    
    print("="*70)
    print("WHITESPACE DETECTION AND REPLACEMENT")
    print("="*70)
    
    for col in columns:
        # Convert column to string type to check for whitespace
        col_str = df[col].astype(str)
        
        # Find whitespace-only strings (including empty after strip)
        whitespace_mask = col_str.str.strip() == ''
        whitespace_count = whitespace_mask.sum()
        
        if whitespace_count > 0:
            # Replace with NaN
            df_cleaned.loc[whitespace_mask, col] = np.nan
            replacement_info[col] = whitespace_count
            total_replacements += whitespace_count
            
            print(f"\\n{col}:")
            print(f"  Whitespace values found: {whitespace_count}")
            print(f"  Affected rows: {df[whitespace_mask].index.tolist()}")
    
    print("\\n" + "="*70)
    print("BEFORE vs AFTER COMPARISON")
    print("="*70)
    print("\\nMissing values BEFORE cleaning:")
    print(df.isnull().sum())
    print("\\nMissing values AFTER cleaning:")
    print(df_cleaned.isnull().sum())
    
    # Additional check: Look for strings with only whitespace characters
    print("\\n" + "="*70)
    print("DETAILED ANALYSIS")
    print("="*70)
    for col in columns:
        if df_cleaned[col].dtype == 'object':
            null_count = df_cleaned[col].isnull().sum()
            if null_count > 0:
                print(f"{col}: {null_count} missing values")
    
    summary = {
        'columns_affected': len(replacement_info),
        'total_replacements': total_replacements,
        'details': replacement_info
    }
    
    return df_cleaned, summary

# Apply the solution
cleaned_df, info = replace_whitespace_with_nan(df)

print("\\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Columns with whitespace issues: {info['columns_affected']}")
print(f"Total whitespace values replaced: {info['total_replacements']}")
print("\\nBreakdown by column:")
for col, count in info['details'].items():
    print(f"  {col}: {count} replacements")

print("\\n\\nCleaned Dataset:")
print(cleaned_df)

# Now you can handle missing values properly
print("\\n" + "="*70)
print("NEXT STEPS: Handle Missing Values")
print("="*70)
print("\\nOption 1: Drop rows with missing values")
print(f"Rows before: {len(cleaned_df)}")
print(f"Rows after dropna: {len(cleaned_df.dropna())}")

print("\\nOption 2: Fill missing values")
print("Example: cleaned_df['department'].fillna('Unknown', inplace=True)")

print("\\nOption 3: Drop columns with too many missing values")
missing_pct = (cleaned_df.isnull().sum() / len(cleaned_df) * 100)
print("\\nMissing percentage by column:")
print(missing_pct[missing_pct > 0])`,
    explanation: `**Why Whitespace as Missing Values is Problematic?**

1. **Silent Failures**: Standard .isnull() doesn't detect whitespace strings
2. **Type Confusion**: Numeric columns with whitespace become object type
3. **Analysis Errors**: Whitespace counted as valid data in statistics
4. **Model Issues**: ML algorithms treat whitespace as a category, not missing

**Common Forms of Whitespace:**

\`\`\`python
' '      # Single space
'  '     # Multiple spaces
'\\t'     # Tab character
'\\n'     # Newline
'   \\t'  # Mixed whitespace
\`\`\`

**How They Sneak In:**

1. **User Input**: Form fields with accidental spaces
2. **Excel/CSV Export**: Empty cells exported as spaces
3. **Data Integration**: Legacy systems using space padding
4. **String Concatenation**: Failed joins leaving whitespace
5. **Database Exports**: CHAR fields with space padding

**Detection Techniques:**

\`\`\`python
# Method 1: Strip and check for empty
df[col].str.strip() == ''

# Method 2: Check string length after strip
df[col].str.strip().str.len() == 0

# Method 3: Regex pattern
df[col].str.match(r'^\\s*$')

# Method 4: Replace all whitespace patterns at once
df.replace(r'^\\s*$', np.nan, regex=True)
\`\`\`

**Best Practices:**

1. **Clean on Import**: Replace whitespace when loading data
   \`\`\`python
   df = pd.read_csv('data.csv', na_values=['', ' ', '  ', 'NA', 'null'])
   \`\`\`

2. **Standardize Early**: Clean whitespace before any analysis

3. **Check All Columns**: Don't assume only text columns have this issue

4. **Document Replacements**: Log how many values were affected

5. **Validate Data Types**: After cleaning, ensure numeric columns are numeric
   \`\`\`python
   df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
   \`\`\`

**Related Issues to Check:**

- **Leading/Trailing Spaces**: 'John ' vs 'John' (Problem #11)
- **Mixed Whitespace**: Tabs, newlines mixed with spaces
- **Zero-Width Characters**: Unicode spaces (U+200B, U+FEFF)

**After Replacement:**

Once whitespace is replaced with NaN, you have standard missing value handling options:

1. **Drop**: Remove rows/columns with missing data
2. **Impute**: Fill with mean, median, mode, or sophisticated methods
3. **Flag**: Create indicator column for "was missing"
4. **Model-based**: Use algorithms that handle missing values

**Prevention Tips:**

- Validate input forms (trim whitespace)
- Set database constraints (NOT NULL, CHECK)
- Use proper null representation in data pipelines
- Test data quality at ingestion points

**Quick Visual Check:**

\`\`\`python
# Show unique values including whitespace
for col in df.columns:
    print(f"\\n{col}:")
    print(df[col].unique())
    print(f"Lengths: {df[col].astype(str).str.len().unique()}")
\`\`\``
  };

  return <ProblemTemplate data={data} problemNumber={5} />;
};

export default MissingSpaces;