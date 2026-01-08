import React from 'react';
import ProblemTemplate from './ProblemTemplate';

const MissingNA = () => {
  const data = {
    title: 'Missing "na"',
    description: 'Sometimes missing values are represented as the string "na", "NA", "N/A", or variations instead of proper null values. These string representations must be identified and converted to actual NaN values for proper data analysis.',
    originalData: [
      { patient_id: 'P001', name: 'John Smith', age: 45, blood_type: 'O+', medication: 'Aspirin', allergies: 'None', last_visit: '2024-01-15' },
      { patient_id: 'P002', name: 'Mary Johnson', age: 'na', blood_type: 'A+', medication: 'na', allergies: 'Penicillin', last_visit: '2024-02-20' },
      { patient_id: 'P003', name: 'Robert Brown', age: 38, blood_type: 'B-', medication: 'Insulin', allergies: 'NA', last_visit: '2024-01-10' },
      { patient_id: 'P004', name: 'Linda Davis', age: 52, blood_type: 'N/A', medication: 'Metformin', allergies: 'None', last_visit: 'na' },
      { patient_id: 'P005', name: 'James Wilson', age: 29, blood_type: 'AB+', medication: 'n/a', allergies: 'Latex', last_visit: '2024-03-05' },
      { patient_id: 'P006', name: 'Patricia Moore', age: 67, blood_type: 'O-', medication: 'Warfarin', allergies: 'na', last_visit: '2024-02-28' },
      { patient_id: 'P007', name: 'Michael Taylor', age: 'NA', blood_type: 'A-', medication: 'Lisinopril', allergies: 'None', last_visit: '2024-01-22' },
      { patient_id: 'P008', name: 'Jennifer Anderson', age: 41, blood_type: 'B+', medication: 'NA', allergies: 'Sulfa', last_visit: '2024-03-12' },
    ],
    cleanedData: [
      { patient_id: 'P001', name: 'John Smith', age: 45, blood_type: 'O+', medication: 'Aspirin', allergies: 'None', last_visit: '2024-01-15' },
      { patient_id: 'P002', name: 'Mary Johnson', age: 'NaN', blood_type: 'A+', medication: 'NaN', allergies: 'Penicillin', last_visit: '2024-02-20' },
      { patient_id: 'P003', name: 'Robert Brown', age: 38, blood_type: 'B-', medication: 'Insulin', allergies: 'NaN', last_visit: '2024-01-10' },
      { patient_id: 'P004', name: 'Linda Davis', age: 52, blood_type: 'NaN', medication: 'Metformin', allergies: 'None', last_visit: 'NaN' },
      { patient_id: 'P005', name: 'James Wilson', age: 29, blood_type: 'AB+', medication: 'NaN', allergies: 'Latex', last_visit: '2024-03-05' },
      { patient_id: 'P006', name: 'Patricia Moore', age: 67, blood_type: 'O-', medication: 'Warfarin', allergies: 'NaN', last_visit: '2024-02-28' },
      { patient_id: 'P007', name: 'Michael Taylor', age: 'NaN', blood_type: 'A-', medication: 'Lisinopril', allergies: 'None', last_visit: '2024-01-22' },
      { patient_id: 'P008', name: 'Jennifer Anderson', age: 41, blood_type: 'B+', medication: 'NaN', allergies: 'Sulfa', last_visit: '2024-03-12' },
    ],
    removedColumns: [],
    testDataset: `import pandas as pd
import numpy as np

# Create test dataset with "na" string variations as missing values
data = {
    'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008'],
    'name': ['John Smith', 'Mary Johnson', 'Robert Brown', 'Linda Davis',
             'James Wilson', 'Patricia Moore', 'Michael Taylor', 'Jennifer Anderson'],
    'age': [45, 'na', 38, 52, 29, 67, 'NA', 41],  # Mixed: numbers and "na" strings
    'blood_type': ['O+', 'A+', 'B-', 'N/A', 'AB+', 'O-', 'A-', 'B+'],
    'medication': ['Aspirin', 'na', 'Insulin', 'Metformin', 'n/a', 'Warfarin', 'Lisinopril', 'NA'],
    'allergies': ['None', 'Penicillin', 'NA', 'None', 'Latex', 'na', 'None', 'Sulfa'],
    'last_visit': ['2024-01-15', '2024-02-20', '2024-01-10', 'na', '2024-03-05', '2024-02-28', '2024-01-22', '2024-03-12']
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print(f"\\nDataset Shape: {df.shape}")
print("\\nData types:")
print(df.dtypes)
print("\\nMissing values (standard detection):")
print(df.isnull().sum())
print("\\nNote: String 'na' values are NOT detected as missing!")`,
    solution: `# Solution: Identify and replace "na" string variations with proper NaN values

def replace_na_strings_with_nan(df, na_values=None, case_sensitive=False):
    """
    Replace various "na" string representations with NaN values
    
    Parameters:
    df: pandas DataFrame
    na_values: list of strings to treat as missing (default: common variations)
    case_sensitive: bool, whether to match case exactly (default: False)
    
    Returns:
    cleaned_df: DataFrame with "na" strings replaced by NaN
    replacement_info: Dictionary with statistics about replacements
    """
    import pandas as pd
    import numpy as np
    
    # Default NA string variations if not provided
    if na_values is None:
        na_values = ['na', 'NA', 'N/A', 'n/a', 'Na', 'N.A.', 'n.a.', 
                     'null', 'NULL', 'Null', 'none', 'NONE', 'None',
                     '-', '--', '---', 'missing', 'MISSING']
    
    df_cleaned = df.copy()
    replacement_info = {}
    total_replacements = 0
    
    print("="*70)
    print("NA STRING DETECTION AND REPLACEMENT")
    print("="*70)
    print(f"\\nSearching for these NA representations: {na_values[:10]}...")
    print(f"Case sensitive: {case_sensitive}")
    
    for col in df.columns:
        replacements_in_col = 0
        col_original = df[col].copy()
        
        # Replace each NA variation
        for na_val in na_values:
            if case_sensitive:
                mask = df_cleaned[col] == na_val
            else:
                # Case-insensitive comparison for string columns
                if df_cleaned[col].dtype == 'object':
                    mask = df_cleaned[col].astype(str).str.lower() == na_val.lower()
                else:
                    mask = df_cleaned[col] == na_val
            
            if mask.any():
                df_cleaned.loc[mask, col] = np.nan
                replacements_in_col += mask.sum()
        
        if replacements_in_col > 0:
            replacement_info[col] = replacements_in_col
            total_replacements += replacements_in_col
            
            print(f"\\n{col}:")
            print(f"  Replacements made: {replacements_in_col}")
            
            # Show which values were replaced
            changed_mask = col_original != df_cleaned[col]
            if changed_mask.any():
                original_vals = col_original[changed_mask].unique()
                print(f"  Original values replaced: {original_vals.tolist()}")
                print(f"  Affected rows: {df[changed_mask].index.tolist()}")
    
    print("\\n" + "="*70)
    print("BEFORE vs AFTER COMPARISON")
    print("="*70)
    print("\\nMissing values BEFORE cleaning:")
    before_missing = df.isnull().sum()
    print(before_missing[before_missing > 0] if (before_missing > 0).any() else "No missing values detected")
    
    print("\\nMissing values AFTER cleaning:")
    after_missing = df_cleaned.isnull().sum()
    print(after_missing[after_missing > 0] if (after_missing > 0).any() else "No missing values")
    
    # Data type check
    print("\\n" + "="*70)
    print("DATA TYPE VERIFICATION")
    print("="*70)
    print("\\nData types BEFORE:")
    print(df.dtypes)
    print("\\nData types AFTER:")
    print(df_cleaned.dtypes)
    
    # Try to convert columns to appropriate types
    print("\\n" + "="*70)
    print("TYPE CONVERSION ATTEMPT")
    print("="*70)
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            # Try converting to numeric
            try:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='ignore')
                if df_cleaned[col].dtype != 'object':
                    print(f"{col}: Converted to {df_cleaned[col].dtype}")
            except:
                pass
    
    summary = {
        'columns_affected': len(replacement_info),
        'total_replacements': total_replacements,
        'details': replacement_info,
        'na_values_searched': na_values
    }
    
    return df_cleaned, summary

# Apply the solution
cleaned_df, info = replace_na_strings_with_nan(df)

print("\\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Columns with 'na' string issues: {info['columns_affected']}")
print(f"Total 'na' strings replaced: {info['total_replacements']}")
print("\\nBreakdown by column:")
for col, count in info['details'].items():
    print(f"  {col}: {count} replacements")

print("\\n\\nCleaned Dataset:")
print(cleaned_df)
print("\\nFinal data types:")
print(cleaned_df.dtypes)

# Demonstrate missing value handling after cleaning
print("\\n" + "="*70)
print("HANDLING MISSING VALUES - OPTIONS")
print("="*70)
print(f"\\nTotal missing values: {cleaned_df.isnull().sum().sum()}")
print(f"Rows with any missing: {cleaned_df.isnull().any(axis=1).sum()}")
print(f"Complete rows: {(~cleaned_df.isnull().any(axis=1)).sum()}")`,
    explanation: `**Why "na" Strings are Problematic?**

1. **Not Recognized as Missing**: Pandas treats "na" as a valid string value
2. **Type Conflicts**: Numeric columns become object type due to string "na"
3. **Analysis Errors**: "na" included in string operations and statistics
4. **Model Confusion**: ML algorithms treat "na" as a category, not missing data

**Common "NA" String Variations:**

\`\`\`
Case variations:
'na', 'NA', 'Na', 'nA'

With delimiters:
'N/A', 'n/a', 'N.A.', 'n.a.'

Related terms:
'null', 'NULL', 'Null'
'none', 'NONE', 'None'
'missing', 'MISSING', 'Missing'
'--', '---', 'N.A'
'not available', 'Not Available'
'#N/A' (Excel export)
\`\`\`

**Where They Come From:**

1. **Manual Data Entry**: Users typing "na" instead of leaving blank
2. **Legacy Systems**: Old databases using string placeholders
3. **Excel Exports**: #N/A errors exported as strings
4. **Survey Data**: "N/A" as a response option
5. **API Responses**: Systems returning "null" as string
6. **Data Integration**: Different systems using different conventions

**Detection Strategy:**

\`\`\`python
# Method 1: Replace during import (BEST)
df = pd.read_csv('data.csv', 
                 na_values=['na', 'NA', 'N/A', 'null', 'none'])

# Method 2: Replace after loading
df.replace(['na', 'NA', 'N/A'], np.nan, inplace=True)

# Method 3: Case-insensitive regex
df.replace(r'(?i)^n/?a$', np.nan, regex=True, inplace=True)

# Method 4: Column-specific replacement
df['age'].replace('na', np.nan, inplace=True)
\`\`\`

**Critical: Type Conversion After Cleaning**

Once "na" strings are replaced, numeric columns need type conversion:

\`\`\`python
# Convert to numeric (force conversion, coerce errors to NaN)
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# Check if successful
print(df['age'].dtype)  # Should be int64 or float64
\`\`\`

**Best Practices:**

1. **Handle at Import**: Specify na_values when reading files
   \`\`\`python
   pd.read_csv('file.csv', na_values=['na', 'NA', 'N/A', '#N/A'])
   \`\`\`

2. **Document Conventions**: Know what NA strings your data sources use

3. **Case-Insensitive**: Always search for both upper and lowercase

4. **Verify Types**: Check and convert data types after cleaning

5. **Log Replacements**: Track how many values were changed

**Common Mistakes:**

❌ **Only checking 'NA'**: Missing lowercase 'na' variations
❌ **Case-sensitive search**: Missing 'Na', 'nA' variations  
❌ **Not converting types**: Leaving numeric columns as object type
❌ **Forgetting delimiters**: Missing 'N/A', 'N.A.' variations

**Validation After Cleaning:**

\`\`\`python
# Check for remaining "na" strings
for col in df.columns:
    if df[col].dtype == 'object':
        na_check = df[col].str.lower().str.contains('na', na=False)
        if na_check.any():
            print(f"{col}: Still contains 'na' strings!")

# Verify numeric columns are truly numeric
numeric_cols = df.select_dtypes(include=['object']).columns
for col in numeric_cols:
    try:
        pd.to_numeric(df[col], errors='raise')
        print(f"{col}: Can be converted to numeric")
    except:
        print(f"{col}: Contains non-numeric values")
\`\`\`

**Related Problems:**

- Problem #5: Whitespace strings (' ', '  ')
- Problem #7: Question marks ('?', '??')
- Problem #10: Wrong data types (after na string replacement)

**International Considerations:**

Different languages/regions use different NA representations:
- Spanish: 'no disponible', 'n/d'
- French: 'non disponible', 'n/d'  
- German: 'nicht verfügbar', 'n.v.'
- Japanese: '不明', 'N/A'`
  };

  return <ProblemTemplate data={data} problemNumber={6} />;
};

export default MissingNA;