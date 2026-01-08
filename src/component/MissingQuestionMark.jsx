import React from 'react';
import ProblemTemplate from './ProblemTemplate';

const MissingQuestionMark = () => {
  const data = {
    title: 'Missing "?"',
    description: 'Question marks ("?", "??", "???") are sometimes used to represent unknown or missing values in datasets. These need to be identified and converted to proper NaN values for accurate data analysis and modeling.',
    originalData: [
      { student_id: 'S001', name: 'Alex Chen', major: 'Computer Science', gpa: 3.8, credits: 120, advisor: 'Dr. Smith', graduation_year: 2024 },
      { student_id: 'S002', name: 'Bella Martinez', major: '?', gpa: 3.5, credits: 90, advisor: 'Dr. Johnson', graduation_year: 2025 },
      { student_id: 'S003', name: 'Chris Taylor', major: 'Mathematics', gpa: '?', credits: 105, advisor: '??', graduation_year: 2024 },
      { student_id: 'S004', name: 'Diana Lee', major: 'Physics', gpa: 3.9, credits: '?', advisor: 'Dr. Williams', graduation_year: 2024 },
      { student_id: 'S005', name: 'Ethan Brown', major: 'Engineering', gpa: 3.2, credits: 115, advisor: 'Dr. Davis', graduation_year: '?' },
      { student_id: 'S006', name: 'Fiona Garcia', major: '???', gpa: 3.7, credits: 95, advisor: 'Dr. Miller', graduation_year: 2025 },
      { student_id: 'S007', name: 'George Wilson', major: 'Biology', gpa: 3.4, credits: 100, advisor: '?', graduation_year: 2026 },
      { student_id: 'S008', name: 'Hannah Moore', major: 'Chemistry', gpa: '??', credits: 110, advisor: 'Dr. Anderson', graduation_year: 2025 },
    ],
    cleanedData: [
      { student_id: 'S001', name: 'Alex Chen', major: 'Computer Science', gpa: 3.8, credits: 120, advisor: 'Dr. Smith', graduation_year: 2024 },
      { student_id: 'S002', name: 'Bella Martinez', major: 'NaN', gpa: 3.5, credits: 90, advisor: 'Dr. Johnson', graduation_year: 2025 },
      { student_id: 'S003', name: 'Chris Taylor', major: 'Mathematics', gpa: 'NaN', credits: 105, advisor: 'NaN', graduation_year: 2024 },
      { student_id: 'S004', name: 'Diana Lee', major: 'Physics', gpa: 3.9, credits: 'NaN', advisor: 'Dr. Williams', graduation_year: 2024 },
      { student_id: 'S005', name: 'Ethan Brown', major: 'Engineering', gpa: 3.2, credits: 115, advisor: 'Dr. Davis', graduation_year: 'NaN' },
      { student_id: 'S006', name: 'Fiona Garcia', major: 'NaN', gpa: 3.7, credits: 95, advisor: 'Dr. Miller', graduation_year: 2025 },
      { student_id: 'S007', name: 'George Wilson', major: 'Biology', gpa: 3.4, credits: 100, advisor: 'NaN', graduation_year: 2026 },
      { student_id: 'S008', name: 'Hannah Moore', major: 'Chemistry', gpa: 'NaN', credits: 110, advisor: 'Dr. Anderson', graduation_year: 2025 },
    ],
    removedColumns: [],
    testDataset: `import pandas as pd
import numpy as np

# Create test dataset with question marks as missing values
data = {
    'student_id': ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008'],
    'name': ['Alex Chen', 'Bella Martinez', 'Chris Taylor', 'Diana Lee',
             'Ethan Brown', 'Fiona Garcia', 'George Wilson', 'Hannah Moore'],
    'major': ['Computer Science', '?', 'Mathematics', 'Physics', 'Engineering', '???', 'Biology', 'Chemistry'],
    'gpa': [3.8, 3.5, '?', 3.9, 3.2, 3.7, 3.4, '??'],  # Mixed: numbers and "?" strings
    'credits': [120, 90, 105, '?', 115, 95, 100, 110],
    'advisor': ['Dr. Smith', 'Dr. Johnson', '??', 'Dr. Williams', 'Dr. Davis', 'Dr. Miller', '?', 'Dr. Anderson'],
    'graduation_year': [2024, 2025, 2024, 2024, '?', 2025, 2026, 2025]
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print(f"\\nDataset Shape: {df.shape}")
print("\\nData types:")
print(df.dtypes)
print("\\nMissing values (standard detection):")
print(df.isnull().sum())
print("\\nNote: Question mark '?' values are NOT detected as missing!")

# Show unique values to spot the question marks
print("\\nUnique values per column:")
for col in df.columns:
    print(f"{col}: {df[col].unique()}")`,
    solution: `# Solution: Identify and replace question mark variations with NaN

def replace_question_marks_with_nan(df, question_marks=None):
    """
    Replace question mark representations with NaN values
    
    Parameters:
    df: pandas DataFrame
    question_marks: list of question mark variations to replace (default: common ones)
    
    Returns:
    cleaned_df: DataFrame with question marks replaced by NaN
    replacement_info: Dictionary with statistics about replacements
    """
    import pandas as pd
    import numpy as np
    
    # Default question mark variations if not provided
    if question_marks is None:
        question_marks = ['?', '??', '???', '????', '?????']
    
    df_cleaned = df.copy()
    replacement_info = {}
    total_replacements = 0
    
    print("="*70)
    print("QUESTION MARK DETECTION AND REPLACEMENT")
    print("="*70)
    print(f"\\nSearching for these question mark patterns: {question_marks}")
    
    for col in df.columns:
        replacements_in_col = 0
        col_original = df[col].copy()
        
        # Replace each question mark variation
        for qmark in question_marks:
            mask = df_cleaned[col].astype(str) == qmark
            
            if mask.any():
                df_cleaned.loc[mask, col] = np.nan
                replacements_in_col += mask.sum()
        
        if replacements_in_col > 0:
            replacement_info[col] = replacements_in_col
            total_replacements += replacements_in_col
            
            print(f"\\n{col}:")
            print(f"  Replacements made: {replacements_in_col}")
            
            # Show which values were replaced
            changed_mask = (col_original.astype(str) != df_cleaned[col].astype(str)) & col_original.notna()
            if changed_mask.any():
                original_vals = col_original[changed_mask].unique()
                print(f"  Question mark values replaced: {original_vals.tolist()}")
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
    print("\\nData types AFTER (before conversion):")
    print(df_cleaned.dtypes)
    
    # Try to convert columns to appropriate types
    print("\\n" + "="*70)
    print("AUTOMATIC TYPE CONVERSION")
    print("="*70)
    for col in df_cleaned.columns:
        original_dtype = df_cleaned[col].dtype
        
        if df_cleaned[col].dtype == 'object':
            # Try converting to numeric
            try:
                numeric_col = pd.to_numeric(df_cleaned[col], errors='coerce')
                # Check if conversion was successful (no additional NaNs created)
                if numeric_col.notna().sum() == df_cleaned[col].notna().sum():
                    df_cleaned[col] = numeric_col
                    print(f"✓ {col}: object → {df_cleaned[col].dtype}")
                else:
                    # Some values couldn't be converted
                    non_numeric = df_cleaned[col][df_cleaned[col].notna() & numeric_col.isna()]
                    if len(non_numeric) > 0:
                        print(f"✗ {col}: Contains non-numeric values: {non_numeric.unique().tolist()}")
            except Exception as e:
                print(f"✗ {col}: Conversion failed - {str(e)}")
    
    # Pattern detection: Check for mixed question marks with text
    print("\\n" + "="*70)
    print("ADDITIONAL PATTERN CHECK")
    print("="*70)
    print("Checking for question marks within text values...")
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if any values contain '?' but aren't just '?'
            mixed_values = df[col][df[col].notna() & df[col].astype(str).str.contains('\\\\?', na=False) & (df[col].astype(str) != '?')]
            if len(mixed_values) > 0:
                print(f"⚠ {col}: Found '?' within text: {mixed_values.unique().tolist()}")
    
    summary = {
        'columns_affected': len(replacement_info),
        'total_replacements': total_replacements,
        'details': replacement_info,
        'question_marks_searched': question_marks
    }
    
    return df_cleaned, summary

# Apply the solution
cleaned_df, info = replace_question_marks_with_nan(df)

print("\\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Columns with question mark issues: {info['columns_affected']}")
print(f"Total question marks replaced: {info['total_replacements']}")
print("\\nBreakdown by column:")
for col, count in info['details'].items():
    print(f"  {col}: {count} replacements")

print("\\n\\nCleaned Dataset:")
print(cleaned_df)
print("\\nFinal data types:")
print(cleaned_df.dtypes)

# Missing value analysis
print("\\n" + "="*70)
print("MISSING VALUE ANALYSIS")
print("="*70)
missing_summary = cleaned_df.isnull().sum()
missing_pct = (missing_summary / len(cleaned_df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing_summary,
    'Missing %': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])`,
    explanation: `**Why Question Marks are Problematic?**

1. **Not Standard Missing Value**: Pandas doesn't recognize '?' as missing
2. **Type Corruption**: Numeric columns with '?' become object type
3. **Visual Ambiguity**: '?' might be legitimate data in some contexts
4. **Multiple Variations**: ?, ??, ???, ???? all mean "unknown" but are different strings

**Where Question Marks Come From:**

1. **Legacy Databases**: Older systems used '?' for unknown values
2. **Manual Data Entry**: Users typing '?' when uncertain
3. **Import from Questionnaires**: Survey responses where '?' means "prefer not to answer"
4. **Text File Imports**: Character encoding issues creating '?'
5. **Data Migration**: Old system conventions carried forward
6. **Excel Formulas**: Error values displayed as '?'

**Question Mark Variations:**

\`\`\`python
Single:    '?'
Double:    '??'
Triple:    '???'
Multiple:  '????', '?????'

Mixed with spaces:
' ? ', '? ', ' ?'

Within text:
'Unknown?', '???value', 'data?'
\`\`\`

**Detection Challenges:**

Unlike 'na' or whitespace, '?' can be:
- **Legitimate punctuation**: "Is this correct?" 
- **Part of valid data**: "Question?"
- **Missing value indicator**: standalone '?'

**Best Detection Strategy:**

\`\`\`python
# Method 1: Replace only standalone question marks
df.replace(['?', '??', '???'], np.nan, inplace=True)

# Method 2: Regex for exact match
df.replace(r'^\\?+$', np.nan, regex=True, inplace=True)

# Method 3: During import (RECOMMENDED)
df = pd.read_csv('data.csv', na_values=['?', '??', '???'])

# Method 4: Column-specific (when you know which columns)
df['gpa'].replace('?', np.nan, inplace=True)
\`\`\`

**Critical Distinction:**

✓ **Replace**: '?' (standalone)
✗ **Keep**: 'What?' or 'Question?' (legitimate text with ?)

\`\`\`python
# WRONG: Replaces ALL question marks, including in text
df.replace('?', np.nan)  # Bad!

# RIGHT: Replaces only standalone question marks
df.replace(r'^\\?+$', np.nan, regex=True)  # Good!
\`\`\`

**Type Conversion After Cleaning:**

Just like with 'na' strings, numeric columns need conversion:

\`\`\`python
# Convert to numeric after replacing '?'
df['gpa'] = pd.to_numeric(df['gpa'], errors='coerce')
df['credits'] = pd.to_numeric(df['credits'], errors='coerce')
df['graduation_year'] = pd.to_numeric(df['graduation_year'], errors='coerce')

# Verify conversion
print(df.dtypes)
\`\`\`

**Edge Cases to Consider:**

1. **Repeated Question Marks**:
   - '??' often means "very uncertain"
   - '???' might indicate "completely unknown"
   - All should be treated as missing

2. **Whitespace + Question Mark**:
   - ' ? ', '? ', ' ?' all represent missing
   - Need to strip whitespace first

3. **Unicode Variations**:
   - '¿' (inverted question mark)
   - '？' (full-width question mark)
   - '﹖' (small question mark)

**Best Practices:**

1. **Inspect First**: Check unique values to see '?' patterns
   \`\`\`python
   for col in df.columns:
       print(f"{col}: {df[col].unique()}")
   \`\`\`

2. **Use Regex for Precision**: Match only standalone '?'
   \`\`\`python
   df.replace(r'^\\?+$', np.nan, regex=True)
   \`\`\`

3. **Handle at Import**: Specify in na_values parameter

4. **Document Assumptions**: Log that '?' is treated as missing

5. **Verify No Data Loss**: Ensure legitimate '?' in text isn't removed

**Validation After Cleaning:**

\`\`\`python
# Check for remaining standalone question marks
for col in df.columns:
    qmark_check = df[col].astype(str).str.match(r'^\\?+$')
    if qmark_check.any():
        print(f"⚠ {col}: Still contains standalone '?' - {qmark_check.sum()} values")

# Look for question marks within text (might be legitimate)
for col in df.columns:
    if df[col].dtype == 'object':
        mixed = df[col].str.contains('\\?', na=False) & (df[col].str.strip() != '?')
        if mixed.any():
            print(f"ℹ {col}: Contains '?' within text - review manually")
            print(df[col][mixed].unique())
\`\`\`

**Related Problems:**

- Problem #5: Whitespace (' ')
- Problem #6: "na" strings
- Problem #10: Wrong data types
- Problem #16: Corrupted text (encoding issues creating '?')

**International Considerations:**

Different cultures use different symbols:
- Spanish: '¿?' for questions
- Some databases: '？' (full-width)
- Legacy systems: '¿' for unknown`
  };

  return <ProblemTemplate data={data} problemNumber={7} />;
};

export default MissingQuestionMark;