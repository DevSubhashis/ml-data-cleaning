import React from 'react';
import ProblemTemplate from './ProblemTemplate';

const WrongDataTypes = () => {
  const data = {
    title: 'Wrong Data Types',
    description: 'Columns often have incorrect data types due to data import issues, mixed content, or missing value representations. Wrong types prevent proper analysis, cause errors in operations, and affect model performance. Common issues include numeric data stored as strings and dates stored as text.',
    originalData: [
      { employee_id: '1001', name: 'John Smith', salary: '75000', hire_date: '2020-03-15', is_active: 'True', rating: '4.5', department_code: '101' },
      { employee_id: '1002', name: 'Jane Doe', salary: '82000', hire_date: '2019-07-22', is_active: 'True', rating: '4.8', department_code: '102' },
      { employee_id: '1003', name: 'Bob Wilson', salary: '68000', hire_date: '2021-01-10', is_active: 'False', rating: '4.2', department_code: '101' },
      { employee_id: '1004', name: 'Alice Brown', salary: '91000', hire_date: '2018-11-05', is_active: 'True', rating: '4.9', department_code: '103' },
      { employee_id: '1005', name: 'Charlie Davis', salary: '73000', hire_date: '2020-09-18', is_active: 'True', rating: '4.3', department_code: '102' },
      { employee_id: '1006', name: 'Diana Miller', salary: '85000', hire_date: '2019-04-30', is_active: 'False', rating: '4.6', department_code: '101' },
      { employee_id: '1007', name: 'Ethan Garcia', salary: '79000', hire_date: '2021-06-12', is_active: 'True', rating: '4.7', department_code: '103' },
      { employee_id: '1008', name: 'Fiona Martinez', salary: '88000', hire_date: '2018-08-25', is_active: 'True', rating: '4.4', department_code: '102' },
    ],
    cleanedData: [
      { employee_id: 1001, name: 'John Smith', salary: 75000, hire_date: '2020-03-15', is_active: true, rating: 4.5, department_code: '101' },
      { employee_id: 1002, name: 'Jane Doe', salary: 82000, hire_date: '2019-07-22', is_active: true, rating: 4.8, department_code: '102' },
      { employee_id: 1003, name: 'Bob Wilson', salary: 68000, hire_date: '2021-01-10', is_active: false, rating: 4.2, department_code: '101' },
      { employee_id: 1004, name: 'Alice Brown', salary: 91000, hire_date: '2018-11-05', is_active: true, rating: 4.9, department_code: '103' },
      { employee_id: 1005, name: 'Charlie Davis', salary: 73000, hire_date: '2020-09-18', is_active: true, rating: 4.3, department_code: '102' },
      { employee_id: 1006, name: 'Diana Miller', salary: 85000, hire_date: '2019-04-30', is_active: false, rating: 4.6, department_code: '101' },
      { employee_id: 1007, name: 'Ethan Garcia', salary: 79000, hire_date: '2021-06-12', is_active: true, rating: 4.7, department_code: '103' },
      { employee_id: 1008, name: 'Fiona Martinez', salary: 88000, hire_date: '2018-08-25', is_active: true, rating: 4.4, department_code: '102' },
    ],
    removedColumns: [],
    testDataset: `import pandas as pd
import numpy as np

# Create test dataset with wrong data types
data = {
    'employee_id': ['1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008'],  # Should be int
    'name': ['John Smith', 'Jane Doe', 'Bob Wilson', 'Alice Brown', 
             'Charlie Davis', 'Diana Miller', 'Ethan Garcia', 'Fiona Martinez'],
    'salary': ['75000', '82000', '68000', '91000', '73000', '85000', '79000', '88000'],  # Should be int/float
    'hire_date': ['2020-03-15', '2019-07-22', '2021-01-10', '2018-11-05',
                  '2020-09-18', '2019-04-30', '2021-06-12', '2018-08-25'],  # Should be datetime
    'is_active': ['True', 'True', 'False', 'True', 'True', 'False', 'True', 'True'],  # Should be bool
    'rating': ['4.5', '4.8', '4.2', '4.9', '4.3', '4.6', '4.7', '4.4'],  # Should be float
    'department_code': ['101', '102', '101', '103', '102', '101', '103', '102']  # Keep as string (categorical)
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print(f"\\nDataset Shape: {df.shape}")

# Show the problem: all columns are object type
print("\\nData Types (BEFORE cleaning):")
print(df.dtypes)
print("\\nAll columns are 'object' type - this is wrong!")`,
    solution: `# Solution: Convert columns to appropriate data types

def fix_data_types(df, type_mapping=None, date_columns=None):
    """
    Convert columns to appropriate data types
    
    Parameters:
    df: pandas DataFrame
    type_mapping: dict of {column_name: target_type}
                 Supported types: 'int', 'float', 'bool', 'category', 'datetime'
    date_columns: list of column names to convert to datetime
    
    Returns:
    cleaned_df: DataFrame with corrected data types
    conversion_report: Dictionary with conversion details
    """
    import pandas as pd
    import numpy as np
    
    df_cleaned = df.copy()
    conversion_report = {
        'successful': {},
        'failed': {},
        'warnings': []
    }
    
    print("="*70)
    print("DATA TYPE CONVERSION")
    print("="*70)
    print("\\nOriginal Data Types:")
    print(df.dtypes)
    
    # If no mapping provided, try to infer
    if type_mapping is None:
        type_mapping = {}
    
    print("\\n" + "-"*70)
    print("CONVERSION PROCESS")
    print("-"*70)
    
    for col, target_type in type_mapping.items():
        if col not in df.columns:
            conversion_report['warnings'].append(f"Column '{col}' not found in dataframe")
            continue
        
        print(f"\\n{col}: object → {target_type}")
        original_dtype = df[col].dtype
        
        try:
            if target_type == 'int':
                # Convert to numeric, coerce errors to NaN, then to int
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').astype('Int64')
                print(f"  ✓ Converted to Int64 (nullable integer)")
                
            elif target_type == 'float':
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                print(f"  ✓ Converted to float64")
                
            elif target_type == 'bool':
                # Handle string boolean representations
                bool_map = {
                    'true': True, 'True': True, 'TRUE': True, '1': True, 1: True,
                    'false': False, 'False': False, 'FALSE': False, '0': False, 0: False
                }
                df_cleaned[col] = df_cleaned[col].map(bool_map)
                print(f"  ✓ Converted to bool")
                
            elif target_type == 'category':
                df_cleaned[col] = df_cleaned[col].astype('category')
                print(f"  ✓ Converted to category")
                print(f"    Categories: {df_cleaned[col].cat.categories.tolist()}")
                
            elif target_type == 'datetime':
                df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                print(f"  ✓ Converted to datetime64")
                
            # Check for any NaN values introduced during conversion
            new_nulls = df_cleaned[col].isnull().sum() - df[col].isnull().sum()
            if new_nulls > 0:
                conversion_report['warnings'].append(
                    f"{col}: {new_nulls} values could not be converted (set to NaN)"
                )
                print(f"  ⚠ Warning: {new_nulls} values could not be converted")
            
            conversion_report['successful'][col] = {
                'from': str(original_dtype),
                'to': str(df_cleaned[col].dtype)
            }
            
        except Exception as e:
            conversion_report['failed'][col] = str(e)
            print(f"  ✗ Conversion failed: {str(e)}")
    
    # Handle date columns separately if provided
    if date_columns:
        print("\\n" + "-"*70)
        print("DATE CONVERSIONS")
        print("-"*70)
        for col in date_columns:
            if col in df.columns:
                try:
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                    print(f"\\n{col}:")
                    print(f"  ✓ Converted to datetime64")
                    print(f"    Sample: {df_cleaned[col].iloc[0]}")
                    conversion_report['successful'][col] = {
                        'from': str(df[col].dtype),
                        'to': 'datetime64[ns]'
                    }
                except Exception as e:
                    print(f"  ✗ Failed: {str(e)}")
                    conversion_report['failed'][col] = str(e)
    
    return df_cleaned, conversion_report


# Define the correct data types for each column
type_mapping = {
    'employee_id': 'int',
    'salary': 'int',
    'rating': 'float',
    'is_active': 'bool',
    'department_code': 'category'  # Categorical, not numeric
}

date_columns = ['hire_date']

# Apply the conversion
cleaned_df, report = fix_data_types(df, type_mapping, date_columns)

print("\\n" + "="*70)
print("CONVERSION SUMMARY")
print("="*70)
print(f"\\nSuccessful conversions: {len(report['successful'])}")
for col, info in report['successful'].items():
    print(f"  {col}: {info['from']} → {info['to']}")

if report['failed']:
    print(f"\\nFailed conversions: {len(report['failed'])}")
    for col, error in report['failed'].items():
        print(f"  {col}: {error}")

if report['warnings']:
    print(f"\\nWarnings: {len(report['warnings'])}")
    for warning in report['warnings']:
        print(f"  ⚠ {warning}")

print("\\n" + "="*70)
print("DATA TYPES (AFTER cleaning)")
print("="*70)
print(cleaned_df.dtypes)

print("\\n\\nCleaned Dataset:")
print(cleaned_df)

# Demonstrate the benefits of correct types
print("\\n" + "="*70)
print("BENEFITS OF CORRECT DATA TYPES")
print("="*70)

print("\\n1. Numeric Operations (now possible):")
print(f"   Average salary: \${cleaned_df['salary'].mean():,.2f}")

print(f"   Total payroll: \${cleaned_df['salary'].sum():,.2f}")
print(f"   Salary range: \${cleaned_df['salary'].min():,} - \${cleaned_df['salary'].max():,}")

print("\\n2. Date Operations (now possible):")
cleaned_df['years_employed'] = (pd.Timestamp.now() - cleaned_df['hire_date']).dt.days / 365.25
print(f"   Average years employed: {cleaned_df['years_employed'].mean():.1f} years")

print("\\n3. Boolean Operations (now possible):")
print(f"   Active employees: {cleaned_df['is_active'].sum()}")
print(f"   Inactive employees: {(~cleaned_df['is_active']).sum()}")

print("\\n4. Memory Usage:")
print(f"   Before: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
print(f"   After:  {cleaned_df.memory_usage(deep=True).sum() / 1024:.2f} KB")
print(f"   Savings: {(1 - cleaned_df.memory_usage(deep=True).sum() / df.memory_usage(deep=True).sum()) * 100:.1f}%")`,
explanation: `**Why Wrong Data Types are Problematic?**

1. **Operations Fail**: Can't do math on string "75000"
2. **Sorting Issues**: "100" < "20" when sorted as strings
3. **Memory Waste**: Objects use more memory than numeric types
4. **Analysis Errors**: Statistics computed incorrectly
5. **Model Issues**: ML algorithms expect specific types
6. **Performance**: Numeric operations are much faster on proper types

**Common Wrong Type Scenarios:**

**1. Numeric as Object/String:**

\`\`\`python
'75000' instead of 75000
'4.5' instead of 4.5
'100' instead of 100
\`\`\`
**Problem**: Can't calculate mean, sum, etc.

**2. Dates as Object/String:**
\`\`\`python
'2020-03-15' instead of Timestamp('2020-03-15')
\`\`\`
**Problem**: Can't calculate date differences, sort chronologically

**3. Boolean as Object/String:**
\`\`\`python
'True' instead of True
'False' instead of False
\`\`\`
**Problem**: Can't use logical operations

**4. Categories as Object:**
\`\`\`python
'Low', 'Medium', 'High' as object instead of category
\`\`\`
**Problem**: Wastes memory, no ordering

**Detection Methods:**

\`\`\`python
# Check all data types
print(df.dtypes)

# Find object columns that should be numeric
for col in df.select_dtypes(include=['object']).columns:
    try:
        pd.to_numeric(df[col])
        print(f"{col} can be converted to numeric")
    except:
        print(f"{col} is truly non-numeric")

# Check memory usage
print(df.memory_usage(deep=True))
\`\`\`

**Conversion Strategies:**

**1. Numeric Conversions:**

\`\`\`python
# To integer
df['salary'] = pd.to_numeric(df['salary'], errors='coerce').astype('Int64')

# To float
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# errors='coerce': Invalid values become NaN
# errors='raise': Stops on invalid values
# errors='ignore': Keeps original values
\`\`\`

**2. Date Conversions:**

\`\`\`python
# Automatic format detection
df['hire_date'] = pd.to_datetime(df['hire_date'])

# Specific format
df['hire_date'] = pd.to_datetime(df['hire_date'], format='%Y-%m-%d')

# Handle multiple formats
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
\`\`\`

**3. Boolean Conversions:**

\`\`\`python
# Map string to boolean
bool_map = {'True': True, 'False': False, 'true': True, 'false': False}
df['is_active'] = df['is_active'].map(bool_map)

# Or use replace
df['is_active'] = df['is_active'].replace({'True': True, 'False': False})
\`\`\`

**4. Category Conversions:**

\`\`\`python
# Convert to category
df['department'] = df['department'].astype('category')

# With ordering
df['size'] = pd.Categorical(
    df['size'], 
    categories=['Small', 'Medium', 'Large'], 
    ordered=True
)
\`\`\`

**Special Cases:**

**Nullable Integer (Int64):**
\`\`\`python
# Regular int doesn't support NaN, use Int64
df['count'] = pd.to_numeric(df['count'], errors='coerce').astype('Int64')
\`\`\`

**Mixed Types in Column:**
\`\`\`python
# Example: [100, '200', 'N/A', 300]
df['value'] = pd.to_numeric(df['value'], errors='coerce')
# Result: [100.0, 200.0, NaN, 300.0]
\`\`\`

**Type Inference at Import:**

\`\`\`python
# Let pandas infer types
df = pd.read_csv('data.csv', dtype={'employee_id': int, 'salary': float})

# Parse dates during import
df = pd.read_csv('data.csv', parse_dates=['hire_date'])

# Handle True/False strings
df = pd.read_csv('data.csv', true_values=['True'], false_values=['False'])
\`\`\`

**Best Practices:**

1. **Check Types Immediately:**
   \`\`\`python
   print(df.dtypes)
   print(df.info())
   \`\`\`

2. **Fix at Import Time:**
   Specify dtypes in pd.read_csv()

3. **Validate After Conversion:**
   \`\`\`python
   assert df['salary'].dtype == 'int64'
   \`\`\`

4. **Handle Errors Gracefully:**
   Use errors='coerce' to convert invalid values to NaN

5. **Document Assumptions:**
   Log which columns were converted

6. **Use Appropriate Types:**
   - **int/Int64** for whole numbers
   - **float** for decimals
   - **datetime64** for dates
   - **bool** for True/False
   - **category** for low-cardinality strings

**Common Pitfalls:**

❌ **Converting IDs to numeric**: Keep as string if not used for math
❌ **Forcing invalid conversions**: Check for errors first
❌ **Ignoring timezone**: Use timezone-aware datetime if needed
❌ **Not handling NaN**: Use nullable types (Int64, boolean)

**Type Decision Tree:**

\`\`\`
Is it used for calculations?
├─ Yes → numeric (int/float)
└─ No
   ├─ Is it a date/time?
   │  └─ Yes → datetime64
   └─ No
      ├─ Is it True/False?
      │  └─ Yes → bool
      └─ No
         ├─ Low cardinality (< 50% unique)?
         │  └─ Yes → category
         └─ No → keep as object/string
\`\`\`

**Memory Benefits:**

\`\`\`python
# Before (all object):
# Each value: ~60 bytes
# 8 rows × 7 cols = 3,360 bytes

# After (correct types):
# int: 8 bytes, float: 8 bytes, bool: 1 byte, datetime: 8 bytes
# 8 rows × (8+8+1+8) = 200 bytes
# ~94% memory reduction!
\`\`\`

**Validation After Conversion:**

\`\`\`python
# Check for unexpected NaN values
print(df.isnull().sum())

# Verify ranges
assert df['rating'].between(0, 5).all()
assert df['salary'] > 0).all()

# Check date validity
assert df['hire_date'].max() <= pd.Timestamp.now()
\`\`\``
  };

  return <ProblemTemplate data={data} problemNumber={10} />;
};

export default WrongDataTypes;