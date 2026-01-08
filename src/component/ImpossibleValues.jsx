import React from 'react';
import ProblemTemplate from './ProblemTemplate';

const ImpossibleValues = () => {
  const data = {
    title: 'Impossible Values',
    description: 'Impossible values are data points that violate logical or physical constraints. These include negative ages, future birth dates, percentages over 100%, temperatures below absolute zero, etc. They result from data entry errors, system bugs, or corrupted data and must be identified and handled appropriately.',
    originalData: [
      { person_id: 'P001', name: 'John Smith', age: 35, height_cm: 175, weight_kg: 70, birth_year: 1989, temperature: 36.5, test_score: 85 },
      { person_id: 'P002', name: 'Jane Doe', age: -5, height_cm: 160, weight_kg: 55, birth_year: 2030, temperature: 37.2, test_score: 92 },
      { person_id: 'P003', name: 'Bob Wilson', age: 42, height_cm: 0, weight_kg: 80, birth_year: 1982, temperature: 36.8, test_score: 78 },
      { person_id: 'P004', name: 'Alice Brown', age: 28, height_cm: 165, weight_kg: -10, birth_year: 1996, temperature: -50, test_score: 88 },
      { person_id: 'P005', name: 'Charlie Davis', age: 250, height_cm: 180, weight_kg: 75, birth_year: 1774, temperature: 37.0, test_score: 150 },
      { person_id: 'P006', name: 'Diana Miller', age: 31, height_cm: 500, weight_kg: 60, birth_year: 1993, temperature: 120, test_score: 95 },
      { person_id: 'P007', name: 'Ethan Garcia', age: 45, height_cm: 170, weight_kg: 0, birth_year: 1979, temperature: 36.6, test_score: -20 },
      { person_id: 'P008', name: 'Fiona Martinez', age: 0, height_cm: 155, weight_kg: 50, birth_year: 2024, temperature: 37.5, test_score: 82 },
    ],
    cleanedData: [
      { person_id: 'P001', name: 'John Smith', age: 35, height_cm: 175, weight_kg: 70, birth_year: 1989, temperature: 36.5, test_score: 85 },
    ],
    removedColumns: [],
    testDataset: `import pandas as pd
import numpy as np

# Create test dataset with impossible values
data = {
    'person_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008'],
    'name': ['John Smith', 'Jane Doe', 'Bob Wilson', 'Alice Brown',
             'Charlie Davis', 'Diana Miller', 'Ethan Garcia', 'Fiona Martinez'],
    'age': [35, -5, 42, 28, 250, 31, 45, 0],  # Negative, too old, zero
    'height_cm': [175, 160, 0, 165, 180, 500, 170, 155],  # Zero, unrealistic
    'weight_kg': [70, 55, 80, -10, 75, 60, 0, 50],  # Negative, zero
    'birth_year': [1989, 2030, 1982, 1996, 1774, 1993, 1979, 2024],  # Future, too old
    'temperature': [36.5, 37.2, 36.8, -50, 37.0, 120, 36.6, 37.5],  # Extreme values
    'test_score': [85, 92, 78, 88, 150, 95, -20, 82]  # Out of range (0-100)
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print(f"\\nDataset Shape: {df.shape}")

# Show basic statistics to reveal impossible values
print("\\nBasic Statistics:")
print(df.describe())

print("\\n⚠ Notice impossible values:")
print("  - Negative age: -5")
print("  - Age > 120: 250")
print("  - Height = 0 or > 300cm")
print("  - Negative weight: -10")
print("  - Future birth year: 2030")
print("  - Extreme temperatures: -50°C, 120°C")
print("  - Test scores outside 0-100 range")`,
    solution: `# Solution: Detect and handle impossible values using domain constraints

def detect_impossible_values(df, constraints):
    """
    Detect impossible values based on domain-specific constraints
    
    Parameters:
    df: pandas DataFrame
    constraints: dict of {column_name: constraint_dict}
                constraint_dict can have:
                - 'min': minimum valid value
                - 'max': maximum valid value
                - 'positive': must be positive (> 0)
                - 'non_negative': must be >= 0
                - 'range': tuple of (min, max)
    
    Returns:
    violations_report: Dictionary with all violations found
    violation_mask: Boolean mask for rows with any violation
    """
    import pandas as pd
    import numpy as np
    
    violations_report = {}
    violation_indices = set()
    
    print("="*70)
    print("IMPOSSIBLE VALUE DETECTION")
    print("="*70)
    
    for col, constraint in constraints.items():
        if col not in df.columns:
            print(f"\\n⚠ Column '{col}' not found, skipping")
            continue
        
        print(f"\\n{col}:")
        col_violations = []
        
        # Check minimum value
        if 'min' in constraint:
            min_val = constraint['min']
            mask = df[col] < min_val
            if mask.any():
                violators = df[mask]
                col_violations.append({
                    'type': 'below_minimum',
                    'constraint': f'>= {min_val}',
                    'count': mask.sum(),
                    'indices': violators.index.tolist(),
                    'values': violators[col].tolist()
                })
                violation_indices.update(violators.index)
                print(f"  ✗ {mask.sum()} values below minimum ({min_val})")
                print(f"    Rows: {violators.index.tolist()}")
                print(f"    Values: {violators[col].tolist()}")
        
        # Check maximum value
        if 'max' in constraint:
            max_val = constraint['max']
            mask = df[col] > max_val
            if mask.any():
                violators = df[mask]
                col_violations.append({
                    'type': 'above_maximum',
                    'constraint': f'<= {max_val}',
                    'count': mask.sum(),
                    'indices': violators.index.tolist(),
                    'values': violators[col].tolist()
                })
                violation_indices.update(violators.index)
                print(f"  ✗ {mask.sum()} values above maximum ({max_val})")
                print(f"    Rows: {violators.index.tolist()}")
                print(f"    Values: {violators[col].tolist()}")
        
        # Check positive (> 0)
        if constraint.get('positive', False):
            mask = df[col] <= 0
            if mask.any():
                violators = df[mask]
                col_violations.append({
                    'type': 'not_positive',
                    'constraint': '> 0',
                    'count': mask.sum(),
                    'indices': violators.index.tolist(),
                    'values': violators[col].tolist()
                })
                violation_indices.update(violators.index)
                print(f"  ✗ {mask.sum()} values not positive (must be > 0)")
                print(f"    Rows: {violators.index.tolist()}")
                print(f"    Values: {violators[col].tolist()}")
        
        # Check non-negative (>= 0)
        if constraint.get('non_negative', False):
            mask = df[col] < 0
            if mask.any():
                violators = df[mask]
                col_violations.append({
                    'type': 'negative',
                    'constraint': '>= 0',
                    'count': mask.sum(),
                    'indices': violators.index.tolist(),
                    'values': violators[col].tolist()
                })
                violation_indices.update(violators.index)
                print(f"  ✗ {mask.sum()} negative values (must be >= 0)")
                print(f"    Rows: {violators.index.tolist()}")
                print(f"    Values: {violators[col].tolist()}")
        
        if col_violations:
            violations_report[col] = col_violations
        else:
            print(f"  ✓ All values valid")
    
    violation_mask = df.index.isin(violation_indices)
    
    return violations_report, violation_mask


# Define domain-specific constraints
constraints = {
    'age': {'min': 0, 'max': 120},  # Human age range
    'height_cm': {'min': 50, 'max': 250},  # Realistic height range
    'weight_kg': {'positive': True, 'max': 300},  # Must be positive, under 300kg
    'birth_year': {'min': 1900, 'max': 2024},  # Reasonable birth year range
    'temperature': {'min': 35, 'max': 42},  # Normal body temperature range (Celsius)
    'test_score': {'min': 0, 'max': 100}  # Test score range
}

# Detect violations
violations, violation_mask = detect_impossible_values(df, constraints)

print("\\n" + "="*70)
print("VIOLATION SUMMARY")
print("="*70)
print(f"Columns with violations: {len(violations)}")
print(f"Total rows with violations: {violation_mask.sum()}")
print(f"Clean rows: {(~violation_mask).sum()}")

print("\\nViolations by column:")
for col, col_violations in violations.items():
    total_violations = sum(v['count'] for v in col_violations)
    print(f"  {col}: {total_violations} violations")
    for v in col_violations:
        print(f"    - {v['type']}: {v['count']} values {v['constraint']}")

print("\\n" + "="*70)
print("HANDLING OPTIONS")
print("="*70)

# Option 1: Remove rows with violations
print("\\n1. REMOVE VIOLATING ROWS:")
cleaned_df_remove = df[~violation_mask].copy()
print(f"   Original rows: {len(df)}")
print(f"   Removed rows: {violation_mask.sum()}")
print(f"   Remaining rows: {len(cleaned_df_remove)}")
print(f"   Data retention: {len(cleaned_df_remove)/len(df)*100:.1f}%")

# Option 2: Replace with NaN
print("\\n2. REPLACE VIOLATIONS WITH NaN:")
cleaned_df_nan = df.copy()
for col, col_violations in violations.items():
    for v in col_violations:
        cleaned_df_nan.loc[v['indices'], col] = np.nan
print(f"   Rows preserved: {len(cleaned_df_nan)}")
print(f"   Missing values created: {cleaned_df_nan.isnull().sum().sum()}")

# Option 3: Cap at boundaries
print("\\n3. CAP AT VALID BOUNDARIES:")
cleaned_df_cap = df.copy()
for col, constraint in constraints.items():
    if 'min' in constraint:
        cleaned_df_cap[col] = cleaned_df_cap[col].clip(lower=constraint['min'])
    if 'max' in constraint:
        cleaned_df_cap[col] = cleaned_df_cap[col].clip(upper=constraint['max'])
print(f"   Rows preserved: {len(cleaned_df_cap)}")
print(f"   Values capped to valid range")

print("\\n" + "="*70)
print("RECOMMENDED: REMOVE VIOLATING ROWS")
print("="*70)
print("\\nCleaned Dataset (Option 1):")
print(cleaned_df_remove)

# Show detailed violation report
print("\\n" + "="*70)
print("DETAILED VIOLATION REPORT")
print("="*70)
print("\\nRows with violations:")
print(df[violation_mask][['person_id', 'name', 'age', 'height_cm', 'weight_kg', 'temperature', 'test_score']])`,
    explanation: `**What are Impossible Values?**

Values that violate logical, physical, or domain-specific constraints:

**Physical Impossibilities:**
- Age < 0 or > 120 years
- Height = 0 or > 3 meters
- Weight < 0
- Temperature < -273.15°C (absolute zero)
- Speed > speed of light

**Logical Impossibilities:**
- Birth date in the future
- End date before start date
- Negative counts or quantities
- Percentages < 0% or > 100%

**Domain-Specific Impossibilities:**
- Test scores outside valid range (e.g., > 100)
- Product price = $0
- Invalid zip codes, phone numbers
- Days in month > 31

**Why They're Problematic:**

1. **Statistical Distortion**: Skew means, medians, distributions
2. **Model Errors**: ML algorithms learn from bad data
3. **Business Logic Failures**: Triggers incorrect calculations
4. **Data Integrity**: Indicates upstream quality issues
5. **Trust Issues**: Users lose confidence in data

**Common Sources:**

1. **Data Entry Errors**: Typos, transposed digits (35 → 350)
2. **System Bugs**: Calculation errors, overflow/underflow
3. **Unit Confusion**: Mixing cm/inches, kg/lbs
4. **Placeholder Values**: -999, 9999 used as "unknown"
5. **Missing Value Codes**: 0 used instead of NaN
6. **Date/Time Issues**: Wrong timezone, epoch errors
7. **Sensor Malfunctions**: Faulty hardware readings

**Detection Strategies:**

**1. Statistical Approach:**
\`\`\`python
# Check ranges
print(df.describe())
print(df.min())  # Look for negatives where impossible
print(df.max())  # Look for unrealistic maximums
\`\`\`

**2. Domain Rules:**
\`\`\`python
# Define business rules
assert (df['age'] >= 0).all(), "Negative ages found"
assert (df['age'] <= 120).all(), "Unrealistic ages found"
assert (df['price'] > 0).all(), "Zero/negative prices found"
\`\`\`

**3. Logical Consistency:**
\`\`\`python
# Cross-column validation
assert (df['end_date'] >= df['start_date']).all()
assert (df['total'] == df['subtotal'] + df['tax']).all()
\`\`\`

**Handling Strategies:**

**Option 1: Remove Rows** ⭐ RECOMMENDED when violations are few
\`\`\`python
valid_mask = (df['age'] >= 0) & (df['age'] <= 120)
df_clean = df[valid_mask]
\`\`\`
✓ Clean data guaranteed
✗ Lose information

**Option 2: Replace with NaN**
\`\`\`python
df.loc[df['age'] < 0, 'age'] = np.nan
df.loc[df['age'] > 120, 'age'] = np.nan
\`\`\`
✓ Preserve other column values
✗ Creates missing values to handle

**Option 3: Cap/Clip at Boundaries**
\`\`\`python
df['age'] = df['age'].clip(lower=0, upper=120)
\`\`\`
✓ Preserve row count
✗ Introduces artificial values

**Option 4: Flag and Keep**
\`\`\`python
df['age_valid'] = df['age'].between(0, 120)
\`\`\`
✓ Preserve all data
✓ Allows filtering later
✗ Requires handling in analysis

**Option 5: Manual Review**
- Export violations for human review
- Correct at source if possible

**Best Practices:**

1. **Define Constraints Early:**
   Document valid ranges for each field
   \`\`\`python
   constraints = {
       'age': (0, 120),
       'price': (0.01, 1000000),
       'score': (0, 100)
   }
   \`\`\`

2. **Validate at Multiple Stages:**
   - Input validation (forms)
   - Database constraints
   - ETL pipeline checks
   - Analysis-time validation

3. **Investigate Root Causes:**
   Don't just clean—fix upstream issues

4. **Log Violations:**
   Track what was removed and why
   \`\`\`python
   violations = df[~valid_mask]
   violations.to_csv('violations_log.csv')
   \`\`\`

5. **Use Assertions in Code:**
   \`\`\`python
   assert df['age'].between(0, 120).all(), "Age constraint violated"
   \`\`\`

6. **Domain Expert Review:**
   Consult with subject matter experts on edge cases

**Common Constraint Patterns:**

\`\`\`python
# Non-negative
df['quantity'] >= 0

# Positive
df['price'] > 0

# Range
df['percentage'].between(0, 100)

# Date logic
df['end_date'] >= df['start_date']

# Enum/Category
df['status'].isin(['Active', 'Inactive', 'Pending'])

# Cross-field
df['discount'] <= df['price']

# Format validation
df['email'].str.match(r'^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$')
\`\`\`

**Decision Framework:**

**Remove when:**
- Few violations (< 5% of data)
- Critical fields affected
- Can't determine correct value
- Building predictive models

**Replace with NaN when:**
- Many violations (> 10% of data)
- Other columns still valuable
- Will handle missing values later

**Cap/Clip when:**
- Violations are measurement errors
- Need to preserve row count
- Values close to boundary
- Exploratory analysis only

**Flag when:**
- Need manual review
- Unclear if truly invalid
- Large dataset with many violations
- Want flexibility in analysis

**Example Validation Suite:**

\`\`\`python
def validate_dataframe(df):
    """Comprehensive validation"""
    errors = []
    
    # Age checks
    if (df['age'] < 0).any():
        errors.append("Negative ages found")
    if (df['age'] > 120).any():
        errors.append("Unrealistic ages found")
    
    # Price checks
    if (df['price'] <= 0).any():
        errors.append("Invalid prices found")
    
    # Date logic
    if (df['end_date'] < df['start_date']).any():
        errors.append("End date before start date")
    
    # Percentage checks
    if not df['discount_pct'].between(0, 100).all():
        errors.append("Invalid discount percentages")
    
    if errors:
        raise ValueError(f"Validation failed: {errors}")
    
    return True
\`\`\`

**Real-World Example:**

\`\`\`
Before cleaning:
- Average age: 58 years (skewed by age=250)
- Max temperature: 120°C (physically impossible)
- Min weight: -10 kg (data entry error)

After cleaning:
- Average age: 36 years (realistic)
- Temperature range: 35-42°C (normal)
- All weights positive (valid)
\`\`\``
  };

  return <ProblemTemplate data={data} problemNumber={12} />;
};

export default ImpossibleValues;