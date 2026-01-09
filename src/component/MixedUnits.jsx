import React from 'react';
import ProblemTemplate from './ProblemTemplate';

const MixedUnits = () => {
  const data = {
    title: 'Mixed Units',
    description: 'Mixed units occur when the same measurement is recorded in different units within a single column. This creates inconsistent data that cannot be directly compared or aggregated. Common examples include mixing meters/feet, kilograms/pounds, Celsius/Fahrenheit, or currencies.',
    originalData: [
      { product_id: 'PR001', name: 'Laptop A', weight: '2.5 kg', length: '35 cm', price: '$1200', temperature: '25°C' },
      { product_id: 'PR002', name: 'Laptop B', weight: '5.5 lbs', length: '14 inches', price: '€1100', temperature: '77°F' },
      { product_id: 'PR003', name: 'Monitor', weight: '4.2 kg', length: '60 cm', price: '$350', temperature: '23°C' },
      { product_id: 'PR004', name: 'Keyboard', weight: '1.8 lbs', length: '45 cm', price: '£85', temperature: '72°F' },
      { product_id: 'PR005', name: 'Mouse', weight: '0.15 kg', length: '4 inches', price: '$25', temperature: '24°C' },
      { product_id: 'PR006', name: 'Webcam', weight: '0.4 lbs', length: '8 cm', price: '€45', temperature: '75°F' },
      { product_id: 'PR007', name: 'Headset', weight: '0.3 kg', length: '7 inches', price: '$120', temperature: '22°C' },
      { product_id: 'PR008', name: 'Cable', weight: '2 oz', length: '2 meters', price: '₹500', temperature: '70°F' },
    ],
    cleanedData: [
      { product_id: 'PR001', name: 'Laptop A', weight_kg: 2.5, length_cm: 35, price_usd: 1200, temperature_c: 25 },
      { product_id: 'PR002', name: 'Laptop B', weight_kg: 2.49, length_cm: 35.56, price_usd: 1188, temperature_c: 25 },
      { product_id: 'PR003', name: 'Monitor', weight_kg: 4.2, length_cm: 60, price_usd: 350, temperature_c: 23 },
      { product_id: 'PR004', name: 'Keyboard', weight_kg: 0.82, length_cm: 45, price_usd: 107, temperature_c: 22.22 },
      { product_id: 'PR005', name: 'Mouse', weight_kg: 0.15, length_cm: 10.16, price_usd: 25, temperature_c: 24 },
      { product_id: 'PR006', name: 'Webcam', weight_kg: 0.18, length_cm: 8, price_usd: 49, temperature_c: 23.89 },
      { product_id: 'PR007', name: 'Headset', weight_kg: 0.3, length_cm: 17.78, price_usd: 120, temperature_c: 22 },
      { product_id: 'PR008', name: 'Cable', weight_kg: 0.06, length_cm: 200, price_usd: 6, temperature_c: 21.11 },
    ],
    removedColumns: [],
    testDataset: `import pandas as pd
import numpy as np
import re

# Create test dataset with mixed units
data = {
    'product_id': ['PR001', 'PR002', 'PR003', 'PR004', 'PR005', 'PR006', 'PR007', 'PR008'],
    'name': ['Laptop A', 'Laptop B', 'Monitor', 'Keyboard', 'Mouse', 'Webcam', 'Headset', 'Cable'],
    'weight': ['2.5 kg', '5.5 lbs', '4.2 kg', '1.8 lbs', '0.15 kg', '0.4 lbs', '0.3 kg', '2 oz'],
    'length': ['35 cm', '14 inches', '60 cm', '45 cm', '4 inches', '8 cm', '7 inches', '2 meters'],
    'price': ['$1200', '€1100', '$350', '£85', '$25', '€45', '$120', '₹500'],
    'temperature': ['25°C', '77°F', '23°C', '72°F', '24°C', '75°F', '22°C', '70°F']
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print(f"\\nDataset Shape: {df.shape}")

print("\\n⚠ PROBLEM: Multiple units in same columns:")
print("  - weight: kg, lbs, oz")
print("  - length: cm, inches, meters")
print("  - price: $, €, £, ₹")
print("  - temperature: °C, °F")

print("\\nThis makes calculations impossible!")
print("Example: Can't calculate average weight when mixing kg and lbs")`,
    solution: `# Solution: Detect and standardize mixed units

def standardize_mixed_units(df, column_configs):
    """
    Detect and convert mixed units to a standard unit
    
    Parameters:
    df: pandas DataFrame
    column_configs: dict of {column_name: config_dict}
                   config_dict should have:
                   - 'patterns': dict of {unit_pattern: conversion_factor_to_standard}
                   - 'standard_unit': name of standard unit
                   - 'new_column_name': name for standardized column
    
    Returns:
    cleaned_df: DataFrame with standardized units
    conversion_report: Dictionary with conversion details
    """
    import pandas as pd
    import numpy as np
    import re
    
    df_cleaned = df.copy()
    conversion_report = {}
    
    print("="*70)
    print("MIXED UNIT STANDARDIZATION")
    print("="*70)
    
    for col, config in column_configs.items():
        if col not in df.columns:
            print(f"\\n⚠ Column '{col}' not found, skipping")
            continue
        
        print(f"\\n{col}:")
        print(f"  Target unit: {config['standard_unit']}")
        
        patterns = config['patterns']
        new_col = config.get('new_column_name', f"{col}_{config['standard_unit']}")
        
        # Initialize new column with NaN
        df_cleaned[new_col] = np.nan
        
        conversions = {}
        
        for pattern, conversion_func in patterns.items():
            # Find values matching this pattern
            mask = df[col].astype(str).str.contains(pattern, case=False, regex=True, na=False)
            
            if mask.any():
                count = mask.sum()
                print(f"  Found {count} values with unit pattern: {pattern}")
                
                # Extract numeric value
                for idx in df[mask].index:
                    value_str = str(df.loc[idx, col])
                    
                    # Extract number (handles decimal, negative)
                    number_match = re.search(r'[-+]?\\d*\\.?\\d+', value_str)
                    if number_match:
                        number = float(number_match.group())
                        
                        # Apply conversion
                        if callable(conversion_func):
                            converted = conversion_func(number)
                        else:
                            converted = number * conversion_func
                        
                        df_cleaned.loc[idx, new_col] = converted
                        
                        if pattern not in conversions:
                            conversions[pattern] = []
                        conversions[pattern].append({
                            'original': value_str,
                            'converted': converted
                        })
        
        # Show conversion examples
        for unit_pattern, examples in conversions.items():
            print(f"\\n  Examples for {unit_pattern}:")
            for ex in examples[:3]:  # Show first 3
                print(f"    {ex['original']} → {ex['converted']:.2f} {config['standard_unit']}")
        
        conversion_report[col] = {
            'new_column': new_col,
            'standard_unit': config['standard_unit'],
            'conversions': conversions,
            'total_converted': len(df_cleaned[df_cleaned[new_col].notna()])
        }
    
    return df_cleaned, conversion_report


# Define conversion configurations
column_configs = {
    'weight': {
        'standard_unit': 'kg',
        'new_column_name': 'weight_kg',
        'patterns': {
            r'kg|kilogram': 1.0,  # kg to kg
            r'lbs?|pound': 0.453592,  # lbs to kg
            r'oz|ounce': 0.0283495,  # oz to kg
            r'g\\b|gram': 0.001  # g to kg
        }
    },
    'length': {
        'standard_unit': 'cm',
        'new_column_name': 'length_cm',
        'patterns': {
            r'cm|centimeter': 1.0,  # cm to cm
            r'inch|in\\b|"': 2.54,  # inches to cm
            r'meter|m\\b': 100.0,  # meters to cm
            r'ft|foot|feet': 30.48,  # feet to cm
            r'mm|millimeter': 0.1  # mm to cm
        }
    },
    'temperature': {
        'standard_unit': 'celsius',
        'new_column_name': 'temperature_c',
        'patterns': {
            r'°?C|celsius': 1.0,  # Celsius to Celsius
            r'°?F|fahrenheit': lambda f: (f - 32) * 5/9,  # Fahrenheit to Celsius
            r'°?K|kelvin': lambda k: k - 273.15  # Kelvin to Celsius
        }
    },
    'price': {
        'standard_unit': 'usd',
        'new_column_name': 'price_usd',
        'patterns': {
            r'\\$|USD|usd': 1.0,  # USD to USD
            r'€|EUR|eur': 1.08,  # EUR to USD (example rate)
            r'£|GBP|gbp': 1.26,  # GBP to USD (example rate)
            r'₹|INR|inr': 0.012,  # INR to USD (example rate)
            r'¥|JPY|jpy': 0.0067  # JPY to USD (example rate)
        }
    }
}

# Apply standardization
cleaned_df, report = standardize_mixed_units(df, column_configs)

print("\\n" + "="*70)
print("CONVERSION SUMMARY")
print("="*70)
for col, info in report.items():
    print(f"\\n{col}:")
    print(f"  New column: {info['new_column']}")
    print(f"  Standard unit: {info['standard_unit']}")
    print(f"  Values converted: {info['total_converted']}")

print("\\n" + "="*70)
print("CLEANED DATASET")
print("="*70)
# Select relevant columns for display
display_cols = ['product_id', 'name', 'weight_kg', 'length_cm', 'price_usd', 'temperature_c']
print(cleaned_df[display_cols])

# Now we can do calculations!
print("\\n" + "="*70)
print("NOW WE CAN PERFORM CALCULATIONS!")
print("="*70)

print("\\nWeight Statistics (in kg):")
print(f"  Average: {cleaned_df['weight_kg'].mean():.2f} kg")
print(f"  Min: {cleaned_df['weight_kg'].min():.2f} kg")
print(f"  Max: {cleaned_df['weight_kg'].max():.2f} kg")
print(f"  Total: {cleaned_df['weight_kg'].sum():.2f} kg")

print("\\nLength Statistics (in cm):")
print(f"  Average: {cleaned_df['length_cm'].mean():.2f} cm")
print(f"  Range: {cleaned_df['length_cm'].min():.2f} - {cleaned_df['length_cm'].max():.2f} cm")

print("\\nPrice Statistics (in USD):")
print(f"  Average: \${cleaned_df['price_usd'].mean():.2f}")
print(f"  Total: \${cleaned_df['price_usd'].sum():.2f}")

print("\\nTemperature Statistics (in °C):")
print(f"  Average: {cleaned_df['temperature_c'].mean():.2f}°C")
print(f"  Range: {cleaned_df['temperature_c'].min():.2f}°C - {cleaned_df['temperature_c'].max():.2f}°C")`,
    explanation: `**What are Mixed Units?**

Mixed units occur when the same measurement type is recorded using different unit systems within a single column. This creates fundamentally incomparable data.

**Common Examples:**

**1. Weight/Mass:**
- kg, lbs, oz, g, tons, stones

**2. Length/Distance:**
- cm, inches, feet, meters, miles, km

**3. Temperature:**
- Celsius (°C), Fahrenheit (°F), Kelvin (K)

**4. Currency:**
- USD ($), EUR (€), GBP (£), INR (₹), JPY (¥)

**5. Volume:**
- liters, gallons, ml, cups, oz

**6. Time:**
- seconds, minutes, hours, days

**Why Mixed Units are Problematic:**

1. **Calculations Impossible**: Can't average 5 kg and 10 lbs directly
2. **Comparison Invalid**: Is 2 kg > 5 lbs? Can't tell without conversion
3. **Aggregation Wrong**: Sum of mixed units is meaningless
4. **Sorting Incorrect**: 100 cm vs 1 meter will sort wrong
5. **Visualization Broken**: Charts with mixed units are misleading
6. **Model Training Fails**: ML algorithms can't learn from inconsistent scales

**Where They Come From:**

1. **Multiple Data Sources**: Combining US and international data
2. **Regional Differences**: EU uses metric, US uses imperial
3. **Historical Data**: System changed units over time
4. **Manual Entry**: Users entering in preferred units
5. **Import/Export**: Different systems using different standards
6. **Legacy Systems**: Old data in old units
7. **User Preferences**: App allowing unit selection

**Detection Methods:**

\`\`\`python
# Method 1: Visual inspection of unique values
df['weight'].unique()
# Output: ['2.5 kg', '5.5 lbs', '0.15 kg', ...]

# Method 2: Check for unit markers
has_kg = df['weight'].str.contains('kg', case=False, na=False)
has_lbs = df['weight'].str.contains('lbs|lb', case=False, na=False)
print(f"Has kg: {has_kg.sum()}, Has lbs: {has_lbs.sum()}")

# Method 3: Extract units
df['unit'] = df['weight'].str.extract(r'([a-zA-Z]+)', expand=False)
print(df['unit'].value_counts())

# Method 4: Check numeric ranges (if units stored separately)
# 100 kg vs 100 lbs have very different implications
\`\`\`

**Conversion Strategies:**

**1. Choose Standard Unit:**
- Weight → kg (metric) or lbs (imperial)
- Length → cm or inches
- Temperature → Celsius (science) or Fahrenheit (US)
- Currency → USD or local currency

**2. Create Conversion Functions:**

\`\`\`python
# Weight conversions
def lbs_to_kg(lbs):
    return lbs * 0.453592

def oz_to_kg(oz):
    return oz * 0.0283495

# Temperature conversions
def fahrenheit_to_celsius(f):
    return (f - 32) * 5/9

def kelvin_to_celsius(k):
    return k - 273.15

# Length conversions
def inches_to_cm(inches):
    return inches * 2.54

def feet_to_cm(feet):
    return feet * 30.48
\`\`\`

**3. Parse and Convert:**

\`\`\`python
import re

def standardize_weight(value_str):
    # Extract number
    number = float(re.search(r'\\d+\\.?\\d*', value_str).group())
    
    # Detect unit and convert
    if 'kg' in value_str.lower():
        return number  # Already in kg
    elif 'lbs' in value_str.lower() or 'lb' in value_str.lower():
        return number * 0.453592
    elif 'oz' in value_str.lower():
        return number * 0.0283495
    else:
        return None  # Unknown unit

df['weight_kg'] = df['weight'].apply(standardize_weight)
\`\`\`

**Best Practices:**

1. **Document Standard Units:**
   Clearly state which unit is used in cleaned data
   \`\`\`python
   # Column: weight_kg (all weights in kilograms)
   # Column: price_usd (all prices in US dollars)
   \`\`\`

2. **Keep Original Values:**
   Create new standardized columns, don't overwrite originals
   \`\`\`python
   df['weight_kg'] = standardized_weights
   # Keep df['weight'] for reference
   \`\`\`

3. **Handle Edge Cases:**
   - Missing units: Assume default or flag for review
   - Ambiguous units: 'oz' could be fluid oz or weight oz
   - Multiple numbers: '5 ft 10 in' requires special parsing

4. **Use Appropriate Precision:**
   \`\`\`python
   # Round to reasonable precision
   df['weight_kg'] = df['weight_kg'].round(2)  # 2 decimal places
   df['temperature_c'] = df['temperature_c'].round(1)
   \`\`\`

5. **Validate Conversions:**
   \`\`\`python
   # Check for unrealistic values after conversion
   assert df['weight_kg'].between(0, 1000).all()
   \`\`\`

6. **Update Currency Regularly:**
   Exchange rates change - either use live rates or document rate date
   \`\`\`python
   # EUR to USD at rate of 1.08 (as of 2024-01-08)
   \`\`\`

**Common Conversion Factors:**

**Weight:**
- 1 kg = 2.20462 lbs
- 1 lb = 0.453592 kg
- 1 oz = 0.0283495 kg
- 1 stone = 6.35029 kg

**Length:**
- 1 inch = 2.54 cm
- 1 foot = 30.48 cm
- 1 meter = 100 cm
- 1 mile = 1.60934 km

**Temperature:**
- C = (F - 32) × 5/9
- F = (C × 9/5) + 32
- C = K - 273.15

**Volume:**
- 1 gallon (US) = 3.78541 liters
- 1 liter = 1000 ml
- 1 cup = 236.588 ml

**Handling Currency:**

Currency is special because:
1. Exchange rates change constantly
2. Historical rates needed for old data
3. Different rates for different dates

\`\`\`python
# Option 1: Use fixed rate (document clearly)
eur_to_usd = 1.08  # Rate as of 2024-01-08

# Option 2: Use date-specific rates
def convert_currency(amount, from_curr, to_curr, date):
    rate = get_exchange_rate(from_curr, to_curr, date)
    return amount * rate

# Option 3: Keep original + converted
df['price_original'] = df['price']
df['price_original_currency'] = df['currency']
df['price_usd'] = df.apply(convert_to_usd, axis=1)
\`\`\`

**Testing Conversions:**

\`\`\`python
# Test known conversions
assert abs(lbs_to_kg(1) - 0.453592) < 0.0001
assert abs(fahrenheit_to_celsius(32) - 0) < 0.0001
assert abs(inches_to_cm(1) - 2.54) < 0.0001

# Test round-trip conversions
weight_lbs = 10
weight_kg = lbs_to_kg(weight_lbs)
weight_lbs_back = kg_to_lbs(weight_kg)
assert abs(weight_lbs - weight_lbs_back) < 0.0001
\`\`\`

**After Standardization - Benefits:**

✓ Can calculate averages, sums, statistics
✓ Can sort properly
✓ Can compare values directly
✓ Visualizations are meaningful
✓ ML models can train properly

**Real-World Impact:**

\`\`\`
BEFORE standardization:
Average weight: ERROR (can't mix kg and lbs)
Total weight: MEANINGLESS (5 kg + 10 lbs = ???)
Heaviest item: WRONG (sorting strings, not numbers)

AFTER standardization:
Average weight: 1.52 kg ✓
Total weight: 12.14 kg ✓
Heaviest item: 4.2 kg (Monitor) ✓
\`\`\``
  };

  return <ProblemTemplate data={data} problemNumber={13} />;
};

export default MixedUnits;