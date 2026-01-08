import React from 'react';
import ProblemTemplate from './ProblemTemplate';

const LowVariance = () => {
  const data = {
    title: 'Low Variance',
    description: 'Numeric columns with very low variance have values that are nearly constant. These features provide little predictive power and can be removed to simplify models and reduce computational overhead.',
    originalData: [
      { sensor_id: 1, temperature: 25.001, humidity: 45.2, pressure: 1013.25, voltage: 3.3001, reading_count: 1250 },
      { sensor_id: 2, temperature: 25.002, humidity: 67.8, pressure: 1013.25, voltage: 3.3002, reading_count: 1340 },
      { sensor_id: 3, temperature: 25.001, humidity: 52.3, pressure: 1013.25, voltage: 3.3001, reading_count: 980 },
      { sensor_id: 4, temperature: 25.003, humidity: 71.5, pressure: 1013.25, voltage: 3.3003, reading_count: 1580 },
      { sensor_id: 5, temperature: 25.002, humidity: 38.9, pressure: 1013.25, voltage: 3.3002, reading_count: 2100 },
      { sensor_id: 6, temperature: 25.001, humidity: 61.4, pressure: 1013.25, voltage: 3.3001, reading_count: 1750 },
      { sensor_id: 7, temperature: 25.002, humidity: 55.7, pressure: 1013.25, voltage: 3.3002, reading_count: 1420 },
      { sensor_id: 8, temperature: 25.003, humidity: 48.6, pressure: 1013.25, voltage: 3.3003, reading_count: 1890 },
    ],
    cleanedData: [
      { sensor_id: 1, humidity: 45.2, reading_count: 1250 },
      { sensor_id: 2, humidity: 67.8, reading_count: 1340 },
      { sensor_id: 3, humidity: 52.3, reading_count: 980 },
      { sensor_id: 4, humidity: 71.5, reading_count: 1580 },
      { sensor_id: 5, humidity: 38.9, reading_count: 2100 },
      { sensor_id: 6, humidity: 61.4, reading_count: 1750 },
      { sensor_id: 7, humidity: 55.7, reading_count: 1420 },
      { sensor_id: 8, humidity: 48.6, reading_count: 1890 },
    ],
    removedColumns: ['temperature', 'pressure', 'voltage'],
    testDataset: `import pandas as pd
import numpy as np

# Create test dataset with low variance columns
np.random.seed(42)

data = {
    'sensor_id': range(1, 9),
    'temperature': [25.001, 25.002, 25.001, 25.003, 25.002, 25.001, 25.002, 25.003],  # Very low variance
    'humidity': np.random.uniform(35, 75, 8).round(1),  # Normal variance
    'pressure': [1013.25] * 8,  # Zero variance (constant)
    'voltage': [3.3001, 3.3002, 3.3001, 3.3003, 3.3002, 3.3001, 3.3002, 3.3003],  # Very low variance
    'reading_count': np.random.randint(900, 2200, 8)  # Normal variance
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print("\\nDataset Shape:", df.shape)
print("\\nVariance Analysis:")
print(df.var())`,
    solution: `# Solution: Identify and remove low variance columns

def remove_low_variance_columns(df, threshold=0.01):
    """
    Identify and remove numeric columns with variance below threshold
    
    Parameters:
    df: pandas DataFrame
    threshold: float, minimum variance threshold (default: 0.01)
               Columns with variance < threshold will be removed
    
    Returns:
    cleaned_df: DataFrame without low-variance columns
    removed_cols: Dictionary with removed columns and their stats
    """
    import pandas as pd
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    removed_cols = {}
    cols_to_remove = []
    
    print(f"\\nAnalyzing numeric columns with variance threshold: {threshold}")
    print("-" * 70)
    
    for col in numeric_cols:
        variance = df[col].var()
        std_dev = df[col].std()
        mean_val = df[col].mean()
        
        # Calculate coefficient of variation for relative measure
        cv = (std_dev / mean_val * 100) if mean_val != 0 else 0
        
        print(f"{col}:")
        print(f"  Mean: {mean_val:.6f}")
        print(f"  Std Dev: {std_dev:.6f}")
        print(f"  Variance: {variance:.6f}")
        print(f"  Coefficient of Variation: {cv:.4f}%")
        
        if variance < threshold:
            cols_to_remove.append(col)
            removed_cols[col] = {
                'variance': variance,
                'std_dev': std_dev,
                'mean': mean_val,
                'cv': cv
            }
            print(f"  ❌ REMOVED (variance {variance:.6f} < {threshold})")
        else:
            print(f"  ✓ KEPT (variance {variance:.6f} >= {threshold})")
        print()
    
    # Remove low-variance columns
    cleaned_df = df.drop(columns=cols_to_remove)
    
    return cleaned_df, removed_cols

# Apply the solution
cleaned_df, removed = remove_low_variance_columns(df, threshold=0.01)

print("\\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {cleaned_df.shape}")
print(f"\\nRemoved {len(removed)} column(s):")
for col, stats in removed.items():
    print(f"  - {col}:")
    print(f"      Variance: {stats['variance']:.6f}")
    print(f"      CV: {stats['cv']:.4f}%")

print("\\n\\nCleaned Dataset:")
print(cleaned_df)

# Visualization of variance comparison
print("\\n" + "="*70)
print("VARIANCE COMPARISON")
print("="*70)
print("\\nRemaining columns variance:")
print(cleaned_df.select_dtypes(include=[np.number]).var())`,
    explanation: `**Why Remove Low Variance Columns?**

1. **Minimal Information**: Features with nearly constant values don't help distinguish between different observations
2. **Computational Efficiency**: Reduces dimensionality without losing predictive power
3. **Model Stability**: Prevents numerical instability in algorithms that use inverse matrices or gradient descent
4. **Feature Selection**: Helps identify truly informative features

**Understanding Variance Metrics:**

**Variance (σ²):**
- Measures how spread out values are from the mean
- Variance = Average of squared differences from mean
- Zero variance = Constant column

**Standard Deviation (σ):**
- Square root of variance
- More interpretable (same units as original data)

**Coefficient of Variation (CV):**
- CV = (Std Dev / Mean) × 100%
- Relative measure of variance
- Useful for comparing columns with different scales
- CV < 1% often indicates low variance

**Choosing the Right Threshold:**

**Absolute Variance Threshold:**
- Depends on the scale of your data
- 0.01 works for normalized/standardized data
- For raw data: analyze variance distribution first

**Relative Thresholds (Better approach):**
- Use Coefficient of Variation < 1%
- Or use VarianceThreshold with percentile-based cutoff
- Example: Remove bottom 5% of variance distribution

**When to Keep Low-Variance Columns:**
- Binary indicators (0/1) that are meaningful despite low variance
- Features with domain importance (e.g., calibration constants)
- When working with small datasets
- Columns that interact meaningfully with other features

**Best Practices:**

1. **Standardize First**: Apply variance threshold after standardization for fair comparison
2. **Document Thresholds**: Always log the threshold used
3. **Visualize**: Plot variance distribution to choose informed thresholds
4. **Domain Knowledge**: Consult experts before removing features

**Alternative Approaches:**
\`\`\`python
from sklearn.feature_selection import VarianceThreshold

# Remove features with < 80% of samples having same value
selector = VarianceThreshold(threshold=0.8 * (1 - 0.8))
X_high_variance = selector.fit_transform(X)
\`\`\``
  };

  return <ProblemTemplate data={data} problemNumber={3} />;
};

export default LowVariance;