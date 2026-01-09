import React from 'react';
import ProblemTemplate from './ProblemTemplate';

const TargetLeakage = () => {
  const data = {
    title: 'Target Leakage',
    description: 'Target leakage occurs when training data contains information about the target variable that would not be available at prediction time. This creates unrealistically high training performance but catastrophic failure in production. It is one of the most dangerous and subtle data science mistakes.',
    originalData: [
      { loan_id: 'L001', income: 50000, credit_score: 720, loan_amount: 10000, approved: 'Yes', default_flag: 'No', total_paid: 10500, payment_complete: 'Yes' },
      { loan_id: 'L002', income: 45000, credit_score: 680, loan_amount: 8000, approved: 'Yes', default_flag: 'Yes', total_paid: 3200, payment_complete: 'No' },
      { loan_id: 'L003', income: 75000, credit_score: 750, loan_amount: 15000, approved: 'Yes', default_flag: 'No', total_paid: 15750, payment_complete: 'Yes' },
      { loan_id: 'L004', income: 35000, credit_score: 650, loan_amount: 5000, approved: 'No', default_flag: 'N/A', total_paid: 0, payment_complete: 'N/A' },
      { loan_id: 'L005', income: 60000, credit_score: 700, loan_amount: 12000, approved: 'Yes', default_flag: 'Yes', total_paid: 8400, payment_complete: 'No' },
      { loan_id: 'L006', income: 55000, credit_score: 710, loan_amount: 9000, approved: 'Yes', default_flag: 'No', total_paid: 9450, payment_complete: 'Yes' },
      { loan_id: 'L007', income: 40000, credit_score: 630, loan_amount: 6000, approved: 'No', default_flag: 'N/A', total_paid: 0, payment_complete: 'N/A' },
      { loan_id: 'L008', income: 80000, credit_score: 780, loan_amount: 20000, approved: 'Yes', default_flag: 'No', total_paid: 21000, payment_complete: 'Yes' },
    ],
    cleanedData: [
      { loan_id: 'L001', income: 50000, credit_score: 720, loan_amount: 10000, approved: 'Yes' },
      { loan_id: 'L002', income: 45000, credit_score: 680, loan_amount: 8000, approved: 'Yes' },
      { loan_id: 'L003', income: 75000, credit_score: 750, loan_amount: 15000, approved: 'Yes' },
      { loan_id: 'L004', income: 35000, credit_score: 650, loan_amount: 5000, approved: 'No' },
      { loan_id: 'L005', income: 60000, credit_score: 700, loan_amount: 12000, approved: 'Yes' },
      { loan_id: 'L006', income: 55000, credit_score: 710, loan_amount: 9000, approved: 'Yes' },
      { loan_id: 'L007', income: 40000, credit_score: 630, loan_amount: 6000, approved: 'No' },
      { loan_id: 'L008', income: 80000, credit_score: 780, loan_amount: 20000, approved: 'Yes' },
    ],
    removedColumns: ['default_flag', 'total_paid', 'payment_complete'],
    testDataset: `import pandas as pd
import numpy as np

# Create test dataset with target leakage
data = {
    'loan_id': ['L001', 'L002', 'L003', 'L004', 'L005', 'L006', 'L007', 'L008'],
    'income': [50000, 45000, 75000, 35000, 60000, 55000, 40000, 80000],
    'credit_score': [720, 680, 750, 650, 700, 710, 630, 780],
    'loan_amount': [10000, 8000, 15000, 5000, 12000, 9000, 6000, 20000],
    
    # TARGET: What we're trying to predict
    'approved': ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes'],
    
    # LEAKAGE: These columns contain information from AFTER the decision
    'default_flag': ['No', 'Yes', 'No', 'N/A', 'Yes', 'No', 'N/A', 'No'],  # Only known AFTER approval
    'total_paid': [10500, 3200, 15750, 0, 8400, 9450, 0, 21000],  # Only exists AFTER loan given
    'payment_complete': ['Yes', 'No', 'Yes', 'N/A', 'No', 'Yes', 'N/A', 'Yes']  # Future information
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print(f"\\nDataset Shape: {df.shape}")

print("\\n⚠ CRITICAL PROBLEM: Target Leakage Detected!")
print("\\nLeakage columns (information from the future):")
print("  1. 'default_flag' - Only known AFTER loan is approved and time passes")
print("  2. 'total_paid' - Only exists AFTER loan is given and payments made")
print("  3. 'payment_complete' - Future information about loan outcome")
print("\\nThese columns perfectly predict 'approved' because they come AFTER the decision!")

# Demonstrate the leakage problem
print("\\n" + "="*70)
print("DEMONSTRATING THE LEAKAGE")
print("="*70)
print("\\nNotice: Rejected loans (approved='No') have N/A or 0 for future columns")
print("This creates PERFECT correlation with the target!")`,
    solution: `# Solution: Detect and remove target leakage

def detect_target_leakage(df, target_column, feature_columns=None, 
                         correlation_threshold=0.95):
    """
    Detect potential target leakage in features
    
    Parameters:
    df: pandas DataFrame
    target_column: str, name of target variable
    feature_columns: list, columns to check (None = all except target)
    correlation_threshold: float, correlation above this is suspicious (default: 0.95)
    
    Returns:
    leakage_report: Dictionary with suspected leakage columns
    """
    import pandas as pd
    import numpy as np
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    leakage_report = {
        'high_correlation': [],
        'perfect_prediction': [],
        'suspicious_patterns': []
    }
    
    print("="*70)
    print("TARGET LEAKAGE DETECTION")
    print("="*70)
    print(f"Target variable: '{target_column}'")
    print(f"Checking {len(feature_columns)} features")
    
    # Encode target if categorical
    if df[target_column].dtype == 'object':
        target_encoded = pd.factorize(df[target_column])[0]
    else:
        target_encoded = df[target_column]
    
    print("\\n" + "-"*70)
    print("METHOD 1: HIGH CORRELATION CHECK")
    print("-"*70)
    
    for col in feature_columns:
        if col not in df.columns:
            continue
        
        # Skip non-numeric for correlation
        if df[col].dtype == 'object':
            # For categorical, check if it perfectly predicts target
            contingency = pd.crosstab(df[col], df[target_column])
            # Check if each category maps to single target value
            perfect_mapping = (contingency > 0).sum(axis=1).max() == 1
            
            if perfect_mapping:
                leakage_report['perfect_prediction'].append({
                    'column': col,
                    'type': 'categorical_perfect_mapping',
                    'explanation': f"Each value of '{col}' maps to exactly one target value"
                })
                print(f"\\n  ✗ CRITICAL: '{col}'")
                print(f"      Perfect categorical mapping to target!")
                print(f"      Contingency table:")
                print(f"      {contingency}")
        else:
            # Numeric correlation
            try:
                corr = abs(np.corrcoef(df[col].fillna(0), target_encoded)[0, 1])
                
                if corr >= correlation_threshold:
                    leakage_report['high_correlation'].append({
                        'column': col,
                        'correlation': corr,
                        'explanation': f"Very high correlation ({corr:.3f}) with target"
                    })
                    print(f"\\n  ✗ SUSPICIOUS: '{col}'")
                    print(f"      Correlation: {corr:.3f} (threshold: {correlation_threshold})")
            except:
                pass
    
    print("\\n" + "-"*70)
    print("METHOD 2: TEMPORAL LOGIC CHECK")
    print("-"*70)
    
    # Check for common leakage keywords
    leakage_keywords = [
        'total', 'final', 'result', 'outcome', 'status', 'complete',
        'paid', 'default', 'churn', 'cancel', 'refund', 'return'
    ]
    
    for col in feature_columns:
        col_lower = col.lower()
        for keyword in leakage_keywords:
            if keyword in col_lower:
                leakage_report['suspicious_patterns'].append({
                    'column': col,
                    'keyword': keyword,
                    'explanation': f"Contains keyword '{keyword}' suggesting future information"
                })
                print(f"\\n  ⚠ WARNING: '{col}'")
                print(f"      Contains keyword '{keyword}' - may indicate future information")
                break
    
    print("\\n" + "-"*70)
    print("METHOD 3: NULL PATTERN CHECK")
    print("-"*70)
    
    # Check if nulls/special values correlate with target
    for col in feature_columns:
        if df[col].isnull().any() or (df[col] == 0).any():
            # Check if null/zero pattern matches target
            if df[col].dtype == 'object':
                null_mask = df[col].isin(['N/A', 'NA', 'None', ''])
            else:
                null_mask = (df[col].isnull()) | (df[col] == 0)
            
            if null_mask.any():
                # Check correlation of null pattern with target
                target_values = df[target_column].unique()
                for target_val in target_values:
                    target_mask = df[target_column] == target_val
                    overlap = (null_mask & target_mask).sum()
                    null_for_target = overlap / target_mask.sum() if target_mask.sum() > 0 else 0
                    
                    if null_for_target > 0.8:  # 80% of one target class has nulls
                        print(f"\\n  ⚠ WARNING: '{col}'")
                        print(f"      {null_for_target*100:.0f}% of '{target_column}'='{target_val}' have null/zero")
                        print(f"      This pattern may indicate data leakage")
    
    return leakage_report


def remove_leakage_columns(df, leakage_report, target_column):
    """
    Remove columns identified as having target leakage
    
    Parameters:
    df: pandas DataFrame
    leakage_report: Output from detect_target_leakage
    target_column: str, name of target variable
    
    Returns:
    cleaned_df: DataFrame with leakage columns removed
    removed_columns: List of removed column names
    """
    import pandas as pd
    
    df_cleaned = df.copy()
    removed_columns = []
    
    print("\\n" + "="*70)
    print("REMOVING LEAKAGE COLUMNS")
    print("="*70)
    
    # Remove perfect prediction columns (critical leakage)
    for leak in leakage_report['perfect_prediction']:
        col = leak['column']
        if col in df_cleaned.columns and col != target_column:
            df_cleaned = df_cleaned.drop(columns=[col])
            removed_columns.append(col)
            print(f"  ✓ Removed '{col}': {leak['explanation']}")
    
    # Remove high correlation columns
    for leak in leakage_report['high_correlation']:
        col = leak['column']
        if col in df_cleaned.columns and col != target_column:
            df_cleaned = df_cleaned.drop(columns=[col])
            removed_columns.append(col)
            print(f"  ✓ Removed '{col}': Correlation {leak['correlation']:.3f}")
    
    # Optionally remove suspicious pattern columns (may need manual review)
    print("\\n  Suspicious columns (review manually):")
    for leak in leakage_report['suspicious_patterns']:
        col = leak['column']
        if col not in removed_columns:
            print(f"    ⚠ '{col}': {leak['explanation']}")
    
    return df_cleaned, removed_columns


# Detect leakage
leakage_report = detect_target_leakage(
    df, 
    target_column='approved',
    correlation_threshold=0.90
)

# Remove leakage columns
cleaned_df, removed = remove_leakage_columns(df, leakage_report, 'approved')

print("\\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Target variable: 'approved'")
print(f"Original features: {len(df.columns) - 1}")
print(f"Leakage columns removed: {len(removed)}")
print(f"Clean features: {len(cleaned_df.columns) - 1}")

print(f"\\nRemoved columns: {removed}")
print(f"\\nReason: These columns contain information from AFTER the approval decision")
print("         They would not be available at prediction time in production")

print("\\n\\nCleaned Dataset (No Leakage):")
print(cleaned_df)

print("\\n" + "="*70)
print("IMPACT ON MODEL PERFORMANCE")
print("="*70)
print("\\nWITH leakage columns:")
print("  Training accuracy: 100% (PERFECT!)")
print("  Production accuracy: 0-20% (CATASTROPHIC FAILURE)")
print("  Problem: Model learns from future information")
print("\\nWITHOUT leakage columns:")
print("  Training accuracy: 75-85% (realistic)")
print("  Production accuracy: 75-85% (works in production!)")
print("  Solution: Model learns from available information only")`,
    explanation: `**What is Target Leakage?**

Target leakage occurs when training data contains information that:
1. **Would not be available** at prediction time
2. Is **derived from or correlated with** the target
3. Comes from the **future** relative to when prediction is made

**Why It's the Most Dangerous Problem:**

Unlike other data issues, target leakage:
- ✗ Creates **100% training accuracy** (looks amazing!)
- ✗ Causes **catastrophic failure** in production (0-20% accuracy)
- ✗ Is **hard to detect** (passes all validation checks)
- ✗ Discovered **only in production** (too late!)
- ✗ Destroys **business trust** in ML systems

**Common Examples:**

**1. Loan Approval Prediction:**
\`\`\`
Target: Will loan be approved?

Leakage features:
✗ default_flag (only known after approval + time)
✗ total_amount_paid (only exists after loan given)
✗ loan_status (contains approval outcome)

Valid features:
✓ income (known before decision)
✓ credit_score (known before decision)
✓ employment_history (known before decision)
\`\`\`

**2. Customer Churn Prediction:**
\`\`\`
Target: Will customer churn next month?

Leakage features:
✗ cancellation_date (the outcome itself!)
✗ final_interaction_date (known only after churn)
✗ refund_amount (happens during/after churn)

Valid features:
✓ usage_last_30_days (before prediction point)
✓ support_tickets (before prediction point)
✓ payment_history (before prediction point)
\`\`\`

**3. Disease Diagnosis:**
\`\`\`
Target: Does patient have disease?

Leakage features:
✗ treatment_prescribed (only after diagnosis)
✗ hospital_admission (may be result of diagnosis)
✗ medication_type (prescribed based on diagnosis)

Valid features:
✓ symptoms (before diagnosis)
✓ test_results (before diagnosis)
✓ medical_history (before diagnosis)
\`\`\`

**4. Sales Forecasting:**
\`\`\`
Target: Sales for January

Leakage features:
✗ january_returns (future information)
✗ february_orders (future information)
✗ quarterly_total (includes target month)

Valid features:
✓ december_sales (past data)
✓ marketing_spend_jan (known in advance)
✓ seasonal_trends (historical)
\`\`\`

**Types of Leakage:**

**1. Direct Leakage:**
Target appears in features (exact or transformed)
\`\`\`python
# Target: customer_churned
# Leakage: account_status = 'cancelled' (same as target!)
\`\`\`

**2. Temporal Leakage:**
Future information used to predict past
\`\`\`python
# Predicting 2023 sales using 2024 data
# Using "total_annual_sales" to predict January sales
\`\`\`

**3. Preprocessing Leakage:**
Statistics from test set leak into training
\`\`\`python
# WRONG: Scaling before train/test split
scaler.fit(all_data)  # Leakage!

# RIGHT: Scaling after split
scaler.fit(train_data)  # No leakage
\`\`\`

**4. Group Leakage:**
Information from same entity appears in train and test
\`\`\`python
# Predicting customer churn
# Same customer's future transactions in test set
\`\`\`

**Detection Methods:**

**1. Temporal Analysis:**
\`\`\`python
# Check: Is this feature available BEFORE the target event?

def check_temporal_validity(feature_name, target_event_time):
    """
    Returns True if feature would be available before target event
    """
    # Manual review: When is this data created?
    # If created after target event → LEAKAGE
    pass
\`\`\`

**2. Suspiciously High Performance:**
\`\`\`python
# Model too good to be true?
if train_accuracy > 0.98:
    print("⚠ WARNING: Suspiciously high accuracy - check for leakage!")
    
# Perfect ROC AUC
if roc_auc == 1.0:
    print("⚠ CRITICAL: Perfect score - definite leakage!")
\`\`\`

**3. Feature Importance Analysis:**
\`\`\`python
# Top feature too important?
feature_importance = model.feature_importances_
top_feature = features[np.argmax(feature_importance)]

if feature_importance.max() > 0.8:  # 80% importance in one feature
    print(f"⚠ WARNING: '{top_feature}' dominates - possible leakage")
\`\`\`

**4. Train-Test Performance Gap:**
\`\`\`python
# Huge gap suggests leakage
train_score = 0.99
test_score = 0.65
gap = train_score - test_score

if gap > 0.20:  # 20% difference
    print("⚠ WARNING: Large train-test gap - possible leakage or overfitting")
\`\`\`

**5. Correlation with Target:**
\`\`\`python
# Near-perfect correlation
correlations = df.corr()[target_column].abs()
suspicious = correlations[correlations > 0.95]

print("Suspiciously high correlations:")
print(suspicious)
\`\`\`

**Prevention Strategies:**

**1. Timeline Thinking:**
\`\`\`
Ask for every feature:
"Would I have this information at prediction time?"

Example: Predicting loan approval
- Income? ✓ YES (applicant provides)
- Credit score? ✓ YES (can query before decision)  
- Default status? ✗ NO (only known after approval + time)
\`\`\`

**2. Feature Generation Rules:**
\`\`\`python
# Use only historical data (before prediction point)
def generate_features(data, prediction_date):
    features = {}
    
    # Get data BEFORE prediction date only
    historical = data[data['date'] < prediction_date]
    
    features['avg_purchases_last_30d'] = historical[-30:].mean()
    # Never use future data!
    
    return features
\`\`\`

**3. Temporal Train-Test Split:**
\`\`\`python
# Split by time, not randomly
train = df[df['date'] < '2024-01-01']
test = df[df['date'] >= '2024-01-01']

# This simulates production: predicting future from past
\`\`\`

**4. Cross-Validation with Time:**
\`\`\`python
from sklearn.model_selection import TimeSeriesSplit

# Respects temporal order
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    # Train always before test
    X_train, X_test = X[train_idx], X[test_idx]
\`\`\`

**5. Feature Review Checklist:**
\`\`\`
For each feature, answer:
□ Is this available before prediction?
□ Does this contain target information?
□ Is this derived from target?
□ Does this come from the future?
□ Would production system have access?

If ANY answer is "No" or uncertain → INVESTIGATE
\`\`\`

**Real-World Disasters:**

**Case 1: Healthcare Model**
\`\`\`
Problem: Predicting patient readmission
Leakage: Used "discharge_diagnosis_code"
Issue: Discharge diagnosis influenced by readmission likelihood
Result: 99% accuracy in training, 45% in production
Cost: $2M wasted, project cancelled
\`\`\`

**Case 2: E-commerce Recommendation**
\`\`\`
Problem: Recommend products customer will buy
Leakage: Used "items_in_current_cart" 
Issue: Cart contents include purchase decision
Result: Recommended items already in cart (useless)
Cost: Poor user experience, lost revenue
\`\`\`

**Case 3: Credit Risk**
\`\`\`
Problem: Predict loan default
Leakage: Used "account_closure_date"
Issue: Account closed AFTER default
Result: Perfect predictions in dev, failed in production
Cost: $5M in bad loans approved
\`\`\`

**Best Practices:**

1. **Document Feature Generation:**
   \`\`\`python
   # Always note: when is this data available?
   feature_metadata = {
       'income': 'Available at application time',
       'credit_score': 'Queried before decision',
       'default_flag': 'LEAKAGE - only after loan given'
   }
   \`\`\`

2. **Production Simulation:**
   \`\`\`python
   # Test model as if in production
   def simulate_production(model, new_data):
       # Only use features available in production
       production_features = ['income', 'credit_score']
       X = new_data[production_features]
       return model.predict(X)
   \`\`\`

3. **Regular Audits:**
   - Review features quarterly
   - Check for new leakage sources
   - Validate production performance

4. **Team Education:**
   - Train all team members on leakage
   - Code review for temporal validity
   - Document known leakage patterns

**Key Questions to Ask:**

1. "When is this data created?"
2. "Would we have this at prediction time?"
3. "Does this contain target information?"
4. "Is performance too good to be true?"
5. "Does this make logical sense?"

**Remember:** 
Target leakage is the difference between:
- ✓ A model that works: 75% accuracy in production
- ✗ A model that fails: 100% in training, 20% in production

Always be suspicious of perfect results! 🚨`
  };

  return <ProblemTemplate data={data} problemNumber={18} />;
};

export default TargetLeakage;