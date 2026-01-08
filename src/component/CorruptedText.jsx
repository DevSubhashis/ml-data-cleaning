import React from 'react';
import ProblemTemplate from './ProblemTemplate';

const CorruptedText = () => {
  const data = {
    title: 'Corrupted Text',
    description: 'Corrupted text occurs due to encoding issues, character set mismatches, data transmission errors, or file corruption. It appears as garbled characters, mojibake, question marks, or replacement characters. This makes text unreadable and unusable for analysis.',
    originalData: [
      { user_id: 'U001', name: 'John Smith', comment: 'Great product!', email: 'john@example.com', city: 'New York' },
      { user_id: 'U002', name: 'MarÃ­a GarcÃ­a', comment: 'ExcelÃªncia!', email: 'maria@example.com', city: 'SÃ£o Paulo' },
      { user_id: 'U003', name: 'Müller Hans', comment: 'Sehr gut! â˜…â˜…â˜…', email: 'hans@example.com', city: 'MÃ¼nchen' },
      { user_id: 'U004', name: 'æ�Žæ˜Žï¼ˆLi Ming)', comment: 'å¾ˆå¥½ï¼�', email: 'liming@example.com', city: 'åŒ—äº¬' },
      { user_id: 'U005', name: 'Ã‰milie Dubois', comment: 'TrÃ¨s bien!', email: 'emilie@example.com', city: 'Paris' },
      { user_id: 'U006', name: 'Alice ???', comment: 'Good but ??? issues', email: 'alice@example.com', city: 'London' },
      { user_id: 'U007', name: 'José Silva', comment: '¡Perfecto! ♥', email: 'jose@example.com', city: 'Madrid' },
      { user_id: 'U008', name: '\\ufffd\\ufffd\\ufffd', comment: 'Nice \\ufffd', email: 'user@example.com', city: '\\ufffd\\ufffd' },
    ],
    cleanedData: [
      { user_id: 'U001', name: 'John Smith', comment: 'Great product!', email: 'john@example.com', city: 'New York' },
      { user_id: 'U002', name: 'María García', comment: 'Excelência!', email: 'maria@example.com', city: 'São Paulo' },
      { user_id: 'U003', name: 'Müller Hans', comment: 'Sehr gut! ★★★', email: 'hans@example.com', city: 'München' },
      { user_id: 'U004', name: '李明(Li Ming)', comment: '很好！', email: 'liming@example.com', city: '北京' },
      { user_id: 'U005', name: 'Émilie Dubois', comment: 'Très bien!', email: 'emilie@example.com', city: 'Paris' },
      { user_id: 'U006', name: 'Alice [?]', comment: 'Good but [corrupted] issues', email: 'alice@example.com', city: 'London' },
      { user_id: 'U007', name: 'José Silva', comment: '¡Perfecto! ♥', email: 'jose@example.com', city: 'Madrid' },
      { user_id: 'U008', name: '[CORRUPTED]', comment: 'Nice [?]', email: 'user@example.com', city: '[CORRUPTED]' },
    ],
    removedColumns: [],
    testDataset: `import pandas as pd
import numpy as np

# Create test dataset with corrupted text (encoding issues)
data = {
    'user_id': ['U001', 'U002', 'U003', 'U004', 'U005', 'U006', 'U007', 'U008'],
    'name': [
        'John Smith',
        'MarÃ­a GarcÃ­a',  # UTF-8 decoded as Latin-1
        'Müller Hans',
        'æ�Žæ˜Žï¼ˆLi Ming)',  # Chinese characters corrupted
        'Ã‰milie Dubois',  # Accented characters corrupted
        'Alice ???',  # Question marks from unknown chars
        'José Silva',
        '\\ufffd\\ufffd\\ufffd'  # Replacement characters
    ],
    'comment': [
        'Great product!',
        'ExcelÃªncia!',
        'Sehr gut! â˜…â˜…â˜…',  # Stars corrupted
        'å¾ˆå¥½ï¼�',  # Chinese corrupted
        'TrÃ¨s bien!',
        'Good but ??? issues',
        '¡Perfecto! ♥',
        'Nice \\ufffd'
    ],
    'email': [
        'john@example.com',
        'maria@example.com',
        'hans@example.com',
        'liming@example.com',
        'emilie@example.com',
        'alice@example.com',
        'jose@example.com',
        'user@example.com'
    ],
    'city': [
        'New York',
        'SÃ£o Paulo',
        'MÃ¼nchen',
        'åŒ—äº¬',  # Beijing
        'Paris',
        'London',
        'Madrid',
        '\\ufffd\\ufffd'
    ]
}

df = pd.DataFrame(data)
print("Original Dataset (with corrupted text):")
print(df)
print(f"\\nDataset Shape: {df.shape}")

print("\\n⚠ PROBLEM: Text corruption visible in multiple columns")
print("  - Mojibake: 'MarÃ­a' instead of 'María'")
print("  - Replacement chars: '\\ufffd' or '???'")
print("  - Garbled characters: Accents, special chars corrupted")`,
    solution: `# Solution: Detect and handle corrupted text

def detect_corrupted_text(df, columns=None):
    """
    Detect corrupted text in string columns
    
    Parameters:
    df: pandas DataFrame
    columns: list of columns to check (None = all object columns)
    
    Returns:
    corruption_report: Dictionary with corruption detection results
    """
    import pandas as pd
    import re
    
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    
    corruption_report = {}
    
    print("="*70)
    print("CORRUPTED TEXT DETECTION")
    print("="*70)
    
    # Common corruption patterns
    patterns = {
        'replacement_char': r'\\ufffd|�',  # Unicode replacement character
        'mojibake_latin': r'[ÃƒÂ][\\x80-\\xBF]',  # Common UTF-8 → Latin-1 issue
        'question_marks': r'\\?{2,}',  # Multiple question marks
        'mixed_encoding': r'[Ã€-Ã¿][a-zA-Z]',  # Typical encoding mismatch
        'box_characters': r'[\\u2400-\\u243F]',  # Control character display
    }
    
    for col in columns:
        if col not in df.columns:
            continue
        
        print(f"\\n{col}:")
        col_corrupted = False
        corruption_details = []
        
        for pattern_name, pattern in patterns.items():
            matches = df[col].astype(str).str.contains(pattern, regex=True, na=False)
            
            if matches.any():
                col_corrupted = True
                count = matches.sum()
                examples = df[matches][col].head(3).tolist()
                
                corruption_details.append({
                    'pattern': pattern_name,
                    'count': count,
                    'examples': examples
                })
                
                print(f"  ✗ {pattern_name}: {count} values")
                print(f"    Examples: {examples[:2]}")
        
        if col_corrupted:
            corruption_report[col] = corruption_details
        else:
            print(f"  ✓ No corruption detected")
    
    return corruption_report


def fix_common_encoding_issues(df, columns=None):
    """
    Attempt to fix common encoding issues
    
    Parameters:
    df: pandas DataFrame
    columns: list of columns to fix (None = all object columns)
    
    Returns:
    cleaned_df: DataFrame with attempted fixes
    fix_report: Dictionary with fix details
    """
    import pandas as pd
    
    df_cleaned = df.copy()
    fix_report = {}
    
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    
    print("\\n" + "="*70)
    print("ENCODING FIX ATTEMPTS")
    print("="*70)
    
    for col in columns:
        if col not in df.columns:
            continue
        
        print(f"\\n{col}:")
        fixes_applied = []
        
        for idx, value in df[col].items():
            if pd.isna(value):
                continue
            
            original = str(value)
            fixed = original
            
            # Try to detect and fix UTF-8 misinterpreted as Latin-1
            try:
                # Check if it looks like mojibake
                if any(char in original for char in ['Ã', 'Â', 'â']):
                    # Try encoding as Latin-1 and decoding as UTF-8
                    fixed = original.encode('latin-1').decode('utf-8', errors='ignore')
                    if fixed != original and fixed.isprintable():
                        df_cleaned.loc[idx, col] = fixed
                        fixes_applied.append({
                            'row': idx,
                            'before': original,
                            'after': fixed
                        })
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass
            
            # Replace replacement characters
            if '\\ufffd' in original or '�' in original:
                fixed = original.replace('\\ufffd', '[?]').replace('�', '[?]')
                df_cleaned.loc[idx, col] = fixed
                fixes_applied.append({
                    'row': idx,
                    'before': original,
                    'after': fixed
                })
        
        if fixes_applied:
            print(f"  ✓ Fixed {len(fixes_applied)} values")
            for fix in fixes_applied[:3]:  # Show first 3
                print(f"    Row {fix['row']}: '{fix['before']}' → '{fix['after']}'")
            fix_report[col] = fixes_applied
        else:
            print(f"  No fixes applied")
    
    return df_cleaned, fix_report


def flag_unrecoverable_corruption(df, columns=None, flag_pattern=r'\\?{3,}|\\ufffd{2,}'):
    """
    Flag rows with unrecoverable corruption
    
    Parameters:
    df: pandas DataFrame
    columns: list of columns to check
    flag_pattern: regex pattern for severe corruption
    
    Returns:
    df with corruption flag column
    """
    import pandas as pd
    
    df_flagged = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    
    print("\\n" + "="*70)
    print("FLAGGING UNRECOVERABLE CORRUPTION")
    print("="*70)
    
    # Check for severe corruption
    severe_corruption = pd.Series([False] * len(df), index=df.index)
    
    for col in columns:
        if col in df.columns:
            matches = df[col].astype(str).str.contains(flag_pattern, regex=True, na=False)
            severe_corruption |= matches
    
    df_flagged['has_corruption'] = severe_corruption
    
    corrupted_count = severe_corruption.sum()
    print(f"\\nRows with severe corruption: {corrupted_count}")
    
    if corrupted_count > 0:
        print(f"Corrupted row indices: {df[severe_corruption].index.tolist()}")
        print(f"\\nCorrupted rows:")
        print(df[severe_corruption])
    
    return df_flagged


# Step 1: Detect corruption
corruption_report = detect_corrupted_text(df, columns=['name', 'comment', 'city'])

# Step 2: Attempt to fix common issues
cleaned_df, fix_report = fix_common_encoding_issues(df, columns=['name', 'comment', 'city'])

# Step 3: Flag unrecoverable corruption
cleaned_df = flag_unrecoverable_corruption(cleaned_df, columns=['name', 'comment', 'city'])

print("\\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Columns with corruption detected: {len(corruption_report)}")
print(f"Columns with fixes applied: {len(fix_report)}")
print(f"Rows with unrecoverable corruption: {cleaned_df['has_corruption'].sum()}")

print("\\n\\nCleaned Dataset:")
print(cleaned_df)

print("\\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)
print("\\n1. For rows with 'has_corruption' = True:")
print("   - Option A: Remove these rows")
print("   - Option B: Request original data")
print("   - Option C: Manual review and correction")
print("\\n2. Prevention:")
print("   - Always use UTF-8 encoding when reading/writing")
print("   - Specify encoding explicitly: pd.read_csv('file.csv', encoding='utf-8')")
print("   - Validate encoding on data import")`,
    explanation: `**What is Corrupted Text?**

Corrupted text results from encoding/decoding mismatches, file corruption, or data transmission errors. It appears as:
- **Mojibake**: "MarÃ­a" instead of "María"
- **Replacement characters**: � or \\ufffd
- **Question marks**: ??? for unknown characters
- **Garbled text**: Unreadable character sequences

**Common Causes:**

1. **Encoding Mismatches** (Most Common):
   - UTF-8 text read as Latin-1
   - Latin-1 text read as UTF-8
   - Windows-1252 vs UTF-8
   - ASCII assumptions with Unicode data

2. **Database Issues**:
   - Wrong charset in database
   - Double-encoding (UTF-8 → Latin-1 → UTF-8)
   - Charset conversion during migration

3. **File Operations**:
   - Copy-paste from different systems
   - Excel CSV exports (often wrong encoding)
   - Text editors with wrong encoding settings

4. **Data Transmission**:
   - API responses without proper encoding header
   - Email attachments
   - FTP transfers in wrong mode

5. **Character Set Limitations**:
   - ASCII-only systems receiving Unicode
   - 7-bit systems receiving 8-bit data

**Types of Corruption:**

**1. UTF-8 Misread as Latin-1:**
\`\`\`
Original: María
Corrupted: MarÃ­a

Original: São Paulo  
Corrupted: SÃ£o Paulo

Original: Très bien
Corrupted: TrÃ¨s bien
\`\`\`

**2. Replacement Characters:**
\`\`\`
Original: 北京 (Beijing)
Corrupted: \\ufffd\\ufffd or ��

Original: ♥ (heart symbol)
Corrupted: � or ?
\`\`\`

**3. Double Encoding:**
\`\`\`
Original: María
After first encode/decode: MarÃ­a
After second encode/decode: MarÃƒÂ­a
\`\`\`

**Detection Strategies:**

**1. Visual Patterns:**
\`\`\`python
# Common corruption indicators
patterns = {
    'mojibake': r'[ÃƒÂ][\\x80-\\xBF]',
    'replacement': r'\\ufffd|�',
    'question_marks': r'\\?{2,}',
    'non_printable': r'[\\x00-\\x1F\\x7F-\\x9F]'
}

for pattern_name, pattern in patterns.items():
    matches = df['text'].str.contains(pattern, regex=True, na=False)
    if matches.any():
        print(f"Found {pattern_name}: {matches.sum()} instances")
\`\`\`

**2. Statistical Analysis:**
\`\`\`python
# Check for unusual character distributions
def has_corruption(text):
    if pd.isna(text):
        return False
    # Check for high ratio of non-ASCII characters
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / len(text) > 0.5 if len(text) > 0 else False

df['possibly_corrupted'] = df['text'].apply(has_corruption)
\`\`\`

**3. Encoding Detection:**
\`\`\`python
import chardet

def detect_encoding(text):
    if isinstance(text, str):
        text = text.encode('utf-8')
    result = chardet.detect(text)
    return result['encoding'], result['confidence']

# Check encoding of sample
encoding, confidence = detect_encoding(df['text'].iloc[0])
print(f"Detected: {encoding} (confidence: {confidence})")
\`\`\`

**Fixing Strategies:**

**1. Re-encode/Decode (for Mojibake):**

\`\`\`python
def fix_mojibake(text):
    """Fix UTF-8 misread as Latin-1"""
    try:
        # Encode as Latin-1, decode as UTF-8
        return text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text

df['name_fixed'] = df['name'].apply(fix_mojibake)
\`\`\`

**2. Try Multiple Encodings:**

\`\`\`python
def try_fix_encoding(text):
    """Try common encoding fixes"""
    if pd.isna(text):
        return text
    
    encodings = [
        ('latin-1', 'utf-8'),
        ('cp1252', 'utf-8'),  # Windows encoding
        ('iso-8859-1', 'utf-8'),
    ]
    
    for from_enc, to_enc in encodings:
        try:
            fixed = text.encode(from_enc).decode(to_enc)
            # Check if fix looks reasonable
            if fixed.isprintable() and '�' not in fixed:
                return fixed
        except:
            continue
    
    return text  # Return original if no fix works
\`\`\`

**3. Use ftfy Library (Third-party):**

\`\`\`python
import ftfy

# Fixes common encoding issues automatically
df['text_fixed'] = df['text'].apply(lambda x: ftfy.fix_text(x) if pd.notna(x) else x)
\`\`\`

**4. Handle Replacement Characters:**

\`\`\`python
# Replace with placeholder
df['text'] = df['text'].str.replace('\\ufffd', '[?]')
df['text'] = df['text'].str.replace('�', '[?]')

# Or remove
df['text'] = df['text'].str.replace('[\\ufffd�]', '', regex=True)

# Or flag and exclude
df['has_replacement_char'] = df['text'].str.contains('\\ufffd|�', na=False)
\`\`\`

**Prevention Best Practices:**

**1. Always Specify Encoding:**

\`\`\`python
# Reading files
df = pd.read_csv('file.csv', encoding='utf-8')

# Writing files
df.to_csv('file.csv', encoding='utf-8', index=False)

# Try with error handling
try:
    df = pd.read_csv('file.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('file.csv', encoding='latin-1')
\`\`\`

**2. Use UTF-8 Everywhere:**
- Database: SET NAMES utf8mb4
- Files: Save as UTF-8
- APIs: Content-Type: text/html; charset=utf-8
- Python: Default in Python 3

**3. Validate on Import:**

\`\`\`python
def validate_text_encoding(df, text_columns):
    """Check for encoding issues"""
    issues = {}
    
    for col in text_columns:
        # Check for replacement characters
        has_replacement = df[col].str.contains('\\ufffd|�', na=False).sum()
        
        # Check for mojibake patterns
        has_mojibake = df[col].str.contains('[ÃƒÂ]', na=False).sum()
        
        if has_replacement > 0 or has_mojibake > 0:
            issues[col] = {
                'replacement_chars': has_replacement,
                'mojibake_patterns': has_mojibake
            }
    
    return issues
\`\`\`

**4. Set Python Defaults:**

\`\`\`python
import sys
import locale

# Set default encoding
if sys.version_info[0] >= 3:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
\`\`\`

**Handling Unrecoverable Corruption:**

When text cannot be fixed:

**Option 1: Flag and Keep**
\`\`\`python
df['text_corrupted'] = df['text'].str.contains('[\\ufffd�]|\\?{3,}', regex=True, na=False)
\`\`\`

**Option 2: Replace with Placeholder**
\`\`\`python
df.loc[df['text_corrupted'], 'text'] = '[CORRUPTED TEXT]'
\`\`\`

**Option 3: Remove Rows**
\`\`\`python
df = df[~df['text_corrupted']]
\`\`\`

**Option 4: Request Re-import**
- Contact data source
- Re-export with correct encoding

**Testing for Corruption:**

\`\`\`python
def test_encoding_roundtrip(text, encoding='utf-8'):
    """Test if encoding survives roundtrip"""
    try:
        encoded = text.encode(encoding)
        decoded = encoded.decode(encoding)
        return text == decoded
    except:
        return False

# Test sample
sample = "María García - São Paulo"
assert test_encoding_roundtrip(sample, 'utf-8')
\`\`\`

**Common Scenarios:**

**Excel CSV Export:**
\`\`\`python
# Excel often exports as cp1252 or latin-1
df = pd.read_csv('excel_export.csv', encoding='cp1252')
\`\`\`

**Web Scraping:**
\`\`\`python
# Detect from HTTP response
response = requests.get(url)
encoding = response.encoding or 'utf-8'
content = response.content.decode(encoding)
\`\`\`

**Database Export:**
\`\`\`python
# Specify connection encoding
from sqlalchemy import create_engine
engine = create_engine('mysql://...', 
                      connect_args={'charset': 'utf8mb4'})
\`\`\``
  };

  return <ProblemTemplate data={data} problemNumber={16} />;
};

export default CorruptedText;