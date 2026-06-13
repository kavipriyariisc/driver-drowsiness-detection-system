import zipfile
from pathlib import Path

# Extract
with zipfile.ZipFile('src.zip', 'r') as z:
    z.extractall('temp_check')

# Read file
fpath = Path('temp_check/src/models/m6_train.py')
with open(fpath, encoding='utf-8', errors='replace') as f:
    content = f.read()

# Search for the problematic raise
if 'raise FileNotFoundError' in content and 'Missing embedding sessions' in content:
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'raise FileNotFoundError' in line:
            print(f"Found raise FileNotFoundError at line {i+1}")
            for j in range(max(0, i-3), min(len(lines), i+6)):
                marker = ">>>" if j == i else "   "
                print(f"{marker} {j+1}: {lines[j]}")
            break
else:
    print("No problematic 'raise FileNotFoundError' found for missing embeddings")
    print("Checking for any raise statements...")
    for i, line in enumerate(lines):
        if 'raise FileNotFoundError' in line:
            print(f"Line {i+1}: {line}")
