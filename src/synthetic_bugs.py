"""
Synthetic Bug Injection Module

Creates buggy versions of clean code by injecting known bug patterns.
Useful for testing when real buggy data collection is incomplete.

Bug Patterns Injected:
1. Off-by-one errors in loops
2. Missing null checks
3. Wrong comparison operators
4. Swapped arguments
5. Missing return statements
"""

import ast
import random
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def inject_off_by_one(code: str) -> str:
    """Inject off-by-one error in range() calls."""
    # Change range(n) to range(n+1) or range(n-1)
    patterns = [
        (r'range\((\w+)\)', r'range(\1 + 1)'),
        (r'range\((\w+)\)', r'range(\1 - 1)'),
        (r'range\((\d+)\)', lambda m: f'range({int(m.group(1)) + 1})'),
    ]
    pattern, repl = random.choice(patterns[:2])
    return re.sub(pattern, repl, code, count=1)


def inject_wrong_operator(code: str) -> str:
    """Replace comparison operator with wrong one."""
    replacements = [
        ('<=', '<'),
        ('>=', '>'),
        ('<', '<='),
        ('>', '>='),
        ('==', '!='),
        ('!=', '=='),
        ('and', 'or'),
        ('or', 'and'),
    ]
    for old, new in replacements:
        if old in code:
            return code.replace(old, new, 1)
    return code


def inject_missing_return(code: str) -> str:
    """Remove a return statement."""
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if 'return ' in line and not line.strip().startswith('#'):
            # Comment out the return
            lines[i] = line.replace('return ', '# return ')
            break
    return '\n'.join(lines)


def inject_index_error(code: str) -> str:
    """Change array index access to potentially cause IndexError."""
    # Change arr[i] to arr[i+1] or arr[-1] to arr[-2]
    patterns = [
        (r'\[(\w+)\]', r'[\1 + 1]'),
        (r'\[0\]', '[1]'),
        (r'\[-1\]', '[-2]'),
    ]
    for pattern, repl in patterns:
        if re.search(pattern, code):
            return re.sub(pattern, repl, code, count=1)
    return code


def inject_type_error(code: str) -> str:
    """Inject operation that could cause type error."""
    # Add string concatenation with int
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if '=' in line and 'def ' not in line and '#' not in line:
            indent = len(line) - len(line.lstrip())
            lines.insert(i + 1, ' ' * indent + 'result = "text" + 1  # Bug: type error')
            break
    return '\n'.join(lines)


def inject_infinite_loop(code: str) -> str:
    """Modify while loop to potentially be infinite."""
    # Remove increment in while loop
    return re.sub(r'(\w+)\s*\+=\s*1', r'# \1 += 1  # Bug: missing increment', code, count=1)


def inject_random_bug(code: str) -> str:
    """Inject a random bug into the code."""
    injectors = [
        inject_off_by_one,
        inject_wrong_operator,
        inject_missing_return,
        inject_index_error,
    ]

    # Try injectors until one works
    random.shuffle(injectors)
    for injector in injectors:
        try:
            modified = injector(code)
            if modified != code:
                return modified
        except:
            continue

    # Fallback: add a simple bug
    lines = code.split('\n')
    if len(lines) > 2:
        # Find a line to modify
        for i in range(1, len(lines)):
            line = lines[i]
            if 'return' in line:
                # Modify return value
                lines[i] = line.replace('return ', 'return None  # ')
                break
    return '\n'.join(lines)


def create_synthetic_dataset(input_csv: Path, output_csv: Path,
                            num_buggy: int = 2500) -> pd.DataFrame:
    """
    Create synthetic buggy data from clean functions.

    Args:
        input_csv: Path to original functions.csv
        output_csv: Path to save augmented dataset
        num_buggy: Number of buggy samples to create

    Returns:
        Augmented DataFrame
    """
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Get clean functions
    clean_df = df[df['has_bug'] == 0].copy()
    print(f"Found {len(clean_df)} clean functions")

    # Sample functions to inject bugs
    num_to_inject = min(num_buggy, len(clean_df))
    samples = clean_df.sample(n=num_to_inject, random_state=42)

    print(f"Creating {num_to_inject} buggy versions...")
    buggy_functions = []

    for _, row in samples.iterrows():
        try:
            buggy_code = inject_random_bug(row['code'])

            buggy_functions.append({
                'code': buggy_code,
                'has_bug': 1,
                'repo': row['repo'] + '_synthetic',
                'commit_sha': 'synthetic',
                'file_path': row['file_path'],
                'function_name': row['function_name'],
                'stars': row['stars'],
                'lines_of_code': len(buggy_code.split('\n'))
            })
        except:
            continue

    # Create buggy DataFrame
    buggy_df = pd.DataFrame(buggy_functions)
    print(f"Created {len(buggy_df)} buggy functions")

    # Combine with original clean data
    # Take equal amounts of clean and buggy
    num_each = min(len(clean_df), len(buggy_df))
    clean_sample = clean_df.sample(n=num_each, random_state=42)

    combined_df = pd.concat([clean_sample, buggy_df.head(num_each)], ignore_index=True)

    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    combined_df.to_csv(output_csv, index=False)

    print(f"\nSaved augmented dataset to {output_csv}")
    print(f"Total: {len(combined_df)} functions")
    print(f"  Buggy: {combined_df['has_bug'].sum()}")
    print(f"  Clean: {(combined_df['has_bug'] == 0).sum()}")

    return combined_df


if __name__ == "__main__":
    # Create synthetic dataset
    input_path = Path("data/raw/functions.csv")
    output_path = Path("data/raw/functions_augmented.csv")

    if input_path.exists():
        df = create_synthetic_dataset(input_path, output_path)

        # Also create a backup and use augmented as main
        import shutil
        backup_path = Path("data/raw/functions_original.csv")
        shutil.copy(input_path, backup_path)
        shutil.copy(output_path, input_path)
        print(f"\nBackup saved to {backup_path}")
        print(f"Augmented data is now the main dataset")
    else:
        print(f"Input file not found: {input_path}")
