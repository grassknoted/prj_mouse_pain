#!/usr/bin/env python3
"""Debug script to inspect DLC CSV structure."""

import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Usage: python debug_dlc_csv.py <path_to_dlc_csv>")
    sys.exit(1)

csv_path = sys.argv[1]

print("="*80)
print(f"Inspecting DLC CSV: {csv_path}")
print("="*80)

# Try multi-level header
try:
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    print("\n✓ Successfully parsed with multi-level header (3 rows)")
    print(f"  Shape: {df.shape}")
    print(f"  Columns (first 5):")
    for i, col in enumerate(df.columns[:5]):
        print(f"    {i}: {col}")

    # Flatten column names
    df.columns = ['_'.join(str(c).strip() for c in col).lower() for col in df.columns.values]
    print(f"\n  Flattened columns (first 10):")
    for i, col in enumerate(df.columns[:10]):
        print(f"    {i}: {col}")

    # Search for keypoint patterns
    print("\n  Searching for keypoint patterns:")
    keypoints = ['mouth', 'l_frontpaw', 'r_frontpaw', 'l_hindpaw', 'r_hindpaw', 'tail_base']

    for kp in keypoints:
        x_cols = [c for c in df.columns if kp in c and 'x' in c and 'likelihood' not in c]
        y_cols = [c for c in df.columns if kp in c and 'y' in c and 'likelihood' not in c]
        like_cols = [c for c in df.columns if kp in c and 'likelihood' in c]

        print(f"\n    {kp}:")
        if x_cols:
            print(f"      X: {x_cols[0]}")
        if y_cols:
            print(f"      Y: {y_cols[0]}")
        if like_cols:
            print(f"      Likelihood: {like_cols[0]}")
        if not (x_cols and y_cols):
            print(f"      ✗ NOT FOUND")

    # Sample values
    print("\n  Sample values (first 3 rows):")
    print(df.head(3))

except Exception as e:
    print(f"\n✗ Failed to parse with multi-level header: {e}")

    # Try single-level
    try:
        df = pd.read_csv(csv_path)
        print("\n✓ Successfully parsed with single-level header")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e2:
        print(f"\n✗ Failed to parse with single-level header: {e2}")

print("\n" + "="*80)
