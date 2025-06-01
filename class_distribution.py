import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("Cancer2025exam.csv")
df.to_csv("Cancer2025exam_semicolon.csv", sep=';', index=False)
# Check the number of representations from each class (assuming class label is in column 'V1')
class_counts = df['V1'].value_counts().sort_index()
print("Class representation counts:")
print(class_counts)
print("\n")

# Compute min, max, mean, and variance
min_vals = df.min()
max_vals = df.max()
mean_vals = df.mean()
var_vals = df.var()

# Format span as "min → max" with 3 significant digits
span = [f"{np.format_float_positional(min_vals[c], precision=3, unique=False, fractional=False, trim='k')} → "
        f"{np.format_float_positional(max_vals[c], precision=3, unique=False, fractional=False, trim='k')}"
        for c in df.columns]

# Round mean and variance to 3 significant digits
mean_rounded = mean_vals.apply(lambda x: f"{x:.3g}")
var_rounded = var_vals.apply(lambda x: f"{x:.3g}")

# Combine into summary dataframe
summary = pd.DataFrame({
    'Span (min → max)': span,
    'Mean': mean_rounded,
    'Variance': var_rounded
}, index=df.columns)

# Save to CSV with semicolon separator
summary.to_csv("feature_summary.csv", sep=';')

print("Feature summary saved to 'feature_summary.csv' with semicolon delimiter.")
