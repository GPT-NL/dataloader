import yaml
import pandas as pd


def flatten_dict(d, parent_key="", sep="."):
    """Recursively flattens a nested dict with dot notation."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


# Load YAML file
with open("datasets.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

# Create rows for Excel: each top-level entry becomes one row
rows = []
for entry_key, entry in data.items():
    flat_entry = flatten_dict(entry)
    # Optionally store the entry name as a column
    flat_entry["Entry_Name"] = entry_key
    rows.append(flat_entry)

# Assemble DataFrame, then save to Excel
df = pd.DataFrame(rows)
# Move Entry_Name to first column
columns = ["Entry_Name"] + [c for c in df.columns if c != "Entry_Name"]
df = df[columns]
df.to_excel("datasets.xlsx", index=False)
