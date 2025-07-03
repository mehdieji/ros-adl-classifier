import pandas as pd
from pathlib import Path
import re

def parse_sensomative_sensor_data(csv_file: str) -> dict:
    """
    Parse Sensomative sensor raw data from a CSV file.
    Only handles: 'datetime', '_pressure'.
    The '_pressure' column contains a string like "array('H', [3, 0, ...])" for 12 cells.
    Returns a dict: {modality: DataFrame}, with standardized column names.
    The only modality is 'pressure'.
    """
    csv_file = Path(csv_file)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)

    # Only keep the specified columns
    keep_cols = [
        'datetime',
        '_pressure'
    ]
    df = df[[col for col in keep_cols if col in df.columns]].copy()

    # Standardize column names
    rename_map = {
        'datetime': 'datetime',
        '_pressure': 'pressure',
    }
    df = df.rename(columns=rename_map)

    # Ensure datetime is parsed (ISO format)
    df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%dT%H:%M:%S.%f")

    # Parse the 'pressure' column into 12 separate columns
    def parse_pressure_array(val):
        matches = re.findall(r"\[(.*?)\]", str(val))
        if matches:
            nums = re.findall(r"\d+", matches[0])
            nums = [int(x) for x in nums]
            return (nums + [None]*12)[:12]
        return [None]*12

    pressure_values = df['pressure'].apply(parse_pressure_array)
    for i in range(12):
        df[f'pressure_{i}'] = pressure_values.apply(lambda x: x[i] if x and len(x) == 12 else None)

    # Only keep datetime and the 12 pressure columns
    out_cols = ['datetime'] + [f'pressure_{i}' for i in range(12)]
    modalities = {}
    modalities['pressure'] = df[out_cols].copy()
    return modalities 