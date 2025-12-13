#!/usr/bin/env python3
"""
Update Airtable france_facilities with nearest substation from grid database.
"""
import requests
import pandas as pd
import math
import time
from typing import List, Dict, Tuple, Optional

# Airtable configuration - set via environment variables
import os
AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY", "")
BASE_ID = os.environ.get("AIRTABLE_BASE_ID", "app1151DXORQwxYwI")
TABLE_ID = os.environ.get("AIRTABLE_TABLE_ID", "tblubPzFuvdUXSXjL")
GRID_DATABASE = os.environ.get("GRID_DATABASE", "/home/user/grid_infrastructure/france_grid_database.csv")

if not AIRTABLE_API_KEY:
    raise ValueError("AIRTABLE_API_KEY environment variable must be set")

HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    "Content-Type": "application/json"
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km."""
    R = 6371  # Earth's radius in km

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


def fetch_all_facilities() -> List[Dict]:
    """Fetch all facilities from Airtable."""
    print("Fetching facilities from Airtable...")

    all_records = []
    offset = None

    while True:
        url = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_ID}"
        params = {"pageSize": 100}
        if offset:
            params["offset"] = offset

        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()

        records = data.get("records", [])
        all_records.extend(records)
        print(f"  Fetched {len(all_records)} records...")

        offset = data.get("offset")
        if not offset:
            break

        time.sleep(0.2)  # Rate limiting

    print(f"Total facilities: {len(all_records)}")
    return all_records


def load_grid_database() -> pd.DataFrame:
    """Load the grid database with coordinates."""
    print(f"\nLoading grid database from {GRID_DATABASE}...")
    df = pd.read_csv(GRID_DATABASE)

    # Filter to only records with coordinates
    df_with_coords = df[df['lat'].notna() & df['lon'].notna()].copy()
    print(f"  Total records: {len(df)}")
    print(f"  With coordinates: {len(df_with_coords)}")

    return df_with_coords


def find_nearest_substation(
    facility_lat: float,
    facility_lon: float,
    substations: pd.DataFrame
) -> Tuple[Optional[pd.Series], Optional[float]]:
    """Find the nearest substation to a facility."""

    if pd.isna(facility_lat) or pd.isna(facility_lon):
        return None, None

    min_distance = float('inf')
    nearest = None

    for _, sub in substations.iterrows():
        dist = haversine_distance(facility_lat, facility_lon, sub['lat'], sub['lon'])
        if dist < min_distance:
            min_distance = dist
            nearest = sub

    return nearest, min_distance


def find_nearest_substation_fast(
    facility_lat: float,
    facility_lon: float,
    substations: pd.DataFrame,
    lat_col: str = 'lat',
    lon_col: str = 'lon'
) -> Tuple[Optional[pd.Series], Optional[float]]:
    """Find the nearest substation using vectorized operations."""
    import numpy as np

    if pd.isna(facility_lat) or pd.isna(facility_lon):
        return None, None

    # Vectorized haversine calculation with numpy
    R = 6371
    lat1 = np.radians(facility_lat)
    lon1 = np.radians(facility_lon)

    lat2 = np.radians(substations[lat_col].values)
    lon2 = np.radians(substations[lon_col].values)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distances = R * c

    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    nearest = substations.iloc[min_idx]

    return nearest, min_distance


def update_airtable_batch(records: List[Dict]) -> bool:
    """Update a batch of Airtable records."""
    url = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_ID}"

    payload = {"records": records}

    response = requests.patch(url, headers=HEADERS, json=payload)

    if response.status_code != 200:
        print(f"  Error updating batch: {response.status_code}")
        print(f"  Response: {response.text[:500]}")
        return False

    return True


def get_voltage_class(voltage_in, voltage_out) -> str:
    """Determine voltage class string."""
    classes = []

    # Safely collect valid voltages
    voltages = []
    for v in [voltage_in, voltage_out]:
        try:
            if v is not None and not pd.isna(v) and float(v) > 0:
                voltages.append(float(v))
        except (ValueError, TypeError):
            pass

    if not voltages:
        return "UNKNOWN"

    max_v = max(voltages)
    min_v = min(voltages)

    if max_v >= 200:
        classes.append("EHV")
    elif max_v >= 60:
        classes.append("HV")
    elif max_v >= 10:
        classes.append("MV")
    else:
        classes.append("LV")

    if min_v != max_v:
        if min_v >= 200:
            if "EHV" not in classes:
                classes.append("EHV")
        elif min_v >= 60:
            if "HV" not in classes:
                classes.append("HV")
        elif min_v >= 10:
            if "MV" not in classes:
                classes.append("MV")
        else:
            if "LV" not in classes:
                classes.append("LV")

    return ",".join(classes) if classes else "UNKNOWN"


def main():
    print("=" * 60)
    print("UPDATING NEAREST SUBSTATIONS FOR FRANCE FACILITIES")
    print("=" * 60)

    # Load data
    facilities = fetch_all_facilities()
    substations = load_grid_database()

    # Prepare updates
    print("\nCalculating nearest substations...")
    updates = []

    for i, facility in enumerate(facilities):
        if (i + 1) % 50 == 0:
            print(f"  Processing {i + 1}/{len(facilities)}...")

        fields = facility.get("fields", {})
        lat = fields.get("latitude_2024")
        lon = fields.get("longitude_2024")

        if lat is None or lon is None:
            continue

        # Find nearest substation
        nearest, distance = find_nearest_substation_fast(lat, lon, substations)

        if nearest is None:
            continue

        # Prepare update fields
        voltage_in = nearest.get('voltage_in_kv')
        voltage_out = nearest.get('voltage_out_kv')

        update_fields = {
            "nearest_substation_name": str(nearest.get('name', ''))[:100] if nearest.get('name') else '',
            "nearest_substation_id": str(nearest.get('id', ''))[:100],
            "nearest_substation_operator": str(nearest.get('operator', ''))[:100] if nearest.get('operator') else '',
            "nearest_substation_distance_km": round(distance, 1),
            "nearest_substation_source": str(nearest.get('source', ''))[:50],
        }

        # Add voltage fields if available
        if voltage_in and not pd.isna(voltage_in):
            update_fields["nearest_substation_voltage_max"] = int(voltage_in)
        if voltage_out and not pd.isna(voltage_out):
            update_fields["nearest_substation_voltage_min"] = int(voltage_out)

        # Voltage class
        voltage_class = get_voltage_class(voltage_in, voltage_out)
        if voltage_class != "UNKNOWN":
            update_fields["nearest_substation_voltage_class"] = voltage_class

        updates.append({
            "id": facility["id"],
            "fields": update_fields
        })

    print(f"\nPrepared {len(updates)} updates")

    # Update in batches of 10 (Airtable limit)
    print("\nUpdating Airtable...")
    batch_size = 10
    success_count = 0

    for i in range(0, len(updates), batch_size):
        batch = updates[i:i + batch_size]

        if update_airtable_batch(batch):
            success_count += len(batch)

        if (i + batch_size) % 100 == 0 or i + batch_size >= len(updates):
            print(f"  Updated {min(i + batch_size, len(updates))}/{len(updates)} records...")

        time.sleep(0.25)  # Rate limiting

    print(f"\n{'=' * 60}")
    print(f"COMPLETE: Updated {success_count}/{len(updates)} facility records")
    print("=" * 60)


if __name__ == "__main__":
    main()
