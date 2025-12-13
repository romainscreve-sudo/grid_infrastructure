#!/usr/bin/env python3
"""
Substation Data Refinement Script
Cleans and prepares data for Airtable mapping sessions.

Refinements:
1. Fix ORE ID generation (remove 'None' values)
2. Clean malformed OSM names (filter garbage data)
3. Standardize and validate data
4. Deduplicate records across sources
5. Create Airtable-ready export
"""
import pandas as pd
import re
import os
from datetime import datetime
from typing import List, Tuple, Optional
import hashlib

# Configuration
INPUT_FILE = 'full_output.csv'
OUTPUT_FILE = 'refined_output.csv'
AIRTABLE_FILE = 'airtable_ready.csv'
REJECTED_FILE = 'rejected_records.csv'

# ============================================================================
# DATA CLEANING RULES
# ============================================================================

# Names that indicate garbage/non-substation data
GARBAGE_NAME_PATTERNS = [
    r'papiers?\s*(et|,)\s*v[eê]tements',  # Recycling containers
    r'd[eé]chets?\s*(verres?|papiers?|v[eê]tements?)',  # Waste containers
    r'poubelles?',  # Trash bins
    r'conteneurs?',  # Containers
    r'recyclage',  # Recycling
    r'tri\s+s[eé]lectif',  # Selective sorting
    r'ordures',  # Garbage
    r'point\s+d[\'e\s]apport',  # Collection point (waste)
    r'bennes?',  # Dumpsters
]

# Compile patterns for efficiency
GARBAGE_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in GARBAGE_NAME_PATTERNS]

# Valid substation name patterns (to keep)
VALID_SUBSTATION_PATTERNS = [
    r'^PS\s+',  # Poste Source
    r'^Poste\s+',  # Poste
    r'substation',  # English substation
    r'transformer',  # Transformer
    r'sous[- ]?station',  # French substation
    r'R\.?D\.?\s*[A-Z]',  # R.D. distribution cabinets (Strasbourg)
    r'PSSA',  # Poste Source Sous Antenne
    r'kV',  # Contains voltage info
    r'HTA|HTB|BT',  # French voltage levels
    r'^[A-Z]{2,6}\s*\d',  # Alphanumeric codes like "ABC123"
    r'transmission',
    r'distribution',
]

VALID_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in VALID_SUBSTATION_PATTERNS]

# ============================================================================
# CLEANING FUNCTIONS
# ============================================================================

def is_garbage_name(name: str) -> bool:
    """Check if a name indicates garbage/non-substation data."""
    if not name or pd.isna(name):
        return False

    name_lower = name.lower().strip()

    for pattern in GARBAGE_PATTERNS_COMPILED:
        if pattern.search(name_lower):
            return True

    return False


def is_valid_substation_name(name: str) -> bool:
    """Check if name looks like a valid substation name."""
    if not name or pd.isna(name):
        return False

    # If it starts with OSM_ prefix (auto-generated), it's okay
    if name.startswith('OSM_'):
        return True

    for pattern in VALID_PATTERNS_COMPILED:
        if pattern.search(name):
            return True

    # Names with just letters/numbers and common separators are likely valid
    if re.match(r'^[A-Z0-9][A-Za-z0-9\s\-\.\,\']+$', name):
        return True

    return True  # Default to keeping unless explicitly garbage


def clean_ore_id(row: pd.Series) -> str:
    """Fix ORE IDs that contain 'None' values."""
    id_global = row['id_global']

    if 'FR_ORE_None_' in id_global:
        # Generate a new ID using name and coordinates
        name = row['name'] if pd.notna(row['name']) else 'unknown'
        lat = row['lat'] if pd.notna(row['lat']) else 0
        lon = row['lon'] if pd.notna(row['lon']) else 0

        # Clean the name for ID
        name_clean = re.sub(r'[^a-zA-Z0-9]', '_', name)[:30]

        # Create deterministic ID from coordinates
        coord_hash = hashlib.md5(f"{lat:.6f}_{lon:.6f}".encode()).hexdigest()[:12]

        return f"FR_ORE_{name_clean}_{coord_hash}"

    return id_global


def standardize_name(name: str) -> str:
    """Standardize substation names."""
    if not name or pd.isna(name):
        return name

    name = str(name).strip()

    # Expand common abbreviations
    name = re.sub(r'^PS\s+', 'Poste Source ', name)
    name = re.sub(r'^PSSA\.?', 'PSSA ', name)

    # Clean up multiple spaces
    name = re.sub(r'\s+', ' ', name)

    return name


def extract_commune_from_notes(notes: str) -> Optional[str]:
    """Extract commune name from notes field."""
    if not notes or pd.isna(notes):
        return None

    match = re.search(r'Commune:\s*([^;]+)', notes)
    if match:
        return match.group(1).strip()

    return None


def extract_region_from_notes(notes: str) -> Optional[str]:
    """Extract region name from notes field."""
    if not notes or pd.isna(notes):
        return None

    match = re.search(r'Region:\s*([^;]+)', notes)
    if match:
        return match.group(1).strip()

    return None


def compute_dedup_key(row: pd.Series) -> str:
    """Compute a key for deduplication based on location and name similarity."""
    # Round coordinates to ~100m precision for matching
    lat_round = round(row['lat'] / 0.001) * 0.001 if pd.notna(row['lat']) else 0
    lon_round = round(row['lon'] / 0.001) * 0.001 if pd.notna(row['lon']) else 0

    # Normalize name
    name = str(row['name']).lower().strip() if pd.notna(row['name']) else ''
    name = re.sub(r'[^a-z0-9]', '', name)[:20]

    return f"{lat_round:.3f}_{lon_round:.3f}_{name}"


def select_best_record(group: pd.DataFrame) -> pd.Series:
    """Select the best record from a group of potential duplicates."""
    # Priority: TSO_DSO_OFFICIAL > GOVERNMENT_REPOSITORY > COMMERCIAL > COMMUNITY
    source_priority = {1: 0, 2: 1, 3: 2, 4: 3}

    # Sort by source rank (lower is better), then by data completeness
    def score_record(row):
        source_score = source_priority.get(row['source_rank'], 99)

        # Count non-null fields as completeness
        completeness = row.notna().sum()

        return (source_score, -completeness)

    sorted_group = group.copy()
    sorted_group['_score'] = sorted_group.apply(score_record, axis=1)
    sorted_group = sorted_group.sort_values('_score')

    best = sorted_group.iloc[0].drop('_score')

    # Merge in any additional info from other records
    if len(group) > 1:
        # Collect all source IDs for cross-reference
        all_sources = group['source_primary'].unique().tolist()
        all_ids = group['id_source'].dropna().unique().tolist()

        best['notes'] = (
            str(best['notes'] or '') +
            f"; Merged from {len(group)} sources: {', '.join(all_sources)}"
        ).strip('; ')

    return best


# ============================================================================
# MAIN REFINEMENT PIPELINE
# ============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV data."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  -> Loaded {len(df):,} records")
    return df


def clean_garbage_records(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove records with garbage names."""
    print("\nCleaning garbage records...")

    is_garbage = df['name'].apply(is_garbage_name)

    garbage_df = df[is_garbage].copy()
    clean_df = df[~is_garbage].copy()

    print(f"  -> Removed {len(garbage_df):,} garbage records")
    print(f"  -> Kept {len(clean_df):,} valid records")

    if len(garbage_df) > 0:
        print("\n  Sample garbage records:")
        for _, row in garbage_df.head(5).iterrows():
            print(f"    - {row['name'][:60]}")

    return clean_df, garbage_df


def fix_ore_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Fix ORE IDs that contain 'None' values."""
    print("\nFixing ORE IDs...")

    none_count = df['id_global'].str.contains('None', na=False).sum()
    print(f"  -> Found {none_count:,} IDs with 'None' values")

    df['id_global'] = df.apply(clean_ore_id, axis=1)

    remaining = df['id_global'].str.contains('None', na=False).sum()
    print(f"  -> Fixed {none_count - remaining:,} IDs")

    return df


def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived fields for Airtable."""
    print("\nAdding derived fields...")

    # Extract commune and region from notes
    df['commune'] = df['notes'].apply(extract_commune_from_notes)
    df['region'] = df['notes'].apply(extract_region_from_notes)

    # Standardize names
    df['name_standardized'] = df['name'].apply(standardize_name)

    # Add data quality score (simple scoring)
    def quality_score(row):
        score = 0
        if pd.notna(row['name']) and len(str(row['name'])) > 2:
            score += 20
        if pd.notna(row['lat']) and pd.notna(row['lon']):
            score += 30
        if pd.notna(row['voltage_kv_nominal_max']):
            score += 20
        if pd.notna(row['operator_name']):
            score += 15
        if row['source_rank'] in [1, 2]:
            score += 15
        return score

    df['data_quality_score'] = df.apply(quality_score, axis=1)

    # Count records with each field populated
    populated = {
        'coordinates': df[['lat', 'lon']].notna().all(axis=1).sum(),
        'voltage': df['voltage_kv_nominal_max'].notna().sum(),
        'operator': df['operator_name'].notna().sum(),
        'commune': df['commune'].notna().sum(),
    }

    print(f"  -> Added commune/region for {populated['commune']:,} records")
    print(f"  -> Records with coordinates: {populated['coordinates']:,}")
    print(f"  -> Records with voltage: {populated['voltage']:,}")

    return df


def deduplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate records based on location and name similarity."""
    print("\nDeduplicating records...")

    original_count = len(df)

    # Compute dedup keys
    df['_dedup_key'] = df.apply(compute_dedup_key, axis=1)

    # Group by dedup key and select best record
    deduped = df.groupby('_dedup_key', group_keys=False).apply(
        select_best_record, include_groups=False
    ).reset_index(drop=True)

    # Clean up
    deduped = deduped.drop(columns=['_dedup_key'], errors='ignore')

    duplicates_removed = original_count - len(deduped)
    print(f"  -> Removed {duplicates_removed:,} duplicate records")
    print(f"  -> Kept {len(deduped):,} unique records")

    return deduped


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and flag problematic records."""
    print("\nValidating data...")

    # Flag records for review
    df['needs_review'] = False

    # Flag records with missing coordinates
    no_coords = df[['lat', 'lon']].isna().any(axis=1)
    df.loc[no_coords, 'needs_review'] = True

    # Flag records with no name
    no_name = df['name'].isna() | (df['name'].str.len() < 2)
    df.loc[no_name, 'needs_review'] = True

    # Flag records with suspicious coordinates (outside France/Australia)
    def check_coords(row):
        if pd.isna(row['lat']) or pd.isna(row['lon']):
            return False

        lat, lon = row['lat'], row['lon']

        # France bounds (including overseas territories)
        if row['country'] == 'FR':
            # Metropolitan France (including Ouessant, Corsica)
            if -5.5 < lon < 10 and 41 < lat < 52:
                return False
            # Mayotte (Indian Ocean)
            if 44 < lon < 46 and -14 < lat < -11:
                return False
            # Réunion (Indian Ocean)
            if 54 < lon < 57 and -22 < lat < -19:
                return False
            # French Polynesia (Pacific)
            if -155 < lon < -134 and -28 < lat < -7:
                return False
            # Guadeloupe/Martinique (Caribbean)
            if -64 < lon < -60 and 14 < lat < 18:
                return False
            # Saint-Martin/Saint-Barthélemy (Caribbean)
            if -64 < lon < -62 and 17 < lat < 19:
                return False
            # French Guiana (South America)
            if -55 < lon < -50 and 2 < lat < 7:
                return False
            # New Caledonia (Pacific)
            if 163 < lon < 172 and -23 < lat < -18:
                return False
            # Saint Pierre and Miquelon (Atlantic)
            if -57 < lon < -55 and 46 < lat < 48:
                return False
            # Wallis and Futuna (Pacific)
            if -179 < lon < -176 and -15 < lat < -13:
                return False
            return True  # Outside all known French territories

        # Australia bounds (rough)
        elif row['country'] == 'AU':
            if not (112 < lon < 154 and -44 < lat < -10):
                return True

        return False

    bad_coords = df.apply(check_coords, axis=1)
    df.loc[bad_coords, 'needs_review'] = True

    review_count = df['needs_review'].sum()
    print(f"  -> Flagged {review_count:,} records for review")
    print(f"    - Missing coordinates: {no_coords.sum():,}")
    print(f"    - Missing/short name: {no_name.sum():,}")
    print(f"    - Suspicious coordinates: {bad_coords.sum():,}")

    return df


def create_airtable_export(df: pd.DataFrame, output_path: str):
    """Create an Airtable-ready export with clean field mappings."""
    print(f"\nCreating Airtable export: {output_path}")

    # Select and rename columns for Airtable
    airtable_columns = {
        'id_global': 'ID',
        'name_standardized': 'Name',
        'country': 'Country',
        'source_primary': 'Source',
        'source_rank': 'Source Rank',
        'level_tx_dx': 'Network Level',
        'substation_role': 'Substation Role',
        'operator_name': 'Operator',
        'voltage_kv_nominal_max': 'Voltage Max (kV)',
        'voltage_kv_nominal_min': 'Voltage Min (kV)',
        'voltage_classes': 'Voltage Classes',
        'voltage_quality_flag': 'Voltage Quality',
        'capacity_quality_flag': 'Capacity Quality',
        'lat': 'Latitude',
        'lon': 'Longitude',
        'geom_quality_flag': 'Geometry Quality',
        'commune': 'Commune',
        'region': 'Region',
        'licence_code': 'License',
        'data_quality_score': 'Quality Score',
        'needs_review': 'Needs Review',
        'notes': 'Notes',
    }

    # Create export DataFrame
    export_df = df[list(airtable_columns.keys())].rename(columns=airtable_columns)

    # Format for Airtable
    export_df['Needs Review'] = export_df['Needs Review'].map({True: 'Yes', False: 'No'})

    # Save
    export_df.to_csv(output_path, index=False)

    print(f"  -> Exported {len(export_df):,} records")
    print(f"  -> Columns: {len(export_df.columns)}")

    return export_df


def print_summary(df: pd.DataFrame):
    """Print a summary of the refined data."""
    print("\n" + "=" * 70)
    print("REFINED DATA SUMMARY")
    print("=" * 70)

    print(f"\nTotal records: {len(df):,}")

    print("\nBy country:")
    print(df['country'].value_counts().to_string())

    print("\nBy source:")
    print(df['source_primary'].value_counts().to_string())

    print("\nBy network level:")
    print(df['level_tx_dx'].value_counts().to_string())

    print("\nData quality distribution:")
    quality_bins = pd.cut(df['data_quality_score'], bins=[0, 25, 50, 75, 100], labels=['Low', 'Medium', 'Good', 'Excellent'])
    print(quality_bins.value_counts().to_string())

    print("\nRecords by review status:")
    print(df['needs_review'].value_counts().rename({True: 'Needs Review', False: 'OK'}).to_string())


def run_refinement_pipeline():
    """Run the complete data refinement pipeline."""
    print("=" * 70)
    print("SUBSTATION DATA REFINEMENT PIPELINE")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Check input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return

    # Load data
    df = load_data(INPUT_FILE)

    # Step 1: Clean garbage records
    df, garbage_df = clean_garbage_records(df)

    # Save rejected records for review
    if len(garbage_df) > 0:
        garbage_df.to_csv(REJECTED_FILE, index=False)
        print(f"  -> Saved rejected records to: {REJECTED_FILE}")

    # Step 2: Fix ORE IDs
    df = fix_ore_ids(df)

    # Step 3: Add derived fields
    df = add_derived_fields(df)

    # Step 4: Deduplicate
    df = deduplicate_records(df)

    # Step 5: Validate
    df = validate_data(df)

    # Save refined output
    print(f"\nSaving refined data: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"  -> Saved {len(df):,} records")

    # Create Airtable export
    create_airtable_export(df, AIRTABLE_FILE)

    # Print summary
    print_summary(df)

    print("\n" + "=" * 70)
    print(f"COMPLETE: {datetime.now().isoformat()}")
    print("=" * 70)

    return df


if __name__ == '__main__':
    run_refinement_pipeline()
