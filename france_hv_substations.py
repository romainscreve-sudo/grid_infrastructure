#!/usr/bin/env python3
"""
France High-Voltage Substations (>11kV) - Unified View

Combines data from:
1. RTE/ODRÉ - Transmission substations (explicit voltage)
2. Agence ORE - Postes source (HTB/HTA interface, all >11kV)
3. OpenStreetMap - Substations with voltage >11kV or HTA type

French voltage conventions:
- THT (Très Haute Tension): 225-400 kV
- HTB (Haute Tension B): 63-90 kV
- HTA (Haute Tension A): 15-20 kV
- BT (Basse Tension): 230-400 V

All postes source (ORE) are by definition >11kV as they interface HTB→HTA.
"""
import requests
import json
import re
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import hashlib

# ============================================================================
# CONFIGURATION
# ============================================================================

# API endpoints
RTE_ODRE_API = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/postes-electriques-rte/records"
AGENCE_ORE_API = "https://opendata.agenceore.fr/api/explore/v2.1/catalog/datasets/postes-source/records"
OVERPASS_API = "https://overpass-api.de/api/interpreter"

# French voltage levels (kV)
VOLTAGE_LEVELS = {
    'THT': [400, 225],           # Très Haute Tension
    'HTB': [90, 63],              # Haute Tension B
    'HTA': [20, 15],              # Haute Tension A
    'BT': [0.4, 0.23],            # Basse Tension
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Substation:
    """Unified substation record."""
    id: str
    name: str
    source: str
    source_rank: int  # 1=TSO/DSO, 2=Gov, 3=Commercial, 4=Community

    # Voltage
    voltage_max_kv: Optional[float]
    voltage_min_kv: Optional[float]
    voltage_level: str  # THT, HTB, HTA, etc.
    voltage_source: str  # 'explicit', 'implied_from_type', 'unknown'

    # Location
    lat: Optional[float]
    lon: Optional[float]
    coord_quality: str  # 'exact', 'approximate', 'degraded'

    # Metadata
    operator: Optional[str]
    commune: Optional[str]
    region: Optional[str]
    substation_type: str  # 'transmission', 'poste_source', 'distribution'

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'source': self.source,
            'source_rank': self.source_rank,
            'voltage_max_kv': self.voltage_max_kv,
            'voltage_min_kv': self.voltage_min_kv,
            'voltage_level': self.voltage_level,
            'voltage_source': self.voltage_source,
            'lat': self.lat,
            'lon': self.lon,
            'coord_quality': self.coord_quality,
            'operator': self.operator,
            'commune': self.commune,
            'region': self.region,
            'substation_type': self.substation_type,
        }


def classify_voltage_level(kv: float) -> str:
    """Classify voltage to French standard levels."""
    if kv >= 200:
        return 'THT'
    elif kv >= 45:
        return 'HTB'
    elif kv >= 10:
        return 'HTA'
    else:
        return 'BT'


# ============================================================================
# RTE/ODRÉ EXTRACTION (Transmission)
# ============================================================================

def extract_rte() -> List[Substation]:
    """Extract RTE transmission substations."""
    substations = []
    offset = 0
    limit = 100

    print("  [RTE] Fetching transmission substations...")

    while True:
        try:
            response = requests.get(RTE_ODRE_API, params={'limit': limit, 'offset': offset}, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"    Error fetching RTE data: {e}")
            break

        records = data.get('results', [])
        if not records:
            break

        for r in records:
            # Parse voltage from string like "400 kV / 225 kV"
            voltage_str = r.get('tension', '')
            voltages = re.findall(r'(\d+)\s*kV', voltage_str, re.IGNORECASE)
            voltages = [float(v) for v in voltages] if voltages else []

            v_max = max(voltages) if voltages else None
            v_min = min(voltages) if voltages else None

            geo = r.get('geo_point_2d', {}) or {}
            name = r.get('nom_poste', '') or ''
            code = r.get('code_poste', '') or name.replace(' ', '_')

            sub = Substation(
                id=f"RTE_{code}",
                name=name,
                source='RTE_ODRE',
                source_rank=1,
                voltage_max_kv=v_max,
                voltage_min_kv=v_min,
                voltage_level=classify_voltage_level(v_max) if v_max else 'HTB',
                voltage_source='explicit' if voltages else 'unknown',
                lat=geo.get('lat'),
                lon=geo.get('lon'),
                coord_quality='degraded',  # RTE degrades GPS for security
                operator='RTE',
                commune=None,
                region=r.get('region', ''),
                substation_type='transmission',
            )
            substations.append(sub)

        offset += limit
        if len(records) < limit:
            break

    print(f"    -> {len(substations)} RTE substations")
    return substations


# ============================================================================
# AGENCE ORE EXTRACTION (Postes Source)
# ============================================================================

def extract_ore() -> List[Substation]:
    """Extract Agence ORE postes source (HTB/HTA interface)."""
    substations = []
    offset = 0
    limit = 100

    print("  [ORE] Fetching postes source (HTB/HTA)...")

    while True:
        try:
            response = requests.get(AGENCE_ORE_API, params={'limit': limit, 'offset': offset}, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"    Error fetching ORE data: {e}")
            break

        records = data.get('results', [])
        if not records:
            break

        for r in records:
            geo = r.get('geo_point_2d', {}) or {}
            name = r.get('nom_ps', '') or f"PS {r.get('commune', '')}"
            operator = r.get('nom_grd', '') or 'Unknown DSO'

            # Generate stable ID from coordinates
            lat = geo.get('lat', 0)
            lon = geo.get('lon', 0)
            coord_hash = hashlib.md5(f"{lat:.6f}_{lon:.6f}".encode()).hexdigest()[:8]

            sub = Substation(
                id=f"ORE_{coord_hash}",
                name=name,
                source='AGENCE_ORE',
                source_rank=1,
                # Postes source interface HTB (63-90kV) to HTA (20kV)
                # We report the HTB side as max, HTA as min
                voltage_max_kv=None,  # Unknown - could be 63 or 90
                voltage_min_kv=20.0,  # HTA standard
                voltage_level='HTB/HTA',
                voltage_source='implied_from_type',  # All postes source are HTB/HTA
                lat=lat if lat else None,
                lon=lon if lon else None,
                coord_quality='exact',
                operator=operator,
                commune=r.get('commune', ''),
                region=r.get('region', ''),
                substation_type='poste_source',
            )
            substations.append(sub)

        offset += limit
        if len(records) < limit:
            break

    print(f"    -> {len(substations)} ORE postes source")
    return substations


# ============================================================================
# OSM EXTRACTION (>11kV substations)
# ============================================================================

def extract_osm_hv() -> List[Substation]:
    """Extract OSM substations with voltage >11kV or transmission type."""
    substations = []

    print("  [OSM] Fetching HV substations (>11kV)...")

    # Overpass query for French substations with HV characteristics
    query = """
    [out:json][timeout:180];
    area["ISO3166-1"="FR"]->.france;
    (
      // Transmission substations
      node["power"="substation"]["substation"="transmission"](area.france);
      way["power"="substation"]["substation"="transmission"](area.france);
      relation["power"="substation"]["substation"="transmission"](area.france);

      // Substations with voltage >= 11000V
      node["power"="substation"]["voltage"](area.france);
      way["power"="substation"]["voltage"](area.france);
      relation["power"="substation"]["voltage"](area.france);
    );
    out center tags;
    """

    try:
        response = requests.post(OVERPASS_API, data={'data': query}, timeout=200)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"    Error fetching OSM data: {e}")
        return substations

    elements = data.get('elements', [])

    for elem in elements:
        tags = elem.get('tags', {})

        # Get coordinates
        if elem['type'] == 'node':
            lon, lat = elem.get('lon'), elem.get('lat')
        else:
            center = elem.get('center', {})
            lon, lat = center.get('lon'), center.get('lat')

        if not lon or not lat:
            continue

        # Parse voltage
        voltage_str = tags.get('voltage', '')
        v_max, v_min = None, None

        if voltage_str:
            # Handle formats: "400000", "400000;225000", "400 kV"
            voltages = re.findall(r'(\d+)', voltage_str)
            if voltages:
                # Convert to kV if in volts
                voltages = [float(v)/1000 if float(v) > 1000 else float(v) for v in voltages]
                v_max = max(voltages)
                v_min = min(voltages)

        # Filter: only include if voltage >11kV or transmission type
        substation_type = tags.get('substation', '')
        is_transmission = substation_type == 'transmission'
        is_hv = v_max and v_max >= 11

        if not (is_transmission or is_hv):
            continue

        name = tags.get('name', '') or tags.get('ref', '') or f"OSM_{elem['type']}_{elem['id']}"
        operator = tags.get('operator', '') or tags.get('owner', '')

        # Determine voltage level
        if v_max:
            v_level = classify_voltage_level(v_max)
        elif is_transmission:
            v_level = 'HTB'  # Assume transmission is at least HTB
        else:
            v_level = 'HTA'

        sub = Substation(
            id=f"OSM_{elem['type']}_{elem['id']}",
            name=name,
            source='OSM',
            source_rank=4,
            voltage_max_kv=v_max,
            voltage_min_kv=v_min,
            voltage_level=v_level,
            voltage_source='explicit' if v_max else 'implied_from_type',
            lat=lat,
            lon=lon,
            coord_quality='exact',
            operator=operator if operator else None,
            commune=None,
            region=None,
            substation_type='transmission' if is_transmission else 'distribution',
        )
        substations.append(sub)

    print(f"    -> {len(substations)} OSM HV substations")
    return substations


# ============================================================================
# MATCHING & DEDUPLICATION
# ============================================================================

def match_rte_with_osm(rte_subs: List[Substation], osm_subs: List[Substation]) -> List[Substation]:
    """
    Try to match RTE substations (degraded coords) with OSM (good coords).
    Returns enriched RTE records where matches found.
    """
    print("\n  Matching RTE with OSM for coordinate enrichment...")

    matches = 0
    enriched = []

    # Index OSM by normalized name for fast lookup
    osm_by_name = {}
    for osm in osm_subs:
        name_norm = re.sub(r'[^a-z0-9]', '', osm.name.lower())
        if name_norm and len(name_norm) > 3:
            if name_norm not in osm_by_name:
                osm_by_name[name_norm] = []
            osm_by_name[name_norm].append(osm)

    for rte in rte_subs:
        rte_name_norm = re.sub(r'[^a-z0-9]', '', rte.name.lower())

        # Try exact name match
        if rte_name_norm in osm_by_name:
            # Find closest OSM match by distance
            best_match = None
            best_dist = float('inf')

            for osm in osm_by_name[rte_name_norm]:
                if rte.lat and rte.lon and osm.lat and osm.lon:
                    # Simple distance check (not accounting for earth curvature)
                    dist = ((rte.lat - osm.lat)**2 + (rte.lon - osm.lon)**2)**0.5
                    if dist < best_dist and dist < 0.1:  # ~10km tolerance
                        best_dist = dist
                        best_match = osm

            if best_match:
                # Enrich RTE with OSM coordinates
                rte.lat = best_match.lat
                rte.lon = best_match.lon
                rte.coord_quality = 'exact'  # Now we have good coords
                matches += 1

        enriched.append(rte)

    print(f"    -> Matched {matches} RTE substations with OSM coordinates")
    return enriched


def deduplicate(substations: List[Substation]) -> List[Substation]:
    """
    Deduplicate substations, preferring higher-ranked sources.
    """
    print("\n  Deduplicating substations...")

    # Group by approximate location (~500m)
    location_groups = {}

    for sub in substations:
        if sub.lat and sub.lon:
            # Round to ~500m precision
            key = (round(sub.lat * 200) / 200, round(sub.lon * 200) / 200)
        else:
            key = ('no_coords', sub.name)

        if key not in location_groups:
            location_groups[key] = []
        location_groups[key].append(sub)

    # Select best from each group
    deduped = []
    duplicates = 0

    for key, group in location_groups.items():
        if len(group) == 1:
            deduped.append(group[0])
        else:
            # Sort by source rank (lower is better), then by voltage data quality
            def score(s):
                rank = s.source_rank
                has_voltage = 0 if s.voltage_max_kv else 1
                has_coords = 0 if (s.lat and s.lon) else 1
                return (rank, has_voltage, has_coords)

            group.sort(key=score)
            best = group[0]

            # Merge info from other records
            for other in group[1:]:
                if not best.voltage_max_kv and other.voltage_max_kv:
                    best.voltage_max_kv = other.voltage_max_kv
                    best.voltage_min_kv = other.voltage_min_kv
                    best.voltage_source = other.voltage_source
                if not best.operator and other.operator:
                    best.operator = other.operator
                if not best.commune and other.commune:
                    best.commune = other.commune

            deduped.append(best)
            duplicates += len(group) - 1

    print(f"    -> Removed {duplicates} duplicates")
    print(f"    -> {len(deduped)} unique substations")

    return deduped


# ============================================================================
# MAIN
# ============================================================================

def extract_france_hv_substations(output_path: str = 'france_hv_substations.csv'):
    """
    Extract all France substations >11kV from multiple sources.
    """
    print("=" * 70)
    print("FRANCE HIGH-VOLTAGE SUBSTATIONS (>11kV)")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Extract from each source
    print("\n" + "-" * 70)
    print("EXTRACTION")
    print("-" * 70)

    rte_subs = extract_rte()
    ore_subs = extract_ore()
    osm_subs = extract_osm_hv()

    # Match RTE with OSM to improve coordinates
    print("\n" + "-" * 70)
    print("MATCHING & ENRICHMENT")
    print("-" * 70)

    rte_enriched = match_rte_with_osm(rte_subs, osm_subs)

    # Combine all sources
    all_subs = rte_enriched + ore_subs + osm_subs
    print(f"\n  Total before dedup: {len(all_subs)}")

    # Deduplicate
    deduped = deduplicate(all_subs)

    # Convert to DataFrame
    print("\n" + "-" * 70)
    print("EXPORT")
    print("-" * 70)

    records = [s.to_dict() for s in deduped]
    df = pd.DataFrame(records)

    # Summary
    print(f"\nTotal HV substations (>11kV): {len(df)}")

    print("\nBy source:")
    print(df['source'].value_counts().to_string())

    print("\nBy voltage level:")
    print(df['voltage_level'].value_counts().to_string())

    print("\nBy substation type:")
    print(df['substation_type'].value_counts().to_string())

    print("\nVoltage data quality:")
    print(df['voltage_source'].value_counts().to_string())

    print("\nCoordinate quality:")
    print(df['coord_quality'].value_counts().to_string())

    # With coordinates
    has_coords = df[['lat', 'lon']].notna().all(axis=1).sum()
    print(f"\nRecords with coordinates: {has_coords} ({100*has_coords/len(df):.1f}%)")

    # Export
    df.to_csv(output_path, index=False)
    print(f"\nExported to: {output_path}")

    # Also export GeoJSON
    geojson_path = output_path.replace('.csv', '.geojson')
    export_geojson(deduped, geojson_path)
    print(f"Exported to: {geojson_path}")

    print("\n" + "=" * 70)
    print(f"COMPLETE: {datetime.now().isoformat()}")
    print("=" * 70)

    return df


def export_geojson(substations: List[Substation], filepath: str):
    """Export to GeoJSON."""
    features = []

    for sub in substations:
        if sub.lon and sub.lat:
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [sub.lon, sub.lat]
                },
                'properties': sub.to_dict()
            }
            features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'features': features,
        'metadata': {
            'generated': datetime.now().isoformat(),
            'description': 'France HV Substations (>11kV)',
            'total_features': len(features),
        }
    }

    with open(filepath, 'w') as f:
        json.dump(geojson, f, indent=2)


if __name__ == '__main__':
    extract_france_hv_substations()
