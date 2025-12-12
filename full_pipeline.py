#!/usr/bin/env python3
"""
Full Grid Infrastructure Pipeline
Extracts ALL transmission and distribution substations for France and Australia.

Data Sources:
- France Transmission: RTE/ODRÉ API (public)
- France Distribution: Agence ORE API + Enedis API (public)
- Australia Transmission: Geoscience Australia NEI API (public)
- Australia Distribution: OpenStreetMap Overpass API (public)

No API keys required.
"""
import requests
import json
import re
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import time

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

# ============================================================================
# SCHEMA DEFINITIONS
# ============================================================================

class Country(Enum):
    FR = "FR"
    AU = "AU"

class SourceRank(Enum):
    TSO_DSO_OFFICIAL = 1
    GOVERNMENT_REPOSITORY = 2
    COMMERCIAL = 3
    COMMUNITY = 4

class NetworkLevel(Enum):
    TX = "TX"
    DX_PRIMARY = "DX_PRIMARY"
    DX_SECONDARY = "DX_SECONDARY"
    DX_OTHER = "DX_OTHER"

class SubstationRole(Enum):
    INTERFACE_TX_DX = "INTERFACE_TX_DX"
    ZONE_SUBSTATION = "ZONE_SUBSTATION"
    DISTRIBUTION_SUBSTATION = "DISTRIBUTION_SUBSTATION"
    GEN_CONNECTION_POINT = "GEN_CONNECTION_POINT"
    SWITCHING_STATION = "SWITCHING_STATION"
    UNKNOWN = "UNKNOWN"

class VoltageQuality(Enum):
    NUMERIC_EXACT = "NUMERIC_EXACT"
    RANGE_OR_CLASS_ONLY = "RANGE_OR_CLASS_ONLY"
    UNKNOWN = "UNKNOWN"

class CapacityQuality(Enum):
    TSO_CONFIRMED = "TSO_CONFIRMED"
    INDICATIVE = "INDICATIVE"
    HISTORICAL = "HISTORICAL"
    MISSING = "MISSING"

class GeomQuality(Enum):
    TSO_DSO_EXACT = "TSO_DSO_EXACT"
    GOV_APPROX = "GOV_APPROX"
    DERIVED_FROM_OSM = "DERIVED_FROM_OSM"
    COMMERCIAL = "COMMERCIAL"
    UNKNOWN = "UNKNOWN"

@dataclass
class GridNode:
    """Unified substation / connection-point record."""
    id_global: str
    country: Country
    source_primary: str
    source_rank: SourceRank
    name: str
    level_tx_dx: NetworkLevel
    substation_role: SubstationRole = SubstationRole.UNKNOWN
    operator_name: Optional[str] = None
    operator_id: Optional[str] = None
    voltage_kv_nominal_max: Optional[float] = None
    voltage_kv_nominal_min: Optional[float] = None
    voltage_classes: Optional[List[str]] = None
    voltage_quality_flag: VoltageQuality = VoltageQuality.UNKNOWN
    installed_capacity_mva: Optional[float] = None
    available_capacity_mw: Optional[float] = None
    capacity_quality_flag: CapacityQuality = CapacityQuality.MISSING
    lon: Optional[float] = None
    lat: Optional[float] = None
    geom_type: str = "POINT"
    geom_quality_flag: GeomQuality = GeomQuality.UNKNOWN
    id_source: Optional[str] = None
    last_update: Optional[datetime] = None
    licence_code: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'id_global': self.id_global,
            'country': self.country.value,
            'source_primary': self.source_primary,
            'source_rank': self.source_rank.value,
            'name': self.name,
            'level_tx_dx': self.level_tx_dx.value,
            'substation_role': self.substation_role.value,
            'operator_name': self.operator_name,
            'operator_id': self.operator_id,
            'voltage_kv_nominal_max': self.voltage_kv_nominal_max,
            'voltage_kv_nominal_min': self.voltage_kv_nominal_min,
            'voltage_classes': ','.join(self.voltage_classes) if self.voltage_classes else None,
            'voltage_quality_flag': self.voltage_quality_flag.value,
            'installed_capacity_mva': self.installed_capacity_mva,
            'available_capacity_mw': self.available_capacity_mw,
            'capacity_quality_flag': self.capacity_quality_flag.value,
            'lon': self.lon,
            'lat': self.lat,
            'geom_type': self.geom_type,
            'geom_quality_flag': self.geom_quality_flag.value,
            'id_source': self.id_source,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'licence_code': self.licence_code,
            'notes': self.notes,
        }

def classify_voltage(kv: float) -> str:
    if kv >= 200:
        return 'EHV'
    elif kv >= 30:
        return 'HV'
    elif kv >= 1:
        return 'MV'
    else:
        return 'LV'

def generate_global_id(country: str, source: str, local_id: str) -> str:
    return f"{country}_{source}_{local_id}"

# ============================================================================
# FRANCE: RTE/ODRÉ (Transmission)
# ============================================================================

RTE_ODRE_API = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/postes-electriques-rte/records"

def extract_france_rte(max_records: int = None) -> List[GridNode]:
    """Extract RTE transmission substations."""
    nodes = []
    offset = 0
    limit = 100

    print("  [FR] Fetching RTE/ODRÉ transmission substations...")

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
            try:
                # Parse voltage
                voltage_str = r.get('tension', '')
                voltages = re.findall(r'(\d+)\s*kV', voltage_str, re.IGNORECASE)
                voltages = [float(v) for v in voltages]
                v_max = max(voltages) if voltages else None
                v_min = min(voltages) if voltages else None

                v_classes = []
                if v_max:
                    v_classes.append(classify_voltage(v_max))
                if v_min and classify_voltage(v_min) not in v_classes:
                    v_classes.append(classify_voltage(v_min))

                geo = r.get('geo_point_2d', {}) or {}
                local_id = r.get('code_poste') or r.get('nom_poste', '').replace(' ', '_')

                node = GridNode(
                    id_global=generate_global_id('FR', 'RTE', local_id),
                    country=Country.FR,
                    source_primary='RTE_ODRE',
                    source_rank=SourceRank.TSO_DSO_OFFICIAL,
                    name=r.get('nom_poste', ''),
                    level_tx_dx=NetworkLevel.TX,
                    substation_role=SubstationRole.INTERFACE_TX_DX,
                    operator_name='RTE',
                    operator_id='RTE',
                    voltage_kv_nominal_max=v_max,
                    voltage_kv_nominal_min=v_min,
                    voltage_classes=v_classes if v_classes else None,
                    voltage_quality_flag=VoltageQuality.RANGE_OR_CLASS_ONLY if voltage_str else VoltageQuality.UNKNOWN,
                    capacity_quality_flag=CapacityQuality.MISSING,
                    lon=geo.get('lon'),
                    lat=geo.get('lat'),
                    geom_quality_flag=GeomQuality.GOV_APPROX,
                    id_source=local_id,
                    last_update=datetime.now(),
                    licence_code='FR_LO',
                )
                nodes.append(node)
            except Exception as e:
                print(f"    Warning: Failed to transform RTE record: {e}")

        offset += limit
        if max_records and len(nodes) >= max_records:
            nodes = nodes[:max_records]
            break
        if len(records) < limit:
            break

    print(f"    -> {len(nodes)} RTE transmission substations")
    return nodes

# ============================================================================
# FRANCE: Agence ORE (Distribution Primary)
# ============================================================================

AGENCE_ORE_API = "https://opendata.agenceore.fr/api/explore/v2.1/catalog/datasets/postes-source/records"

def extract_france_ore(max_records: int = None) -> List[GridNode]:
    """Extract Agence ORE distribution primary substations."""
    nodes = []
    offset = 0
    limit = 100

    print("  [FR] Fetching Agence ORE distribution substations...")

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
            try:
                geo = r.get('geo_point_2d', {}) or {}
                operator = r.get('nom_grd', '') or r.get('grd', '') or 'UNKNOWN'
                local_id = f"{r.get('nom_ps', '')}_{geo.get('lat', '')}_{geo.get('lon', '')}"
                local_id = re.sub(r'[^a-zA-Z0-9_-]', '_', local_id)[:50]

                node = GridNode(
                    id_global=generate_global_id('FR', 'ORE', local_id),
                    country=Country.FR,
                    source_primary='AGENCE_ORE',
                    source_rank=SourceRank.TSO_DSO_OFFICIAL,
                    name=r.get('nom_ps', '') or f"PS {r.get('commune', '')}",
                    level_tx_dx=NetworkLevel.DX_PRIMARY,
                    substation_role=SubstationRole.INTERFACE_TX_DX,
                    operator_name=operator,
                    operator_id=operator,
                    voltage_kv_nominal_max=90.0,  # HTB/HTA typical
                    voltage_kv_nominal_min=20.0,
                    voltage_classes=['HV', 'MV'],
                    voltage_quality_flag=VoltageQuality.RANGE_OR_CLASS_ONLY,
                    capacity_quality_flag=CapacityQuality.MISSING,
                    lon=geo.get('lon'),
                    lat=geo.get('lat'),
                    geom_quality_flag=GeomQuality.TSO_DSO_EXACT,
                    id_source=local_id,
                    last_update=datetime.now(),
                    licence_code='FR_LO',
                    notes=f"Commune: {r.get('commune', '')}; Region: {r.get('region', '')}",
                )
                nodes.append(node)
            except Exception as e:
                print(f"    Warning: Failed to transform ORE record: {e}")

        offset += limit
        if max_records and len(nodes) >= max_records:
            nodes = nodes[:max_records]
            break
        if len(records) < limit:
            break

    print(f"    -> {len(nodes)} Agence ORE distribution substations")
    return nodes

# ============================================================================
# FRANCE: OpenStreetMap (Transmission with coordinates)
# ============================================================================

def extract_france_osm(max_records: int = None) -> List[GridNode]:
    """Extract French substations from OpenStreetMap (to get coordinates RTE doesn't provide)."""
    nodes = []

    print("  [FR] Fetching OpenStreetMap substations (for coordinates)...")

    # Overpass query for French substations
    query = """
    [out:json][timeout:180];
    area["ISO3166-1"="FR"]->.france;
    (
      node["power"="substation"](area.france);
      way["power"="substation"](area.france);
      relation["power"="substation"](area.france);
    );
    out center tags;
    """

    try:
        response = requests.post(OVERPASS_API, data={'data': query}, timeout=200)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"    Error fetching OSM data: {e}")
        return nodes

    elements = data.get('elements', [])

    for elem in elements:
        try:
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
            v_classes = []

            if voltage_str:
                voltages = re.findall(r'(\d+)', voltage_str)
                if voltages:
                    voltages = [float(v)/1000 if float(v) > 1000 else float(v) for v in voltages]
                    v_max = max(voltages)
                    v_min = min(voltages)
                    v_classes = list(set([classify_voltage(v) for v in voltages]))

            # Determine network level based on voltage or type
            substation_type = tags.get('substation', '')
            if v_max and v_max >= 63:
                level = NetworkLevel.TX
            elif substation_type == 'transmission':
                level = NetworkLevel.TX
            elif substation_type == 'distribution':
                level = NetworkLevel.DX_PRIMARY
            else:
                level = NetworkLevel.DX_PRIMARY

            # Determine role
            if v_max and v_max >= 200:
                role = SubstationRole.INTERFACE_TX_DX
            elif v_max and v_max >= 63:
                role = SubstationRole.ZONE_SUBSTATION
            else:
                role = SubstationRole.DISTRIBUTION_SUBSTATION

            name = tags.get('name', '') or tags.get('ref', '') or f"OSM_{elem['type']}_{elem['id']}"
            operator = tags.get('operator', '') or tags.get('owner', '')

            node = GridNode(
                id_global=generate_global_id('FR', 'OSM', f"{elem['type']}_{elem['id']}"),
                country=Country.FR,
                source_primary='OSM',
                source_rank=SourceRank.COMMUNITY,
                name=name,
                level_tx_dx=level,
                substation_role=role,
                operator_name=operator if operator else None,
                operator_id=None,
                voltage_kv_nominal_max=v_max,
                voltage_kv_nominal_min=v_min,
                voltage_classes=v_classes if v_classes else None,
                voltage_quality_flag=VoltageQuality.NUMERIC_EXACT if v_max else VoltageQuality.UNKNOWN,
                capacity_quality_flag=CapacityQuality.MISSING,
                lon=lon,
                lat=lat,
                geom_quality_flag=GeomQuality.DERIVED_FROM_OSM,
                id_source=f"{elem['type']}_{elem['id']}",
                last_update=datetime.now(),
                licence_code='ODbL',
                notes=f"OSM substation type: {substation_type}" if substation_type else None,
            )
            nodes.append(node)

            if max_records and len(nodes) >= max_records:
                break

        except Exception as e:
            print(f"    Warning: Failed to transform OSM element: {e}")

    print(f"    -> {len(nodes)} OSM substations (France)")
    return nodes

# ============================================================================
# AUSTRALIA: Geoscience Australia NEI (Transmission)
# ============================================================================

GA_NEI_BASE = "https://services.ga.gov.au/gis/rest/services/National_Electricity_Infrastructure/MapServer"
GA_NEI_SUBSTATIONS_LAYER = 0  # Updated 2024 - was layer 4

def extract_australia_ga_nei(max_records: int = None) -> List[GridNode]:
    """Extract GA NEI transmission substations."""
    nodes = []
    offset = 0
    count = 500

    print("  [AU] Fetching GA NEI transmission substations...")

    operator_map = {
        'New South Wales': 'TransGrid/Ausgrid',
        'Victoria': 'AusNet/AEMO',
        'Queensland': 'Powerlink',
        'South Australia': 'ElectraNet',
        'Tasmania': 'TasNetworks',
        'Western Australia': 'Western Power',
        'Northern Territory': 'Power and Water',
        'NSW': 'TransGrid/Ausgrid',
        'VIC': 'AusNet/AEMO',
        'QLD': 'Powerlink',
        'SA': 'ElectraNet',
        'TAS': 'TasNetworks',
        'WA': 'Western Power',
        'NT': 'Power and Water',
    }

    while True:
        try:
            url = f"{GA_NEI_BASE}/{GA_NEI_SUBSTATIONS_LAYER}/query"
            params = {
                'where': '1=1',
                'outFields': '*',
                'f': 'json',
                'resultOffset': offset,
                'resultRecordCount': count,
                'returnGeometry': 'true',
                'outSR': '4326',
            }
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"    Error fetching GA NEI data: {e}")
            break

        features = data.get('features', [])
        if not features:
            break

        for f in features:
            try:
                attrs = f.get('attributes', {})
                geom = f.get('geometry', {})

                # Handle field names (lowercase in new API)
                name = attrs.get('name') or attrs.get('NAME') or ''
                state = attrs.get('state') or attrs.get('STATE') or ''
                voltage_kv = attrs.get('voltagekv') or attrs.get('VOLTAGEKV')
                obj_id = attrs.get('objectid') or attrs.get('OBJECTID') or ''

                v_max, v_min = None, None
                v_classes = []
                v_quality = VoltageQuality.UNKNOWN

                if voltage_kv:
                    try:
                        v_max = float(voltage_kv)
                        v_min = v_max
                        v_classes = [classify_voltage(v_max)]
                        v_quality = VoltageQuality.NUMERIC_EXACT
                    except (ValueError, TypeError):
                        pass

                operator = operator_map.get(state, state)

                if v_max and v_max >= 200:
                    role = SubstationRole.INTERFACE_TX_DX
                elif v_max and v_max >= 66:
                    role = SubstationRole.ZONE_SUBSTATION
                else:
                    role = SubstationRole.UNKNOWN

                node = GridNode(
                    id_global=generate_global_id('AU', 'GA_NEI', str(obj_id)),
                    country=Country.AU,
                    source_primary='GA_NEI',
                    source_rank=SourceRank.GOVERNMENT_REPOSITORY,
                    name=name,
                    level_tx_dx=NetworkLevel.TX,
                    substation_role=role,
                    operator_name=operator,
                    operator_id=state,
                    voltage_kv_nominal_max=v_max,
                    voltage_kv_nominal_min=v_min,
                    voltage_classes=v_classes if v_classes else None,
                    voltage_quality_flag=v_quality,
                    capacity_quality_flag=CapacityQuality.MISSING,
                    lon=geom.get('x'),
                    lat=geom.get('y'),
                    geom_quality_flag=GeomQuality.TSO_DSO_EXACT,
                    id_source=str(obj_id),
                    last_update=datetime.now(),
                    licence_code='CC_BY_4.0',
                )
                nodes.append(node)
            except Exception as e:
                print(f"    Warning: Failed to transform GA NEI record: {e}")

        offset += count
        if max_records and len(nodes) >= max_records:
            nodes = nodes[:max_records]
            break
        if len(features) < count:
            break

    print(f"    -> {len(nodes)} GA NEI transmission substations")
    return nodes

# ============================================================================
# AUSTRALIA: OpenStreetMap (Distribution)
# ============================================================================

OVERPASS_API = "https://overpass-api.de/api/interpreter"

def extract_australia_osm(max_records: int = None) -> List[GridNode]:
    """Extract Australian substations from OpenStreetMap."""
    nodes = []

    print("  [AU] Fetching OpenStreetMap substations (this may take a minute)...")

    # Overpass query for Australian substations
    query = """
    [out:json][timeout:180];
    area["ISO3166-1"="AU"]->.australia;
    (
      node["power"="substation"](area.australia);
      way["power"="substation"](area.australia);
      relation["power"="substation"](area.australia);
    );
    out center tags;
    """

    try:
        response = requests.post(OVERPASS_API, data={'data': query}, timeout=200)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"    Error fetching OSM data: {e}")
        return nodes

    elements = data.get('elements', [])

    for elem in elements:
        try:
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
            v_classes = []

            if voltage_str:
                voltages = re.findall(r'(\d+)', voltage_str)
                if voltages:
                    voltages = [float(v)/1000 if float(v) > 1000 else float(v) for v in voltages]
                    v_max = max(voltages)
                    v_min = min(voltages)
                    v_classes = list(set([classify_voltage(v) for v in voltages]))

            # Determine network level
            substation_type = tags.get('substation', '')
            if v_max and v_max >= 66:
                level = NetworkLevel.TX
            elif substation_type in ['transmission', 'traction']:
                level = NetworkLevel.TX
            elif substation_type == 'distribution':
                level = NetworkLevel.DX_PRIMARY
            else:
                level = NetworkLevel.DX_PRIMARY  # Default for OSM substations

            # Determine role
            if v_max and v_max >= 200:
                role = SubstationRole.INTERFACE_TX_DX
            elif v_max and v_max >= 66:
                role = SubstationRole.ZONE_SUBSTATION
            else:
                role = SubstationRole.DISTRIBUTION_SUBSTATION

            name = tags.get('name', '') or tags.get('ref', '') or f"OSM_{elem['type']}_{elem['id']}"
            operator = tags.get('operator', '') or tags.get('owner', '')

            node = GridNode(
                id_global=generate_global_id('AU', 'OSM', f"{elem['type']}_{elem['id']}"),
                country=Country.AU,
                source_primary='OSM',
                source_rank=SourceRank.COMMUNITY,
                name=name,
                level_tx_dx=level,
                substation_role=role,
                operator_name=operator if operator else None,
                operator_id=None,
                voltage_kv_nominal_max=v_max,
                voltage_kv_nominal_min=v_min,
                voltage_classes=v_classes if v_classes else None,
                voltage_quality_flag=VoltageQuality.NUMERIC_EXACT if v_max else VoltageQuality.UNKNOWN,
                capacity_quality_flag=CapacityQuality.MISSING,
                lon=lon,
                lat=lat,
                geom_quality_flag=GeomQuality.DERIVED_FROM_OSM,
                id_source=f"{elem['type']}_{elem['id']}",
                last_update=datetime.now(),
                licence_code='ODbL',
                notes=f"OSM substation type: {substation_type}" if substation_type else None,
            )
            nodes.append(node)

            if max_records and len(nodes) >= max_records:
                break

        except Exception as e:
            print(f"    Warning: Failed to transform OSM element: {e}")

    print(f"    -> {len(nodes)} OSM substations")
    return nodes

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_full_pipeline(output_path: str = None, max_records_per_source: int = None) -> pd.DataFrame:
    """Run the full extraction pipeline for France and Australia."""

    print("=" * 70)
    print("GRID INFRASTRUCTURE PIPELINE - FULL EXTRACTION")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    all_nodes = []

    # FRANCE
    print("\n" + "-" * 70)
    print("FRANCE")
    print("-" * 70)

    # France Distribution (Agence ORE) - has coordinates
    france_dx = extract_france_ore(max_records=max_records_per_source)
    all_nodes.extend(france_dx)

    # France Transmission + additional Distribution (OSM) - has coordinates
    # Note: RTE official data lacks coordinates for security reasons, so we use OSM
    france_osm = extract_france_osm(max_records=max_records_per_source)
    all_nodes.extend(france_osm)

    # AUSTRALIA
    print("\n" + "-" * 70)
    print("AUSTRALIA")
    print("-" * 70)

    # Australia Transmission (GA NEI)
    australia_tx = extract_australia_ga_nei(max_records=max_records_per_source)
    all_nodes.extend(australia_tx)

    # Australia Distribution (OSM)
    australia_dx = extract_australia_osm(max_records=max_records_per_source)
    all_nodes.extend(australia_dx)

    # Convert to DataFrame
    print("\n" + "-" * 70)
    print("EXPORT")
    print("-" * 70)

    records = [n.to_dict() for n in all_nodes]
    df = pd.DataFrame(records)

    # Summary
    print(f"\nTotal records: {len(df)}")
    print("\nBy country:")
    print(df['country'].value_counts().to_string())
    print("\nBy source:")
    print(df['source_primary'].value_counts().to_string())
    print("\nBy network level:")
    print(df['level_tx_dx'].value_counts().to_string())

    # Export
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), 'full_output.csv')

    df.to_csv(output_path, index=False)
    print(f"\nExported to: {output_path}")

    # Also export GeoJSON
    geojson_path = output_path.replace('.csv', '.geojson')
    export_geojson(all_nodes, geojson_path)
    print(f"Exported to: {geojson_path}")

    print("\n" + "=" * 70)
    print(f"COMPLETE: {datetime.now().isoformat()}")
    print("=" * 70)

    return df

def export_geojson(nodes: List[GridNode], filepath: str):
    """Export nodes to GeoJSON."""
    features = []
    for node in nodes:
        if node.lon is not None and node.lat is not None:
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [node.lon, node.lat]
                },
                'properties': node.to_dict()
            }
            features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'features': features,
        'metadata': {
            'generated': datetime.now().isoformat(),
            'total_features': len(features)
        }
    }

    with open(filepath, 'w') as f:
        json.dump(geojson, f, indent=2)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Grid Infrastructure Pipeline')
    parser.add_argument('--output', '-o', default='full_output.csv', help='Output CSV path')
    parser.add_argument('--max-records', '-m', type=int, default=None, help='Max records per source (for testing)')

    args = parser.parse_args()

    df = run_full_pipeline(output_path=args.output, max_records_per_source=args.max_records)
