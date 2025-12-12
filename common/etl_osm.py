"""
OSM ETL: OpenStreetMap power infrastructure for QA/rubric
Uses Overpass API for queries. Applicable to both France and Australia.
"""
import requests
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import re
import sys
sys.path.append('/home/claude/grid_infrastructure')

from common.schema import (
    GridNode, Country, SourceRank, NetworkLevel, SubstationRole,
    VoltageQuality, CapacityQuality, GeomQuality, generate_global_id, classify_voltage
)

OVERPASS_API = "https://overpass-api.de/api/interpreter"

def build_substation_query(bbox: Tuple[float, float, float, float], 
                           country_code: str = None) -> str:
    """Build Overpass query for power substations within bbox."""
    south, west, north, east = bbox
    
    query = f"""
    [out:json][timeout:180];
    (
      node["power"="substation"]({south},{west},{north},{east});
      way["power"="substation"]({south},{west},{north},{east});
      relation["power"="substation"]({south},{west},{north},{east});
    );
    out center tags;
    """
    return query

def parse_osm_voltage(voltage_str: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Parse OSM voltage tag (e.g., '400000;225000' or '400 kV')."""
    if not voltage_str:
        return None, None
    
    # Handle semicolon-separated voltages (in Volts)
    if ';' in voltage_str:
        parts = voltage_str.split(';')
        voltages = []
        for p in parts:
            try:
                v = float(p.strip())
                if v > 1000:  # Likely in Volts
                    v = v / 1000
                voltages.append(v)
            except ValueError:
                continue
        if voltages:
            return max(voltages), min(voltages)
    
    # Handle single value
    try:
        v = float(voltage_str.replace('kV', '').replace('KV', '').strip())
        if v > 1000:  # Likely in Volts
            v = v / 1000
        return v, v
    except ValueError:
        pass
    
    # Handle 'kV' notation
    match = re.search(r'(\d+(?:\.\d+)?)\s*kV', voltage_str, re.IGNORECASE)
    if match:
        v = float(match.group(1))
        return v, v
    
    return None, None

def determine_osm_level(voltage_max: Optional[float], 
                        substation_type: Optional[str]) -> NetworkLevel:
    """Determine network level from OSM voltage and substation type."""
    sub_type = (substation_type or '').lower()
    
    if voltage_max and voltage_max >= 100:
        return NetworkLevel.TX
    elif sub_type in ['transmission', 'traction']:
        return NetworkLevel.TX
    elif sub_type in ['distribution', 'minor_distribution']:
        return NetworkLevel.DX_SECONDARY
    elif voltage_max and voltage_max >= 20:
        return NetworkLevel.DX_PRIMARY
    elif voltage_max and voltage_max < 20:
        return NetworkLevel.DX_SECONDARY
    else:
        return NetworkLevel.DX_PRIMARY  # Default assumption

def fetch_osm_substations(bbox: Tuple[float, float, float, float]) -> List[Dict]:
    """Fetch substations from Overpass API."""
    query = build_substation_query(bbox)
    
    response = requests.post(OVERPASS_API, data={'data': query}, timeout=300)
    response.raise_for_status()
    
    return response.json().get('elements', [])

def transform_osm_element(element: Dict, country: Country) -> GridNode:
    """Transform OSM element to GridNode."""
    tags = element.get('tags', {})
    
    # Extract coordinates
    if element['type'] == 'node':
        lon = element.get('lon')
        lat = element.get('lat')
    else:
        # For ways/relations, use center if available
        center = element.get('center', {})
        lon = center.get('lon')
        lat = center.get('lat')
    
    # Parse voltage
    voltage_str = tags.get('voltage')
    v_max, v_min = parse_osm_voltage(voltage_str)
    
    # Build voltage classes
    v_classes = []
    v_quality = VoltageQuality.UNKNOWN
    if v_max:
        v_classes.append(classify_voltage(v_max))
        if v_min and classify_voltage(v_min) not in v_classes:
            v_classes.append(classify_voltage(v_min))
        v_quality = VoltageQuality.NUMERIC_EXACT
    
    # Determine network level
    sub_type = tags.get('substation')
    level = determine_osm_level(v_max, sub_type)
    
    # Get name and operator
    name = tags.get('name', '') or tags.get('ref', '') or f"OSM_{element['id']}"
    operator = tags.get('operator', '')
    
    # Determine substation role
    if sub_type == 'transmission':
        role = SubstationRole.INTERFACE_TX_DX
    elif sub_type == 'distribution':
        role = SubstationRole.DISTRIBUTION_SUBSTATION
    elif v_max and v_max >= 100:
        role = SubstationRole.INTERFACE_TX_DX
    else:
        role = SubstationRole.UNKNOWN
    
    # Generate ID
    osm_id = str(element['id'])
    local_id = f"{element['type'][0]}{osm_id}"  # n123456, w123456, r123456
    global_id = generate_global_id(country.value, 'OSM', local_id)
    
    return GridNode(
        id_global=global_id,
        country=country,
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
        voltage_quality_flag=v_quality,
        capacity_quality_flag=CapacityQuality.MISSING,
        lon=lon,
        lat=lat,
        geom_type='POINT',
        geom_quality_flag=GeomQuality.DERIVED_FROM_OSM,
        id_osm=local_id,
        last_update_from_source=datetime.now(),
        licence_code='ODbL',
    )

# France bounding box (mainland + Corsica)
FRANCE_BBOX = (41.3, -5.2, 51.1, 9.6)

# Australia bounding box
AUSTRALIA_BBOX = (-44.0, 112.0, -10.0, 154.0)

def extract_osm_france(max_records: int = None) -> List[GridNode]:
    """Extract OSM power substations for France."""
    print("Fetching OSM substations for France (this may take a while)...")
    
    # Split France into regions to avoid timeout
    regions = [
        (48.0, -5.2, 51.1, 2.5),   # North-West
        (48.0, 2.5, 51.1, 9.6),    # North-East
        (45.0, -2.0, 48.0, 5.0),   # Center-West
        (45.0, 5.0, 48.0, 9.6),    # Center-East
        (41.3, -2.0, 45.0, 5.0),   # South-West
        (41.3, 5.0, 45.0, 9.6),    # South-East
    ]
    
    all_nodes = []
    seen_ids = set()
    
    for i, bbox in enumerate(regions):
        print(f"  Fetching region {i+1}/{len(regions)}...")
        try:
            elements = fetch_osm_substations(bbox)
            for elem in elements:
                if elem['id'] not in seen_ids:
                    seen_ids.add(elem['id'])
                    try:
                        node = transform_osm_element(elem, Country.FR)
                        all_nodes.append(node)
                    except Exception as e:
                        print(f"    Warning: Failed to transform element {elem['id']}: {e}")
            print(f"    Got {len(elements)} elements, total unique: {len(all_nodes)}")
        except Exception as e:
            print(f"    Error fetching region: {e}")
        
        if max_records and len(all_nodes) >= max_records:
            all_nodes = all_nodes[:max_records]
            break
    
    print(f"Total OSM France substations: {len(all_nodes)}")
    return all_nodes

def extract_osm_australia(max_records: int = None) -> List[GridNode]:
    """Extract OSM power substations for Australia."""
    print("Fetching OSM substations for Australia (this may take a while)...")
    
    # Split Australia into regions
    regions = [
        (-20.0, 112.0, -10.0, 135.0),  # North-West
        (-20.0, 135.0, -10.0, 154.0),  # North-East
        (-32.0, 112.0, -20.0, 130.0),  # Mid-West
        (-32.0, 130.0, -20.0, 154.0),  # Mid-East
        (-44.0, 112.0, -32.0, 130.0),  # South-West
        (-44.0, 130.0, -32.0, 154.0),  # South-East
    ]
    
    all_nodes = []
    seen_ids = set()
    
    for i, bbox in enumerate(regions):
        print(f"  Fetching region {i+1}/{len(regions)}...")
        try:
            elements = fetch_osm_substations(bbox)
            for elem in elements:
                if elem['id'] not in seen_ids:
                    seen_ids.add(elem['id'])
                    try:
                        node = transform_osm_element(elem, Country.AU)
                        all_nodes.append(node)
                    except Exception as e:
                        print(f"    Warning: Failed to transform element {elem['id']}: {e}")
            print(f"    Got {len(elements)} elements, total unique: {len(all_nodes)}")
        except Exception as e:
            print(f"    Error fetching region: {e}")
        
        if max_records and len(all_nodes) >= max_records:
            all_nodes = all_nodes[:max_records]
            break
    
    print(f"Total OSM Australia substations: {len(all_nodes)}")
    return all_nodes

if __name__ == '__main__':
    # Test with a small region (Paris area)
    test_bbox = (48.5, 2.0, 49.0, 2.8)
    print("Testing OSM fetch for Paris region...")
    elements = fetch_osm_substations(test_bbox)
    print(f"Found {len(elements)} substations")
    
    for elem in elements[:3]:
        node = transform_osm_element(elem, Country.FR)
        print(f"  {node.name}: {node.voltage_kv_nominal_max} kV @ ({node.lat}, {node.lon})")
