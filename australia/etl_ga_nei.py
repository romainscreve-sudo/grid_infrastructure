"""
Australia ETL: Geoscience Australia - National Electricity Infrastructure
Source: https://services.ga.gov.au/gis/rest/services/National_Electricity_Infrastructure/MapServer
"""
import requests
from datetime import datetime
from typing import List, Dict, Optional
import sys
sys.path.append('/home/claude/grid_infrastructure')

from common.schema import (
    GridNode, Country, SourceRank, NetworkLevel, SubstationRole,
    VoltageQuality, CapacityQuality, GeomQuality, generate_global_id, classify_voltage
)

# GA NEI REST endpoints
GA_NEI_BASE = "https://services.ga.gov.au/gis/rest/services/National_Electricity_Infrastructure/MapServer"
GA_NEI_SUBSTATIONS_LAYER = 4  # Electricity Transmission Substations

def fetch_ga_nei_substations(offset: int = 0, count: int = 1000) -> Dict:
    """Fetch substations from GA NEI ArcGIS REST service."""
    url = f"{GA_NEI_BASE}/{GA_NEI_SUBSTATIONS_LAYER}/query"
    
    params = {
        'where': '1=1',
        'outFields': '*',
        'f': 'json',
        'resultOffset': offset,
        'resultRecordCount': count,
        'returnGeometry': 'true',
        'outSR': '4326',  # WGS84
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def transform_ga_nei_record(feature: Dict) -> GridNode:
    """Transform a GA NEI feature to GridNode."""
    attrs = feature.get('attributes', {})
    geom = feature.get('geometry', {})
    
    # Extract coordinates
    lon = geom.get('x')
    lat = geom.get('y')
    
    # Extract voltage (GA NEI has VOLTAGEKV field)
    voltage_kv = attrs.get('VOLTAGEKV') or attrs.get('VOLTAGE_KV') or attrs.get('voltage_kv')
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
    
    # Determine operator from NAME or STATE
    name = attrs.get('NAME', '') or attrs.get('SUBSTATIONNAME', '') or ''
    state = attrs.get('STATE', '') or attrs.get('STATEID', '') or ''
    
    # Map state to likely operator
    operator_map = {
        'NSW': 'TransGrid/Ausgrid',
        'VIC': 'AusNet/AEMO',
        'QLD': 'Powerlink',
        'SA': 'ElectraNet',
        'TAS': 'TasNetworks',
        'WA': 'Western Power',
        'NT': 'Power and Water',
    }
    operator = operator_map.get(state, state)
    
    # Generate ID
    local_id = str(attrs.get('OBJECTID', '') or attrs.get('FID', '') or hash(f"{name}_{lon}_{lat}"))
    global_id = generate_global_id('AU', 'GA_NEI', local_id)
    
    # Determine substation role based on voltage
    if v_max and v_max >= 200:
        role = SubstationRole.INTERFACE_TX_DX
    elif v_max and v_max >= 66:
        role = SubstationRole.ZONE_SUBSTATION
    else:
        role = SubstationRole.UNKNOWN
    
    return GridNode(
        id_global=global_id,
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
        lon=lon,
        lat=lat,
        geom_type='POINT',
        geom_quality_flag=GeomQuality.TSO_DSO_EXACT,
        id_ga_nei=local_id,
        last_update_from_source=datetime.now(),
        licence_code='GA_COPYRIGHT',
    )

def extract_ga_nei(max_records: int = None) -> List[GridNode]:
    """Full ETL: extract all GA NEI transmission substations."""
    nodes = []
    offset = 0
    count = 500
    
    print("Fetching GA NEI transmission substations...")
    
    while True:
        data = fetch_ga_nei_substations(offset=offset, count=count)
        features = data.get('features', [])
        
        if not features:
            break
        
        for feature in features:
            try:
                node = transform_ga_nei_record(feature)
                nodes.append(node)
            except Exception as e:
                print(f"  Warning: Failed to transform feature: {e}")
        
        print(f"  Fetched {len(nodes)} substations...")
        
        offset += count
        
        if max_records and len(nodes) >= max_records:
            nodes = nodes[:max_records]
            break
        
        if len(features) < count:
            break
    
    print(f"Total GA NEI substations: {len(nodes)}")
    return nodes

def fetch_ga_nei_lines(offset: int = 0, count: int = 1000) -> Dict:
    """Fetch transmission lines from GA NEI (Layer 3)."""
    url = f"{GA_NEI_BASE}/3/query"  # Transmission lines layer
    
    params = {
        'where': '1=1',
        'outFields': '*',
        'f': 'json',
        'resultOffset': offset,
        'resultRecordCount': count,
        'returnGeometry': 'true',
        'outSR': '4326',
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def extract_ga_nei_lines(max_records: int = None) -> List[Dict]:
    """Extract transmission lines for topology building."""
    lines = []
    offset = 0
    count = 500
    
    print("Fetching GA NEI transmission lines...")
    
    while True:
        data = fetch_ga_nei_lines(offset=offset, count=count)
        features = data.get('features', [])
        
        if not features:
            break
        
        for feature in features:
            attrs = feature.get('attributes', {})
            geom = feature.get('geometry', {})
            
            lines.append({
                'id': attrs.get('OBJECTID'),
                'name': attrs.get('NAME', ''),
                'voltage_kv': attrs.get('VOLTAGEKV'),
                'state': attrs.get('STATE'),
                'geometry': geom,
            })
        
        print(f"  Fetched {len(lines)} lines...")
        
        offset += count
        
        if max_records and len(lines) >= max_records:
            lines = lines[:max_records]
            break
        
        if len(features) < count:
            break
    
    print(f"Total GA NEI lines: {len(lines)}")
    return lines

if __name__ == '__main__':
    nodes = extract_ga_nei(max_records=50)
    for n in nodes[:5]:
        print(f"{n.name} ({n.operator_name}): {n.voltage_kv_nominal_max} kV @ ({n.lat:.4f}, {n.lon:.4f})")
