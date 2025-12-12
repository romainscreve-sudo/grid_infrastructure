"""
France ETL: RTE/ODRÉ - Transmission substations
Source: https://odre.opendatasoft.com/explore/dataset/postes-electriques-rte/
"""
import requests
import json
from datetime import datetime
from typing import List, Dict, Optional
import sys
sys.path.append('/home/claude/grid_infrastructure')

from common.schema import (
    GridNode, Country, SourceRank, NetworkLevel, SubstationRole,
    VoltageQuality, CapacityQuality, GeomQuality, generate_global_id, classify_voltage
)

RTE_ODRE_API = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/postes-electriques-rte/records"

def parse_voltage_levels(voltage_str: Optional[str]) -> tuple:
    """Parse RTE voltage level string (e.g., '400 kV / 225 kV') into min/max kV."""
    if not voltage_str:
        return None, None
    
    import re
    voltages = re.findall(r'(\d+)\s*kV', voltage_str, re.IGNORECASE)
    voltages = [float(v) for v in voltages]
    
    if not voltages:
        return None, None
    
    return max(voltages), min(voltages)

def determine_substation_role(type_str: Optional[str]) -> SubstationRole:
    """Map RTE site type to standardised role."""
    if not type_str:
        return SubstationRole.UNKNOWN
    
    type_lower = type_str.lower()
    if 'transformation' in type_lower:
        return SubstationRole.INTERFACE_TX_DX
    elif 'piquage' in type_lower:
        return SubstationRole.SWITCHING_STATION
    elif 'interconnexion' in type_lower:
        return SubstationRole.INTERFACE_TX_DX
    else:
        return SubstationRole.UNKNOWN

def fetch_rte_odre_data(limit: int = 1000, offset: int = 0) -> Dict:
    """Fetch data from RTE/ODRÉ API."""
    params = {
        'limit': limit,
        'offset': offset,
    }
    
    response = requests.get(RTE_ODRE_API, params=params)
    response.raise_for_status()
    return response.json()

def transform_rte_record(record: Dict) -> GridNode:
    """Transform a single RTE/ODRÉ record to GridNode."""
    fields = record
    
    # Extract coordinates (note: RTE degrades GPS for security)
    lon, lat = None, None
    if 'geo_point_2d' in fields and fields['geo_point_2d']:
        geo = fields['geo_point_2d']
        lon = geo.get('lon')
        lat = geo.get('lat')
    
    # Parse voltages
    voltage_str = fields.get('tension')
    v_max, v_min = parse_voltage_levels(voltage_str)
    
    # Build voltage classes
    v_classes = []
    if v_max:
        v_classes.append(classify_voltage(v_max))
    if v_min and v_min != v_max:
        v_classes.append(classify_voltage(v_min))
    
    # Generate ID
    local_id = fields.get('code_poste') or fields.get('nom_poste', '').replace(' ', '_')
    global_id = generate_global_id('FR', 'RTE_ODRE', local_id)
    
    return GridNode(
        id_global=global_id,
        country=Country.FR,
        source_primary='RTE_ODRE',
        source_rank=SourceRank.TSO_DSO_OFFICIAL,
        name=fields.get('nom_poste', ''),
        level_tx_dx=NetworkLevel.TX,
        substation_role=determine_substation_role(fields.get('type_poste')),
        operator_name='RTE',
        operator_id='RTE',
        voltage_kv_nominal_max=v_max,
        voltage_kv_nominal_min=v_min,
        voltage_classes=v_classes if v_classes else None,
        voltage_quality_flag=VoltageQuality.RANGE_OR_CLASS_ONLY if voltage_str else VoltageQuality.UNKNOWN,
        capacity_quality_flag=CapacityQuality.MISSING,
        lon=lon,
        lat=lat,
        geom_type='POINT',
        geom_quality_flag=GeomQuality.GOV_APPROX,  # RTE degrades coordinates
        id_rte_site=local_id,
        last_update_from_source=datetime.now(),
        licence_code='FR_LO',
    )

def extract_rte_odre(max_records: int = None) -> List[GridNode]:
    """Full ETL: extract all RTE/ODRÉ transmission substations."""
    nodes = []
    offset = 0
    limit = 100
    
    print("Fetching RTE/ODRÉ transmission substations...")
    
    while True:
        data = fetch_rte_odre_data(limit=limit, offset=offset)
        records = data.get('results', [])
        
        if not records:
            break
        
        for record in records:
            try:
                node = transform_rte_record(record)
                nodes.append(node)
            except Exception as e:
                print(f"  Warning: Failed to transform record: {e}")
        
        print(f"  Fetched {len(nodes)} records...")
        
        offset += limit
        
        if max_records and len(nodes) >= max_records:
            nodes = nodes[:max_records]
            break
        
        if len(records) < limit:
            break
    
    print(f"Total RTE/ODRÉ records: {len(nodes)}")
    return nodes

if __name__ == '__main__':
    nodes = extract_rte_odre(max_records=50)
    for n in nodes[:5]:
        print(f"{n.name}: {n.voltage_kv_nominal_max} kV @ ({n.lat}, {n.lon})")
