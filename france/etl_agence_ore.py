"""
France ETL: Agence ORE - Distribution primary substations (all DSOs)
Source: https://opendata.agenceore.fr/explore/dataset/postes-source/
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

AGENCE_ORE_API = "https://opendata.agenceore.fr/api/explore/v2.1/catalog/datasets/postes-source/records"

def parse_ore_voltage_type(type_str: Optional[str]) -> tuple:
    """Parse Agence ORE poste type (HTB/HTA, HTA/HTA) to voltage estimates."""
    if not type_str:
        return None, None
    
    type_upper = type_str.upper()
    
    # HTB/HTA: typically 63-90 kV / 20 kV
    if 'HTB/HTA' in type_upper or 'HTB-HTA' in type_upper:
        return 90.0, 20.0  # Representative values
    
    # HTA/HTA: typically 20 kV / 20 kV (or different MV levels)
    if 'HTA/HTA' in type_upper or 'HTA-HTA' in type_upper:
        return 20.0, 20.0
    
    return None, None

def fetch_ore_data(limit: int = 1000, offset: int = 0) -> Dict:
    """Fetch data from Agence ORE API."""
    params = {
        'limit': limit,
        'offset': offset,
    }
    
    response = requests.get(AGENCE_ORE_API, params=params)
    response.raise_for_status()
    return response.json()

def transform_ore_record(record: Dict) -> GridNode:
    """Transform a single Agence ORE record to GridNode."""
    fields = record
    
    # Extract coordinates
    lon, lat = None, None
    if 'geo_point_2d' in fields and fields['geo_point_2d']:
        geo = fields['geo_point_2d']
        lon = geo.get('lon')
        lat = geo.get('lat')
    
    # Parse voltage type
    poste_type = fields.get('type_poste', '')
    v_max, v_min = parse_ore_voltage_type(poste_type)
    
    # Build voltage classes
    v_classes = []
    if v_max:
        v_classes.append(classify_voltage(v_max))
    if v_min and classify_voltage(v_min) not in v_classes:
        v_classes.append(classify_voltage(v_min))
    
    # Determine substation role
    role = SubstationRole.INTERFACE_TX_DX if 'HTB' in (poste_type or '').upper() else SubstationRole.DISTRIBUTION_SUBSTATION
    
    # Get operator
    operator = fields.get('grd', '') or fields.get('operateur', '') or 'UNKNOWN'
    
    # Generate ID
    local_id = fields.get('code_poste', '') or fields.get('identifiant', '') or str(hash(f"{fields.get('nom_poste', '')}_{lon}_{lat}"))
    global_id = generate_global_id('FR', 'AGENCE_ORE', local_id)
    
    return GridNode(
        id_global=global_id,
        country=Country.FR,
        source_primary='AGENCE_ORE',
        source_rank=SourceRank.TSO_DSO_OFFICIAL,
        name=fields.get('nom_poste', '') or fields.get('nom', ''),
        level_tx_dx=NetworkLevel.DX_PRIMARY,
        substation_role=role,
        operator_name=operator,
        operator_id=operator,
        voltage_kv_nominal_max=v_max,
        voltage_kv_nominal_min=v_min,
        voltage_classes=v_classes if v_classes else None,
        voltage_quality_flag=VoltageQuality.RANGE_OR_CLASS_ONLY,
        capacity_quality_flag=CapacityQuality.MISSING,
        lon=lon,
        lat=lat,
        geom_type='POINT',
        geom_quality_flag=GeomQuality.TSO_DSO_EXACT,
        id_ore_poste=local_id,
        last_update_from_source=datetime.now(),
        licence_code='FR_LO',
    )

def extract_agence_ore(max_records: int = None) -> List[GridNode]:
    """Full ETL: extract all Agence ORE distribution primary substations."""
    nodes = []
    offset = 0
    limit = 100
    
    print("Fetching Agence ORE distribution substations...")
    
    while True:
        data = fetch_ore_data(limit=limit, offset=offset)
        records = data.get('results', [])
        
        if not records:
            break
        
        for record in records:
            try:
                node = transform_ore_record(record)
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
    
    print(f"Total Agence ORE records: {len(nodes)}")
    return nodes

if __name__ == '__main__':
    nodes = extract_agence_ore(max_records=50)
    for n in nodes[:5]:
        print(f"{n.name} ({n.operator_name}): {n.voltage_kv_nominal_max}/{n.voltage_kv_nominal_min} kV")
