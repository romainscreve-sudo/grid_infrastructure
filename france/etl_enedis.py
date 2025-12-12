"""
France ETL: Enedis - Distribution substations (Enedis concession only)
Sources:
- postes-source (HTB/HTA, HTA/HTA): https://data.enedis.fr/explore/dataset/poste-source/
- poste-electrique (HTA/BT): https://data.enedis.fr/explore/dataset/poste-electrique/
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

ENEDIS_POSTE_SOURCE_API = "https://data.enedis.fr/api/explore/v2.1/catalog/datasets/poste-source/records"
ENEDIS_POSTE_ELECTRIQUE_API = "https://data.enedis.fr/api/explore/v2.1/catalog/datasets/poste-electrique/records"

def fetch_enedis_data(api_url: str, limit: int = 1000, offset: int = 0) -> Dict:
    """Fetch data from Enedis API."""
    params = {
        'limit': limit,
        'offset': offset,
    }
    
    response = requests.get(api_url, params=params)
    response.raise_for_status()
    return response.json()

def transform_poste_source_record(record: Dict) -> GridNode:
    """Transform Enedis poste-source (HTB/HTA, HTA/HTA) record."""
    fields = record
    
    lon, lat = None, None
    if 'geo_point_2d' in fields and fields['geo_point_2d']:
        geo = fields['geo_point_2d']
        lon = geo.get('lon')
        lat = geo.get('lat')
    
    # Determine voltage based on type
    poste_type = fields.get('type_poste', '') or ''
    if 'HTB' in poste_type.upper():
        v_max, v_min = 90.0, 20.0
        role = SubstationRole.INTERFACE_TX_DX
    else:
        v_max, v_min = 20.0, 20.0
        role = SubstationRole.DISTRIBUTION_SUBSTATION
    
    v_classes = [classify_voltage(v_max)]
    if v_min and classify_voltage(v_min) not in v_classes:
        v_classes.append(classify_voltage(v_min))
    
    local_id = fields.get('code_poste', '') or fields.get('identifiant_poste', '') or str(hash(f"{fields.get('nom_poste', '')}_{lon}_{lat}"))
    global_id = generate_global_id('FR', 'ENEDIS_PS', local_id)
    
    return GridNode(
        id_global=global_id,
        country=Country.FR,
        source_primary='ENEDIS',
        source_rank=SourceRank.TSO_DSO_OFFICIAL,
        name=fields.get('nom_poste', '') or fields.get('nom', ''),
        level_tx_dx=NetworkLevel.DX_PRIMARY,
        substation_role=role,
        operator_name='ENEDIS',
        operator_id='ENEDIS',
        voltage_kv_nominal_max=v_max,
        voltage_kv_nominal_min=v_min,
        voltage_classes=v_classes,
        voltage_quality_flag=VoltageQuality.RANGE_OR_CLASS_ONLY,
        capacity_quality_flag=CapacityQuality.MISSING,
        lon=lon,
        lat=lat,
        geom_type='POINT',
        geom_quality_flag=GeomQuality.TSO_DSO_EXACT,
        id_enedis_poste=local_id,
        last_update_from_source=datetime.now(),
        licence_code='FR_LO',
    )

def transform_poste_electrique_record(record: Dict) -> GridNode:
    """Transform Enedis poste-electrique (HTA/BT) record."""
    fields = record
    
    lon, lat = None, None
    if 'geo_point_2d' in fields and fields['geo_point_2d']:
        geo = fields['geo_point_2d']
        lon = geo.get('lon')
        lat = geo.get('lat')
    
    # HTA/BT transformers: typically 20kV / 0.4kV
    v_max, v_min = 20.0, 0.4
    v_classes = ['MV', 'LV']
    
    local_id = fields.get('identifiant', '') or fields.get('code_poste', '') or str(hash(f"{lon}_{lat}"))
    global_id = generate_global_id('FR', 'ENEDIS_PE', local_id)
    
    return GridNode(
        id_global=global_id,
        country=Country.FR,
        source_primary='ENEDIS',
        source_rank=SourceRank.TSO_DSO_OFFICIAL,
        name=fields.get('nom', '') or f"Poste HTA/BT {local_id[:8]}",
        level_tx_dx=NetworkLevel.DX_SECONDARY,
        substation_role=SubstationRole.DISTRIBUTION_SUBSTATION,
        operator_name='ENEDIS',
        operator_id='ENEDIS',
        voltage_kv_nominal_max=v_max,
        voltage_kv_nominal_min=v_min,
        voltage_classes=v_classes,
        voltage_quality_flag=VoltageQuality.RANGE_OR_CLASS_ONLY,
        capacity_quality_flag=CapacityQuality.MISSING,
        lon=lon,
        lat=lat,
        geom_type='POINT',
        geom_quality_flag=GeomQuality.TSO_DSO_EXACT,
        id_enedis_poste=local_id,
        last_update_from_source=datetime.now(),
        licence_code='FR_LO',
    )

def extract_enedis_poste_source(max_records: int = None) -> List[GridNode]:
    """Extract Enedis postes-source (HTB/HTA, HTA/HTA)."""
    nodes = []
    offset = 0
    limit = 100
    
    print("Fetching Enedis postes-source...")
    
    while True:
        data = fetch_enedis_data(ENEDIS_POSTE_SOURCE_API, limit=limit, offset=offset)
        records = data.get('results', [])
        
        if not records:
            break
        
        for record in records:
            try:
                node = transform_poste_source_record(record)
                nodes.append(node)
            except Exception as e:
                print(f"  Warning: Failed to transform record: {e}")
        
        print(f"  Fetched {len(nodes)} postes-source...")
        
        offset += limit
        
        if max_records and len(nodes) >= max_records:
            nodes = nodes[:max_records]
            break
        
        if len(records) < limit:
            break
    
    print(f"Total Enedis postes-source: {len(nodes)}")
    return nodes

def extract_enedis_poste_electrique(max_records: int = None) -> List[GridNode]:
    """Extract Enedis postes-electrique (HTA/BT)."""
    nodes = []
    offset = 0
    limit = 100
    
    print("Fetching Enedis postes-electrique (HTA/BT)...")
    
    while True:
        data = fetch_enedis_data(ENEDIS_POSTE_ELECTRIQUE_API, limit=limit, offset=offset)
        records = data.get('results', [])
        
        if not records:
            break
        
        for record in records:
            try:
                node = transform_poste_electrique_record(record)
                nodes.append(node)
            except Exception as e:
                print(f"  Warning: Failed to transform record: {e}")
        
        print(f"  Fetched {len(nodes)} postes-electrique...")
        
        offset += limit
        
        if max_records and len(nodes) >= max_records:
            nodes = nodes[:max_records]
            break
        
        if len(records) < limit:
            break
    
    print(f"Total Enedis postes-electrique: {len(nodes)}")
    return nodes

def extract_all_enedis(max_per_type: int = None) -> Dict[str, List[GridNode]]:
    """Extract all Enedis substation types."""
    return {
        'postes_source': extract_enedis_poste_source(max_records=max_per_type),
        'postes_electrique': extract_enedis_poste_electrique(max_records=max_per_type),
    }

if __name__ == '__main__':
    data = extract_all_enedis(max_per_type=20)
    print(f"\nPostes-source: {len(data['postes_source'])}")
    print(f"Postes-electrique: {len(data['postes_electrique'])}")
