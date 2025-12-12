#!/usr/bin/env python3
"""
Grid Infrastructure Pipeline - Demo with Sample Data
Demonstrates the workflow using local sample data (no network required).
"""
import sys
import json
from datetime import datetime
import pandas as pd

sys.path.append('/home/claude/grid_infrastructure')

from common.schema import (
    GridNode, Country, SourceRank, NetworkLevel, SubstationRole,
    VoltageQuality, CapacityQuality, GeomQuality, generate_global_id, classify_voltage
)
from common.matching import (
    find_best_match, build_spatial_index, get_nearby_candidates,
    haversine_distance
)

def load_sample_france_rte():
    """Load France RTE sample data."""
    with open('/home/claude/grid_infrastructure/sample_data/france_rte_sample.json') as f:
        records = json.load(f)
    
    import re
    nodes = []
    for r in records:
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
        
        geo = r.get('geo_point_2d', {})
        
        node = GridNode(
            id_global=generate_global_id('FR', 'RTE_ODRE', r['id']),
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
            voltage_quality_flag=VoltageQuality.RANGE_OR_CLASS_ONLY,
            capacity_quality_flag=CapacityQuality.MISSING,
            lon=geo.get('lon'),
            lat=geo.get('lat'),
            geom_type='POINT',
            geom_quality_flag=GeomQuality.GOV_APPROX,
            id_rte_site=r['id'],
            last_update_from_source=datetime.now(),
            licence_code='FR_LO',
        )
        nodes.append(node)
    
    return nodes

def load_sample_australia_nei():
    """Load Australia GA NEI sample data."""
    with open('/home/claude/grid_infrastructure/sample_data/australia_nei_sample.json') as f:
        records = json.load(f)
    
    operator_map = {
        'NSW': 'TransGrid/Ausgrid',
        'VIC': 'AusNet/AEMO',
        'QLD': 'Powerlink',
        'SA': 'ElectraNet',
        'TAS': 'TasNetworks',
    }
    
    nodes = []
    for r in records:
        v_kv = r.get('VOLTAGEKV')
        v_classes = [classify_voltage(v_kv)] if v_kv else None
        
        if v_kv and v_kv >= 200:
            role = SubstationRole.INTERFACE_TX_DX
        elif v_kv and v_kv >= 66:
            role = SubstationRole.ZONE_SUBSTATION
        else:
            role = SubstationRole.UNKNOWN
        
        node = GridNode(
            id_global=generate_global_id('AU', 'GA_NEI', str(r['OBJECTID'])),
            country=Country.AU,
            source_primary='GA_NEI',
            source_rank=SourceRank.GOVERNMENT_REPOSITORY,
            name=r.get('NAME', ''),
            level_tx_dx=NetworkLevel.TX,
            substation_role=role,
            operator_name=operator_map.get(r.get('STATE'), r.get('STATE', '')),
            operator_id=r.get('STATE'),
            voltage_kv_nominal_max=v_kv,
            voltage_kv_nominal_min=v_kv,
            voltage_classes=v_classes,
            voltage_quality_flag=VoltageQuality.NUMERIC_EXACT if v_kv else VoltageQuality.UNKNOWN,
            capacity_quality_flag=CapacityQuality.MISSING,
            lon=r.get('x'),
            lat=r.get('y'),
            geom_type='POINT',
            geom_quality_flag=GeomQuality.TSO_DSO_EXACT,
            id_ga_nei=str(r['OBJECTID']),
            last_update_from_source=datetime.now(),
            licence_code='GA_COPYRIGHT',
        )
        nodes.append(node)
    
    return nodes

def export_to_csv(nodes, filepath):
    """Export nodes to CSV."""
    records = [n.to_dict() for n in nodes]
    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False)
    return df

def export_to_geojson(nodes, filepath):
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
    
    return geojson

def demonstrate_matching():
    """Demonstrate the matching algorithm."""
    print("\n" + "="*60)
    print("MATCHING ALGORITHM DEMONSTRATION")
    print("="*60)
    
    # Create two nodes to match
    node1 = {
        'name': 'SYDNEY WEST SUBSTATION',
        'lon': 150.7500,
        'lat': -33.7500,
        'voltage_kv_nominal_max': 330,
        'operator_name': 'TransGrid',
    }
    
    node2 = {
        'name': 'SYDNEY WEST',
        'lon': 150.7489,
        'lat': -33.7456,
        'voltage_kv_nominal_max': 330,
        'operator_name': 'TransGrid/Ausgrid',
    }
    
    from common.matching import match_nodes, normalize_name
    
    print(f"\nNode 1: {node1['name']}")
    print(f"  Normalized: '{normalize_name(node1['name'])}'")
    print(f"  Location: ({node1['lat']}, {node1['lon']})")
    
    print(f"\nNode 2: {node2['name']}")
    print(f"  Normalized: '{normalize_name(node2['name'])}'")
    print(f"  Location: ({node2['lat']}, {node2['lon']})")
    
    result = match_nodes(node1, node2)
    
    print(f"\nMatch Result:")
    print(f"  Matched: {result.matched}")
    print(f"  Score: {result.score:.3f}")
    print(f"  Match Type: {result.match_type}")
    print(f"  Distance: {result.distance_m:.0f}m" if result.distance_m else "  Distance: N/A")
    print(f"  Name Similarity: {result.name_sim:.3f}" if result.name_sim else "  Name Similarity: N/A")

def main():
    print("="*60)
    print("GRID INFRASTRUCTURE PIPELINE - DEMO")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60)
    
    # Load France sample data
    print("\n" + "-"*60)
    print("FRANCE: Loading sample RTE/ODRÉ data")
    print("-"*60)
    
    france_nodes = load_sample_france_rte()
    print(f"Loaded {len(france_nodes)} France Tx substations")
    
    # Summary by voltage
    print("\nFrance substations by max voltage:")
    voltage_counts = {}
    for n in france_nodes:
        v = n.voltage_kv_nominal_max or 'Unknown'
        voltage_counts[v] = voltage_counts.get(v, 0) + 1
    for v, count in sorted(voltage_counts.items(), key=lambda x: (0 if x[0]=='Unknown' else -x[0])):
        print(f"  {v} kV: {count}")
    
    # Export France
    fr_csv = export_to_csv(france_nodes, '/home/claude/grid_infrastructure/france_demo_output.csv')
    export_to_geojson(france_nodes, '/home/claude/grid_infrastructure/france_demo_output.geojson')
    print(f"\nExported to france_demo_output.csv and .geojson")
    
    # Load Australia sample data
    print("\n" + "-"*60)
    print("AUSTRALIA: Loading sample GA NEI data")
    print("-"*60)
    
    australia_nodes = load_sample_australia_nei()
    print(f"Loaded {len(australia_nodes)} Australia Tx substations")
    
    # Summary by state
    print("\nAustralia substations by state:")
    state_counts = {}
    for n in australia_nodes:
        s = n.operator_id or 'Unknown'
        state_counts[s] = state_counts.get(s, 0) + 1
    for s, count in sorted(state_counts.items()):
        print(f"  {s}: {count}")
    
    # Summary by voltage
    print("\nAustralia substations by voltage:")
    voltage_counts = {}
    for n in australia_nodes:
        v = n.voltage_kv_nominal_max or 'Unknown'
        voltage_counts[v] = voltage_counts.get(v, 0) + 1
    for v, count in sorted(voltage_counts.items(), key=lambda x: (0 if x[0]=='Unknown' else -x[0])):
        print(f"  {v} kV: {count}")
    
    # Export Australia
    au_csv = export_to_csv(australia_nodes, '/home/claude/grid_infrastructure/australia_demo_output.csv')
    export_to_geojson(australia_nodes, '/home/claude/grid_infrastructure/australia_demo_output.geojson')
    print(f"\nExported to australia_demo_output.csv and .geojson")
    
    # Demonstrate matching
    demonstrate_matching()
    
    # Show schema sample
    print("\n" + "-"*60)
    print("UNIFIED SCHEMA SAMPLE")
    print("-"*60)
    
    sample_node = france_nodes[0]
    print("\nSample node (first France Tx substation):")
    for k, v in sample_node.to_dict().items():
        if v is not None:
            print(f"  {k}: {v}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print(f"Finished: {datetime.now().isoformat()}")
    print("="*60)
    
    return france_nodes, australia_nodes

if __name__ == '__main__':
    france_nodes, australia_nodes = main()
