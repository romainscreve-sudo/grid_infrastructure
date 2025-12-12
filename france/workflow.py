"""
France Grid Infrastructure Workflow
Combines RTE/ODRÉ, Agence ORE, Enedis, and OSM data.
"""
import sys
import json
from datetime import datetime
from typing import List, Dict
import pandas as pd

sys.path.append('/home/claude/grid_infrastructure')

from common.schema import GridNode, NetworkLevel
from common.matching import (
    find_best_match, build_spatial_index, get_nearby_candidates,
    normalize_name, haversine_distance
)
from france.etl_rte_odre import extract_rte_odre
from france.etl_agence_ore import extract_agence_ore
from france.etl_enedis import extract_all_enedis
from common.etl_osm import extract_osm_france

class FranceGridWorkflow:
    """Orchestrates France grid infrastructure data pipeline."""
    
    def __init__(self):
        self.tx_nodes: List[GridNode] = []
        self.dx_primary_nodes: List[GridNode] = []
        self.dx_secondary_nodes: List[GridNode] = []
        self.osm_nodes: List[GridNode] = []
        self.unified_nodes: List[GridNode] = []
    
    def step1_extract_authoritative_data(self, max_records: int = None):
        """Step 1: Build authoritative Tx and Dx-primary geometry."""
        print("\n" + "="*60)
        print("STEP 1: Extract Authoritative Tx and Dx Data")
        print("="*60)
        
        # 1a. RTE/ODRÉ as Tx backbone
        print("\n1a. Extracting RTE/ODRÉ transmission substations...")
        self.tx_nodes = extract_rte_odre(max_records=max_records)
        
        # 1b. Agence ORE as national Dx-primary layer
        print("\n1b. Extracting Agence ORE distribution primary substations...")
        ore_nodes = extract_agence_ore(max_records=max_records)
        self.dx_primary_nodes.extend(ore_nodes)
        
        # 1c. Enedis for deeper Dx coverage
        print("\n1c. Extracting Enedis substations...")
        enedis_data = extract_all_enedis(max_per_type=max_records)
        
        # Reconcile Enedis postes-source with ORE (spatial join + name match)
        print("\n  Reconciling Enedis postes-source with Agence ORE...")
        ore_index = build_spatial_index([n.to_dict() for n in self.dx_primary_nodes])
        
        for enedis_node in enedis_data['postes_source']:
            if enedis_node.lon and enedis_node.lat:
                candidates = get_nearby_candidates(ore_index, enedis_node.lon, enedis_node.lat)
                match, result = find_best_match(enedis_node.to_dict(), candidates)
                
                if result.matched and result.score >= 0.7:
                    # Update existing ORE node with Enedis ID
                    for ore_node in self.dx_primary_nodes:
                        if ore_node.to_dict() == match:
                            ore_node.id_enedis_poste = enedis_node.id_enedis_poste
                            ore_node.match_status = 'MATCHED_ENEDIS'
                            break
                else:
                    # Add as new node (likely ELD area already covered by ORE)
                    enedis_node.match_status = 'ENEDIS_UNIQUE'
        
        # Add Enedis HTA/BT as Dx-secondary
        self.dx_secondary_nodes.extend(enedis_data['postes_electrique'])
        
        print(f"\n  Summary after Step 1:")
        print(f"    Tx nodes (RTE): {len(self.tx_nodes)}")
        print(f"    Dx-primary nodes (ORE+Enedis): {len(self.dx_primary_nodes)}")
        print(f"    Dx-secondary nodes (HTA/BT): {len(self.dx_secondary_nodes)}")
    
    def step2_attach_capacity_placeholder(self):
        """
        Step 2: Attach capacity from Caparéseau.
        NOTE: Caparéseau requires manual/semi-automated extraction.
        This is a placeholder showing the join logic.
        """
        print("\n" + "="*60)
        print("STEP 2: Capacity Enrichment (Caparéseau)")
        print("="*60)
        print("\nNOTE: Caparéseau data requires manual extraction from:")
        print("  https://www.services-rte.com/en/learn-more-about-our-services/consult-the-reception-capacity-of-the-grid-capareseau.html")
        print("\nThe workflow would:")
        print("  1. Load Caparéseau export (if available)")
        print("  2. Match by name + operator + distance")
        print("  3. Attach available_capacity_mw and reserved_capacity_mw")
        print("  4. Set capacity_quality_flag = INDICATIVE")
        
        # Placeholder for when Caparéseau data is available
        # capareseau_data = load_capareseau_export('capareseau_export.csv')
        # self._join_capareseau(capareseau_data)
    
    def step3_osm_qa_enrichment(self, max_records: int = None):
        """Step 3: Use OSM as rubric and QA layer."""
        print("\n" + "="*60)
        print("STEP 3: OSM QA and Enrichment")
        print("="*60)
        
        print("\nExtracting OSM substations for France...")
        self.osm_nodes = extract_osm_france(max_records=max_records)
        
        # Build OSM spatial index
        osm_index = build_spatial_index([n.to_dict() for n in self.osm_nodes])
        
        # Cross-check Tx nodes
        print("\nCross-checking Tx nodes with OSM...")
        tx_matched = 0
        tx_voltage_enriched = 0
        
        for node in self.tx_nodes:
            if node.lon and node.lat:
                candidates = get_nearby_candidates(osm_index, node.lon, node.lat)
                match, result = find_best_match(node.to_dict(), candidates, distance_threshold_m=2000)
                
                if result.matched:
                    tx_matched += 1
                    node.id_osm = match.get('id_osm')
                    
                    # Enrich voltage if OSM has better data
                    osm_v_max = match.get('voltage_kv_nominal_max')
                    if osm_v_max and not node.voltage_kv_nominal_max:
                        node.voltage_kv_nominal_max = osm_v_max
                        node.voltage_kv_nominal_min = match.get('voltage_kv_nominal_min')
                        tx_voltage_enriched += 1
                else:
                    node.review_flag = True
                    node.notes = "No OSM match found - verify coordinates"
        
        print(f"  Tx nodes matched to OSM: {tx_matched}/{len(self.tx_nodes)}")
        print(f"  Tx nodes voltage-enriched from OSM: {tx_voltage_enriched}")
        
        # Cross-check Dx-primary nodes
        print("\nCross-checking Dx-primary nodes with OSM...")
        dx_matched = 0
        
        for node in self.dx_primary_nodes:
            if node.lon and node.lat:
                candidates = get_nearby_candidates(osm_index, node.lon, node.lat)
                match, result = find_best_match(node.to_dict(), candidates, distance_threshold_m=1000)
                
                if result.matched:
                    dx_matched += 1
                    node.id_osm = match.get('id_osm')
        
        print(f"  Dx-primary nodes matched to OSM: {dx_matched}/{len(self.dx_primary_nodes)}")
    
    def build_unified_dataset(self) -> List[GridNode]:
        """Combine all nodes into unified dataset."""
        print("\n" + "="*60)
        print("Building Unified Dataset")
        print("="*60)
        
        self.unified_nodes = []
        self.unified_nodes.extend(self.tx_nodes)
        self.unified_nodes.extend(self.dx_primary_nodes)
        self.unified_nodes.extend(self.dx_secondary_nodes)
        
        print(f"\nUnified dataset contains {len(self.unified_nodes)} nodes:")
        print(f"  TX: {len([n for n in self.unified_nodes if n.level_tx_dx == NetworkLevel.TX])}")
        print(f"  DX_PRIMARY: {len([n for n in self.unified_nodes if n.level_tx_dx == NetworkLevel.DX_PRIMARY])}")
        print(f"  DX_SECONDARY: {len([n for n in self.unified_nodes if n.level_tx_dx == NetworkLevel.DX_SECONDARY])}")
        
        return self.unified_nodes
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export unified nodes to pandas DataFrame."""
        records = [n.to_dict() for n in self.unified_nodes]
        return pd.DataFrame(records)
    
    def export_to_geojson(self, filepath: str):
        """Export unified nodes to GeoJSON."""
        features = []
        for node in self.unified_nodes:
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
                'country': 'FR',
                'total_features': len(features)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"Exported {len(features)} features to {filepath}")
    
    def run_full_workflow(self, max_records: int = None, include_osm: bool = True):
        """Run the complete France workflow."""
        print("\n" + "="*60)
        print("FRANCE GRID INFRASTRUCTURE WORKFLOW")
        print("="*60)
        
        self.step1_extract_authoritative_data(max_records=max_records)
        self.step2_attach_capacity_placeholder()
        
        if include_osm:
            self.step3_osm_qa_enrichment(max_records=max_records)
        
        self.build_unified_dataset()
        
        return self.unified_nodes

def main():
    """Main entry point."""
    workflow = FranceGridWorkflow()
    
    # Run with limited records for testing
    nodes = workflow.run_full_workflow(max_records=100, include_osm=False)
    
    # Export results
    df = workflow.export_to_dataframe()
    df.to_csv('/home/claude/grid_infrastructure/france_grid_nodes.csv', index=False)
    print(f"\nExported {len(df)} records to france_grid_nodes.csv")
    
    workflow.export_to_geojson('/home/claude/grid_infrastructure/france_grid_nodes.geojson')

if __name__ == '__main__':
    main()
