"""
Australia Grid Infrastructure Workflow
Combines GA NEI, Rosetta (placeholder), MapStand (placeholder), and OSM data.
"""
import sys
import json
from datetime import datetime
from typing import List, Dict
import pandas as pd

sys.path.append('/home/claude/grid_infrastructure')

from common.schema import GridNode, NetworkLevel, SourceRank
from common.matching import (
    find_best_match, build_spatial_index, get_nearby_candidates,
    normalize_name, haversine_distance
)
from australia.etl_ga_nei import extract_ga_nei, extract_ga_nei_lines
from common.etl_osm import extract_osm_australia

class AustraliaGridWorkflow:
    """Orchestrates Australia grid infrastructure data pipeline."""
    
    def __init__(self):
        self.tx_nodes: List[GridNode] = []
        self.dx_primary_nodes: List[GridNode] = []
        self.osm_nodes: List[GridNode] = []
        self.unified_nodes: List[GridNode] = []
        self.tx_lines: List[Dict] = []
    
    def step1_extract_ga_nei(self, max_records: int = None):
        """Step 1: Build authoritative Tx layer from GA NEI."""
        print("\n" + "="*60)
        print("STEP 1: Extract GA NEI Transmission Network")
        print("="*60)
        
        # 1a. Extract substations
        print("\n1a. Extracting GA NEI transmission substations...")
        self.tx_nodes = extract_ga_nei(max_records=max_records)
        
        # 1b. Extract lines for topology (optional)
        print("\n1b. Extracting GA NEI transmission lines...")
        self.tx_lines = extract_ga_nei_lines(max_records=max_records)
        
        print(f"\n  Summary after Step 1:")
        print(f"    Tx substations: {len(self.tx_nodes)}")
        print(f"    Tx lines: {len(self.tx_lines)}")
    
    def step2_rosetta_placeholder(self):
        """
        Step 2: Add capacity/headroom and Dx detail from Rosetta.
        NOTE: Rosetta requires commercial API access.
        This is a placeholder showing the intended logic.
        """
        print("\n" + "="*60)
        print("STEP 2: Rosetta Network Map Integration (Placeholder)")
        print("="*60)
        print("\nNOTE: Rosetta requires commercial license from:")
        print("  https://renewables.networkmap.energy")
        print("\nThe workflow would:")
        print("  1. Authenticate with Rosetta API")
        print("  2. Fetch zone substations and Dx interface substations")
        print("  3. Fetch 'Available Distribution Capacity (MVA)' layer")
        print("  4. Match Rosetta nodes to GA NEI by name/operator/geometry")
        print("  5. Add Dx-primary nodes not in GA NEI")
        print("  6. Attach available_capacity_mva to matched nodes")
        
        # Example of what the integration would look like:
        """
        rosetta_nodes = fetch_rosetta_substations(api_key)
        rosetta_capacity = fetch_rosetta_capacity_layer(api_key)
        
        # Build GA NEI index for matching
        nei_index = build_spatial_index([n.to_dict() for n in self.tx_nodes])
        
        for rosetta_node in rosetta_nodes:
            if rosetta_node.lon and rosetta_node.lat:
                candidates = get_nearby_candidates(nei_index, rosetta_node.lon, rosetta_node.lat)
                match, result = find_best_match(rosetta_node.to_dict(), candidates)
                
                if result.matched:
                    # Merge attributes
                    for nei_node in self.tx_nodes:
                        if nei_node.to_dict() == match:
                            nei_node.id_rosetta = rosetta_node.id_rosetta
                            nei_node.available_capacity_mva = rosetta_node.available_capacity_mva
                            break
                else:
                    # Add as new Dx-primary node
                    self.dx_primary_nodes.append(rosetta_node)
        """
    
    def step3_dnsp_portals_placeholder(self):
        """
        Step 3: Ingest other DNSP-specific open maps.
        NOTE: These vary by operator and availability.
        """
        print("\n" + "="*60)
        print("STEP 3: DNSP Portal Integration (Placeholder)")
        print("="*60)
        print("\nAvailable DNSP open data portals:")
        print("  - CitiPower/Powercor: https://www.powercor.com.au/network-planning-and-projects/network-data/")
        print("  - Western Power: Network planning maps")
        print("  - SA Power Networks: Network capacity maps")
        print("  - TasNetworks: Planning data")
        print("\nThese can add zone substation load/constraint data to complement GA NEI.")
    
    def step4_mapstand_placeholder(self):
        """
        Step 4: Use MapStand for context and gap-filling.
        NOTE: MapStand requires commercial API access.
        """
        print("\n" + "="*60)
        print("STEP 4: MapStand Integration (Placeholder)")
        print("="*60)
        print("\nNOTE: MapStand requires commercial license from:")
        print("  https://mapstand.com")
        print("\nMapStand would be used for:")
        print("  - Filling obvious gaps compared to GA NEI")
        print("  - Cross-validating line routes and station names")
        print("  - Global context (lower source_rank than GA NEI)")
    
    def step5_osm_qa_enrichment(self, max_records: int = None):
        """Step 5: Use OSM as QA layer."""
        print("\n" + "="*60)
        print("STEP 5: OSM QA and Enrichment")
        print("="*60)
        
        print("\nExtracting OSM substations for Australia...")
        self.osm_nodes = extract_osm_australia(max_records=max_records)
        
        # Build OSM spatial index
        osm_index = build_spatial_index([n.to_dict() for n in self.osm_nodes])
        
        # Cross-check Tx nodes
        print("\nCross-checking Tx nodes with OSM...")
        tx_matched = 0
        tx_flagged = 0
        
        for node in self.tx_nodes:
            if node.lon and node.lat:
                candidates = get_nearby_candidates(osm_index, node.lon, node.lat)
                match, result = find_best_match(node.to_dict(), candidates, distance_threshold_m=2000)
                
                if result.matched:
                    tx_matched += 1
                    node.id_osm = match.get('id_osm')
                    
                    # Enrich voltage if OSM has data and GA NEI doesn't
                    osm_v_max = match.get('voltage_kv_nominal_max')
                    if osm_v_max and not node.voltage_kv_nominal_max:
                        node.voltage_kv_nominal_max = osm_v_max
                        node.voltage_kv_nominal_min = match.get('voltage_kv_nominal_min')
                else:
                    node.review_flag = True
                    node.notes = "No OSM match found - verify coordinates"
                    tx_flagged += 1
        
        print(f"  Tx nodes matched to OSM: {tx_matched}/{len(self.tx_nodes)}")
        print(f"  Tx nodes flagged for review: {tx_flagged}")
    
    def build_unified_dataset(self) -> List[GridNode]:
        """Combine all nodes into unified dataset."""
        print("\n" + "="*60)
        print("Building Unified Dataset")
        print("="*60)
        
        self.unified_nodes = []
        self.unified_nodes.extend(self.tx_nodes)
        self.unified_nodes.extend(self.dx_primary_nodes)
        
        print(f"\nUnified dataset contains {len(self.unified_nodes)} nodes:")
        print(f"  TX: {len([n for n in self.unified_nodes if n.level_tx_dx == NetworkLevel.TX])}")
        print(f"  DX_PRIMARY: {len([n for n in self.unified_nodes if n.level_tx_dx == NetworkLevel.DX_PRIMARY])}")
        
        # Summary by state
        states = {}
        for node in self.unified_nodes:
            state = node.operator_id or 'Unknown'
            states[state] = states.get(state, 0) + 1
        
        print("\n  By state/operator:")
        for state, count in sorted(states.items()):
            print(f"    {state}: {count}")
        
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
                'country': 'AU',
                'total_features': len(features)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"Exported {len(features)} features to {filepath}")
    
    def run_full_workflow(self, max_records: int = None, include_osm: bool = True):
        """Run the complete Australia workflow."""
        print("\n" + "="*60)
        print("AUSTRALIA GRID INFRASTRUCTURE WORKFLOW")
        print("="*60)
        
        self.step1_extract_ga_nei(max_records=max_records)
        self.step2_rosetta_placeholder()
        self.step3_dnsp_portals_placeholder()
        self.step4_mapstand_placeholder()
        
        if include_osm:
            self.step5_osm_qa_enrichment(max_records=max_records)
        
        self.build_unified_dataset()
        
        return self.unified_nodes

def main():
    """Main entry point."""
    workflow = AustraliaGridWorkflow()
    
    # Run with limited records for testing
    nodes = workflow.run_full_workflow(max_records=100, include_osm=False)
    
    # Export results
    df = workflow.export_to_dataframe()
    df.to_csv('/home/claude/grid_infrastructure/australia_grid_nodes.csv', index=False)
    print(f"\nExported {len(df)} records to australia_grid_nodes.csv")
    
    workflow.export_to_geojson('/home/claude/grid_infrastructure/australia_grid_nodes.geojson')

if __name__ == '__main__':
    main()
