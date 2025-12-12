#!/usr/bin/env python3
"""
Grid Infrastructure Data Pipeline
Main runner for France and Australia workflows.
"""
import sys
import argparse
from datetime import datetime

sys.path.append('/home/claude/grid_infrastructure')

def run_france(max_records: int = None, include_osm: bool = False):
    """Run France workflow."""
    from france.workflow import FranceGridWorkflow
    
    workflow = FranceGridWorkflow()
    nodes = workflow.run_full_workflow(max_records=max_records, include_osm=include_osm)
    
    # Export
    df = workflow.export_to_dataframe()
    df.to_csv('/home/claude/grid_infrastructure/france_grid_nodes.csv', index=False)
    workflow.export_to_geojson('/home/claude/grid_infrastructure/france_grid_nodes.geojson')
    
    return nodes

def run_australia(max_records: int = None, include_osm: bool = False):
    """Run Australia workflow."""
    from australia.workflow import AustraliaGridWorkflow
    
    workflow = AustraliaGridWorkflow()
    nodes = workflow.run_full_workflow(max_records=max_records, include_osm=include_osm)
    
    # Export
    df = workflow.export_to_dataframe()
    df.to_csv('/home/claude/grid_infrastructure/australia_grid_nodes.csv', index=False)
    workflow.export_to_geojson('/home/claude/grid_infrastructure/australia_grid_nodes.geojson')
    
    return nodes

def main():
    parser = argparse.ArgumentParser(description='Grid Infrastructure Data Pipeline')
    parser.add_argument('--country', choices=['france', 'australia', 'both'], default='both',
                        help='Which country workflow to run')
    parser.add_argument('--max-records', type=int, default=None,
                        help='Maximum records per source (for testing)')
    parser.add_argument('--include-osm', action='store_true',
                        help='Include OSM QA enrichment (slower)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"GRID INFRASTRUCTURE DATA PIPELINE")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*60}")
    
    if args.country in ['france', 'both']:
        print("\n>>> Running France workflow...")
        france_nodes = run_france(max_records=args.max_records, include_osm=args.include_osm)
        print(f"\n>>> France complete: {len(france_nodes)} nodes")
    
    if args.country in ['australia', 'both']:
        print("\n>>> Running Australia workflow...")
        australia_nodes = run_australia(max_records=args.max_records, include_osm=args.include_osm)
        print(f"\n>>> Australia complete: {len(australia_nodes)} nodes")
    
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"Finished: {datetime.now().isoformat()}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
