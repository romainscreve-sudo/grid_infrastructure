# Grid Infrastructure Data Pipeline

A unified ETL pipeline for harmonising electricity grid infrastructure data from France and Australia.

## Overview

This pipeline implements the workflow described in the assessment document, creating a standardised schema for grid nodes (substations, connection points) from multiple authoritative sources:

**France:**
- RTE/ODRÉ (Transmission substations)
- Agence ORE (Distribution primary substations - all DSOs)
- Enedis (Distribution substations - Enedis concession)
- Caparéseau (Hosting capacity - placeholder)
- OpenStreetMap (QA/enrichment layer)

**Australia:**
- Geoscience Australia NEI (Transmission substations)
- Rosetta Network Map (Capacity & Dx - placeholder, commercial)
- MapStand (Context/QA - placeholder, commercial)
- OpenStreetMap (QA/enrichment layer)

## Directory Structure

```
grid_infrastructure/
├── common/
│   ├── __init__.py
│   ├── schema.py           # Unified GridNode schema
│   ├── matching.py         # Entity matching utilities
│   └── etl_osm.py          # OSM/FLOSM extraction
├── france/
│   ├── __init__.py
│   ├── etl_rte_odre.py     # RTE transmission substations
│   ├── etl_agence_ore.py   # Agence ORE Dx primary
│   ├── etl_enedis.py       # Enedis Dx substations
│   └── workflow.py         # France orchestration
├── australia/
│   ├── __init__.py
│   ├── etl_ga_nei.py       # GA NEI transmission
│   └── workflow.py         # Australia orchestration
├── sample_data/
│   ├── france_rte_sample.json
│   └── australia_nei_sample.json
├── run_pipeline.py         # Main runner (requires network)
├── demo_pipeline.py        # Demo with sample data (no network)
└── README.md
```

## Unified Schema

All sources are mapped to a common `GridNode` dataclass with fields for:

### Core Identification
- `id_global`: Unique ID (e.g., `FR_RTE_ODRE_PS123`)
- `country`: FR or AU
- `source_primary`: Source name (RTE_ODRE, GA_NEI, etc.)
- `source_rank`: 1=TSO/DSO, 2=Government, 3=Commercial, 4=Community

### Network Role
- `level_tx_dx`: TX, DX_PRIMARY, DX_SECONDARY
- `substation_role`: INTERFACE_TX_DX, ZONE_SUBSTATION, etc.
- `operator_name`, `operator_id`

### Voltage
- `voltage_kv_nominal_max`, `voltage_kv_nominal_min`
- `voltage_classes`: EHV, HV, MV, LV
- `voltage_quality_flag`: NUMERIC_EXACT, RANGE_OR_CLASS_ONLY, UNKNOWN

### Capacity
- `available_capacity_mw`, `available_capacity_mva`
- `reserved_capacity_mw`
- `capacity_quality_flag`: TSO_CONFIRMED, INDICATIVE, MISSING

### Geometry
- `lon`, `lat` (WGS84)
- `geom_quality_flag`: TSO_DSO_EXACT, GOV_APPROX, DERIVED_FROM_OSM

### Linkage
- `id_rte_site`, `id_enedis_poste`, `id_ore_poste`, `id_ga_nei`, `id_osm`, etc.

## Installation

```bash
pip install pandas requests
```

## Usage

### Demo (no network required)
```bash
python demo_pipeline.py
```

### Full Pipeline (requires network access to APIs)
```bash
# Run both countries
python run_pipeline.py --country both --max-records 100

# Run specific country
python run_pipeline.py --country france --max-records 500

# Include OSM QA (slower)
python run_pipeline.py --country australia --include-osm
```

### Programmatic Usage
```python
from france.workflow import FranceGridWorkflow

workflow = FranceGridWorkflow()
nodes = workflow.run_full_workflow(max_records=1000)

df = workflow.export_to_dataframe()
workflow.export_to_geojson('france_grid.geojson')
```

## Matching Algorithm

The pipeline uses hierarchical entity matching:

1. **Exact Name Match** (score=1.0)
2. **Name Similarity + Distance + Voltage** (score=0.4-0.8)
3. **Operator + Distance** (score=0.3-0.6)
4. **Distance Only** (score=0.0-0.3, weak match)

Match results include:
- `matched`: bool
- `score`: 0.0-1.0
- `match_type`: EXACT_NAME, NAME_DISTANCE_VOLTAGE, etc.
- `distance_m`: Distance in meters
- `name_sim`: Name similarity ratio

## Data Source Notes

### France

**RTE/ODRÉ**
- API: `https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/postes-electriques-rte/records`
- Coverage: National transmission substations
- GPS: Degraded for security
- Licence: Licence Ouverte v2

**Agence ORE**
- API: `https://opendata.agenceore.fr/api/explore/v2.1/catalog/datasets/postes-source/records`
- Coverage: All DSOs (Enedis + ELDs)
- Licence: Licence Ouverte

**Enedis**
- APIs: `https://data.enedis.fr/api/explore/v2.1/catalog/datasets/poste-source/records`
- Coverage: Enedis concession only, but includes HTA/BT substations
- Licence: Licence Ouverte

**Caparéseau**
- Portal: `https://www.services-rte.com/en/learn-more-about-our-services/consult-the-reception-capacity-of-the-grid-capareseau.html`
- Coverage: Indicative hosting capacity at Tx and Dx primary
- Access: Manual/semi-automated extraction required

### Australia

**Geoscience Australia NEI**
- API: `https://services.ga.gov.au/gis/rest/services/National_Electricity_Infrastructure/MapServer`
- Coverage: National transmission substations and lines
- Voltage: VOLTAGEKV field (numeric)
- Licence: GA Copyright (liberal reuse)

**Rosetta Network Map**
- Portal: `https://renewables.networkmap.energy`
- Coverage: Tx + Dx planning layers, hosting capacity
- Access: Commercial licence required

**MapStand**
- Portal: `https://mapstand.com`
- Coverage: Global power infrastructure
- Access: Commercial licence required

## Output Formats

The pipeline exports to:

- **CSV**: Flat table with all fields
- **GeoJSON**: FeatureCollection for mapping
- **DataFrame**: Pandas DataFrame for analysis

## Workflow Steps

### France
1. Extract RTE/ODRÉ as Tx backbone
2. Extract Agence ORE as Dx primary layer
3. Extract Enedis for deeper Dx coverage
4. Reconcile Enedis with ORE (spatial join)
5. Attach Caparéseau capacity (if available)
6. OSM QA enrichment

### Australia
1. Extract GA NEI substations and lines
2. Integrate Rosetta capacity/Dx (if available)
3. Add DNSP portal data (if available)
4. MapStand gap-filling (if available)
5. OSM QA enrichment

## Extending the Pipeline

### Adding a New Source

1. Create ETL module in the appropriate country folder
2. Implement `fetch_*` function for API calls
3. Implement `transform_*` function mapping to GridNode
4. Implement `extract_*` function for full extraction
5. Integrate into workflow.py

### Adding Capacity Sources

The schema supports capacity attributes. When Caparéseau or Rosetta data is available:

```python
node.available_capacity_mw = capacity_value
node.reserved_capacity_mw = reserved_value
node.capacity_quality_flag = CapacityQuality.INDICATIVE
```

## Licence Considerations

- **ODbL (OSM)**: Share-alike; combined databases may need ODbL compatibility
- **Licence Ouverte (France)**: Liberal reuse with attribution
- **GA Copyright**: Liberal reuse with attribution
- **Commercial (Rosetta, MapStand)**: Contract-dependent

## Quality Flags

Every node carries quality flags to support downstream decision-making:

- `source_rank`: Authority level (1=best)
- `voltage_quality_flag`: Precision of voltage data
- `capacity_quality_flag`: Reliability of capacity data
- `geom_quality_flag`: Precision of coordinates
- `review_flag`: Manual review needed
- `match_status`: Matching result

## References

- [RTE/ODRÉ](https://odre.opendatasoft.com/explore/dataset/postes-electriques-rte/)
- [Agence ORE](https://opendata.agenceore.fr/explore/dataset/postes-source/)
- [Enedis](https://data.enedis.fr/explore/dataset/poste-source/)
- [Caparéseau](https://www.services-rte.com/en/learn-more-about-our-services/consult-the-reception-capacity-of-the-grid-capareseau.html)
- [GA NEI](https://services.ga.gov.au/gis/rest/services/National_Electricity_Infrastructure/MapServer)
- [Rosetta](https://renewables.networkmap.energy)
- [MapStand](https://mapstand.com)
- [OSM Power Networks](https://wiki.openstreetmap.org/wiki/Power_networks)
