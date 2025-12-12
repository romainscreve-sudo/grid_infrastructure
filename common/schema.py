"""
Unified Schema for Grid Infrastructure Nodes
Harmonises France and Australia electricity network data sources.
"""
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
from datetime import datetime

class Country(Enum):
    FR = "FR"
    AU = "AU"

class SourceRank(Enum):
    TSO_DSO_OFFICIAL = 1
    GOVERNMENT_REPOSITORY = 2
    COMMERCIAL = 3
    COMMUNITY = 4

class NetworkLevel(Enum):
    TX = "TX"
    DX_PRIMARY = "DX_PRIMARY"
    DX_SECONDARY = "DX_SECONDARY"
    DX_OTHER = "DX_OTHER"

class SubstationRole(Enum):
    INTERFACE_TX_DX = "INTERFACE_TX_DX"
    ZONE_SUBSTATION = "ZONE_SUBSTATION"
    DISTRIBUTION_SUBSTATION = "DISTRIBUTION_SUBSTATION"
    GEN_CONNECTION_POINT = "GEN_CONNECTION_POINT"
    SWITCHING_STATION = "SWITCHING_STATION"
    UNKNOWN = "UNKNOWN"

class VoltageQuality(Enum):
    NUMERIC_EXACT = "NUMERIC_EXACT"
    RANGE_OR_CLASS_ONLY = "RANGE_OR_CLASS_ONLY"
    UNKNOWN = "UNKNOWN"

class CapacityQuality(Enum):
    TSO_CONFIRMED = "TSO_CONFIRMED"
    INDICATIVE = "INDICATIVE"
    HISTORICAL = "HISTORICAL"
    MISSING = "MISSING"

class GeomQuality(Enum):
    TSO_DSO_EXACT = "TSO_DSO_EXACT"
    GOV_APPROX = "GOV_APPROX"
    DERIVED_FROM_OSM = "DERIVED_FROM_OSM"
    COMMERCIAL = "COMMERCIAL"
    UNKNOWN = "UNKNOWN"

@dataclass
class GridNode:
    """Unified substation / connection-point record."""
    
    # Core identification
    id_global: str
    country: Country
    source_primary: str
    source_rank: SourceRank
    name: str
    
    # Network role
    level_tx_dx: NetworkLevel
    substation_role: SubstationRole = SubstationRole.UNKNOWN
    operator_name: Optional[str] = None
    operator_id: Optional[str] = None
    
    # Voltage
    voltage_kv_nominal_max: Optional[float] = None
    voltage_kv_nominal_min: Optional[float] = None
    voltage_classes: Optional[List[str]] = None
    voltage_quality_flag: VoltageQuality = VoltageQuality.UNKNOWN
    
    # Capacity
    installed_capacity_mva: Optional[float] = None
    available_capacity_mw: Optional[float] = None
    available_capacity_mva: Optional[float] = None
    reserved_capacity_mw: Optional[float] = None
    capacity_quality_flag: CapacityQuality = CapacityQuality.MISSING
    
    # Geometry
    lon: Optional[float] = None
    lat: Optional[float] = None
    geom_type: str = "POINT"
    geom_quality_flag: GeomQuality = GeomQuality.UNKNOWN
    
    # Linkage / provenance (nullable foreign keys)
    id_rte_site: Optional[str] = None
    id_enedis_poste: Optional[str] = None
    id_ore_poste: Optional[str] = None
    id_ga_nei: Optional[str] = None
    id_osm: Optional[str] = None
    id_rosetta: Optional[str] = None
    id_mapstand: Optional[str] = None
    id_capareseau: Optional[str] = None
    
    # Metadata
    last_update_from_source: Optional[datetime] = None
    licence_code: Optional[str] = None
    
    # QA flags
    match_status: str = "UNMATCHED"
    review_flag: bool = False
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame/export."""
        return {
            'id_global': self.id_global,
            'country': self.country.value,
            'source_primary': self.source_primary,
            'source_rank': self.source_rank.value,
            'name': self.name,
            'level_tx_dx': self.level_tx_dx.value,
            'substation_role': self.substation_role.value,
            'operator_name': self.operator_name,
            'operator_id': self.operator_id,
            'voltage_kv_nominal_max': self.voltage_kv_nominal_max,
            'voltage_kv_nominal_min': self.voltage_kv_nominal_min,
            'voltage_classes': ','.join(self.voltage_classes) if self.voltage_classes else None,
            'voltage_quality_flag': self.voltage_quality_flag.value,
            'installed_capacity_mva': self.installed_capacity_mva,
            'available_capacity_mw': self.available_capacity_mw,
            'available_capacity_mva': self.available_capacity_mva,
            'reserved_capacity_mw': self.reserved_capacity_mw,
            'capacity_quality_flag': self.capacity_quality_flag.value,
            'lon': self.lon,
            'lat': self.lat,
            'geom_type': self.geom_type,
            'geom_quality_flag': self.geom_quality_flag.value,
            'id_rte_site': self.id_rte_site,
            'id_enedis_poste': self.id_enedis_poste,
            'id_ore_poste': self.id_ore_poste,
            'id_ga_nei': self.id_ga_nei,
            'id_osm': self.id_osm,
            'id_rosetta': self.id_rosetta,
            'id_mapstand': self.id_mapstand,
            'id_capareseau': self.id_capareseau,
            'last_update_from_source': self.last_update_from_source.isoformat() if self.last_update_from_source else None,
            'licence_code': self.licence_code,
            'match_status': self.match_status,
            'review_flag': self.review_flag,
            'notes': self.notes,
        }

# Source constants
FRANCE_SOURCES = {
    'RTE_ODRE': {'rank': SourceRank.TSO_DSO_OFFICIAL, 'licence': 'FR_LO'},
    'AGENCE_ORE': {'rank': SourceRank.TSO_DSO_OFFICIAL, 'licence': 'FR_LO'},
    'ENEDIS': {'rank': SourceRank.TSO_DSO_OFFICIAL, 'licence': 'FR_LO'},
    'CAPARESEAU': {'rank': SourceRank.TSO_DSO_OFFICIAL, 'licence': 'FR_LO'},
    'OSM': {'rank': SourceRank.COMMUNITY, 'licence': 'ODbL'},
}

AUSTRALIA_SOURCES = {
    'GA_NEI': {'rank': SourceRank.GOVERNMENT_REPOSITORY, 'licence': 'GA_COPYRIGHT'},
    'ROSETTA': {'rank': SourceRank.COMMERCIAL, 'licence': 'COMMERCIAL'},
    'MAPSTAND': {'rank': SourceRank.COMMERCIAL, 'licence': 'COMMERCIAL'},
    'OSM': {'rank': SourceRank.COMMUNITY, 'licence': 'ODbL'},
}

# Voltage class mappings
VOLTAGE_CLASSES = {
    'EHV': [400, 330, 275, 225],  # Extra High Voltage
    'HV': [132, 110, 90, 66, 63, 45, 33],  # High Voltage
    'MV': [22, 20, 15, 11, 10, 6.6],  # Medium Voltage
    'LV': [0.4, 0.23],  # Low Voltage
}

def classify_voltage(kv: float) -> str:
    """Classify a voltage in kV to a voltage class."""
    if kv >= 200:
        return 'EHV'
    elif kv >= 30:
        return 'HV'
    elif kv >= 1:
        return 'MV'
    else:
        return 'LV'

def generate_global_id(country: str, source: str, local_id: str) -> str:
    """Generate a globally unique ID."""
    return f"{country}_{source}_{local_id}"
