"""
Entity Matching Utilities for Grid Infrastructure
Hierarchical matching: name → operator → distance → voltage class
"""
import re
from math import radians, sin, cos, sqrt, atan2
from typing import Optional, Tuple, List
from difflib import SequenceMatcher

def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate distance in meters between two points."""
    R = 6371000  # Earth radius in meters
    
    lat1_r, lat2_r = radians(lat1), radians(lat2)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    
    a = sin(dlat/2)**2 + cos(lat1_r) * cos(lat2_r) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

def normalize_name(name: str) -> str:
    """Normalize substation name for matching."""
    if not name:
        return ""
    
    name = name.upper()
    name = re.sub(r'[^\w\s]', ' ', name)
    
    # Remove common suffixes/prefixes
    removals = [
        'SUBSTATION', 'SUB', 'SS', 'ZS', 'TS', 
        'POSTE', 'SOURCE', 'ELECTRIQUE',
        'TERMINAL', 'STATION', 'ZONE',
        'KV', 'V', 'MVA', 'MW',
        'NORTH', 'SOUTH', 'EAST', 'WEST',
        'NORD', 'SUD', 'EST', 'OUEST',
    ]
    for r in removals:
        name = re.sub(rf'\b{r}\b', '', name)
    
    name = ' '.join(name.split())
    return name.strip()

def name_similarity(name1: str, name2: str) -> float:
    """Calculate similarity score between two names (0-1)."""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    
    if not n1 or not n2:
        return 0.0
    
    return SequenceMatcher(None, n1, n2).ratio()

def voltage_compatible(v1_max: Optional[float], v1_min: Optional[float],
                       v2_max: Optional[float], v2_min: Optional[float],
                       tolerance_pct: float = 0.1) -> bool:
    """Check if two voltage ranges are compatible."""
    if not any([v1_max, v1_min, v2_max, v2_min]):
        return True  # Unknown voltages are compatible
    
    # Get effective ranges
    v1_high = v1_max or v1_min or 0
    v1_low = v1_min or v1_max or 0
    v2_high = v2_max or v2_min or 0
    v2_low = v2_min or v2_max or 0
    
    # Check for overlap with tolerance
    tolerance = max(v1_high, v2_high) * tolerance_pct
    return not (v1_high + tolerance < v2_low or v2_high + tolerance < v1_low)

class MatchResult:
    """Container for match results."""
    def __init__(self, matched: bool, score: float, match_type: str,
                 distance_m: Optional[float] = None, name_sim: Optional[float] = None):
        self.matched = matched
        self.score = score
        self.match_type = match_type
        self.distance_m = distance_m
        self.name_sim = name_sim

def match_nodes(node1: dict, node2: dict,
                distance_threshold_m: float = 1000,
                name_threshold: float = 0.7) -> MatchResult:
    """
    Hierarchical matching between two nodes.
    Returns MatchResult with score and match type.
    """
    # Stage 1: Exact name match
    n1, n2 = normalize_name(node1.get('name', '')), normalize_name(node2.get('name', ''))
    if n1 and n2 and n1 == n2:
        return MatchResult(True, 1.0, 'EXACT_NAME')
    
    # Stage 2: Name similarity + distance
    name_sim = name_similarity(node1.get('name', ''), node2.get('name', ''))
    
    distance_m = None
    if all([node1.get('lon'), node1.get('lat'), node2.get('lon'), node2.get('lat')]):
        distance_m = haversine_distance(
            node1['lon'], node1['lat'],
            node2['lon'], node2['lat']
        )
    
    # Stage 3: Check voltage compatibility
    v_compat = voltage_compatible(
        node1.get('voltage_kv_nominal_max'), node1.get('voltage_kv_nominal_min'),
        node2.get('voltage_kv_nominal_max'), node2.get('voltage_kv_nominal_min')
    )
    
    # Scoring logic
    if name_sim >= name_threshold and distance_m is not None and distance_m <= distance_threshold_m:
        if v_compat:
            score = 0.4 * name_sim + 0.4 * (1 - distance_m / distance_threshold_m) + 0.2
            return MatchResult(True, score, 'NAME_DISTANCE_VOLTAGE', distance_m, name_sim)
        else:
            return MatchResult(False, 0.0, 'VOLTAGE_MISMATCH', distance_m, name_sim)
    
    # Stage 4: Operator + distance (if names don't match well)
    op1 = (node1.get('operator_name') or '').upper()
    op2 = (node2.get('operator_name') or '').upper()
    
    if op1 and op2 and op1 == op2:
        if distance_m is not None and distance_m <= distance_threshold_m and v_compat:
            score = 0.3 + 0.5 * (1 - distance_m / distance_threshold_m) + 0.2 * name_sim
            return MatchResult(True, score, 'OPERATOR_DISTANCE', distance_m, name_sim)
    
    # Stage 5: Distance only (weak match)
    if distance_m is not None and distance_m <= distance_threshold_m * 0.5 and v_compat:
        score = 0.3 * (1 - distance_m / (distance_threshold_m * 0.5))
        return MatchResult(True, score, 'DISTANCE_ONLY', distance_m, name_sim)
    
    return MatchResult(False, 0.0, 'NO_MATCH', distance_m, name_sim)

def find_best_match(target: dict, candidates: List[dict],
                    distance_threshold_m: float = 1000,
                    name_threshold: float = 0.7) -> Tuple[Optional[dict], MatchResult]:
    """Find the best matching candidate for a target node."""
    best_candidate = None
    best_result = MatchResult(False, 0.0, 'NO_MATCH')
    
    for candidate in candidates:
        result = match_nodes(target, candidate, distance_threshold_m, name_threshold)
        if result.matched and result.score > best_result.score:
            best_candidate = candidate
            best_result = result
    
    return best_candidate, best_result

def build_spatial_index(nodes: List[dict], grid_size_deg: float = 0.1) -> dict:
    """Build a simple grid-based spatial index for faster matching."""
    index = {}
    for node in nodes:
        if node.get('lon') is not None and node.get('lat') is not None:
            grid_x = int(node['lon'] / grid_size_deg)
            grid_y = int(node['lat'] / grid_size_deg)
            key = (grid_x, grid_y)
            if key not in index:
                index[key] = []
            index[key].append(node)
    return index

def get_nearby_candidates(index: dict, lon: float, lat: float,
                          grid_size_deg: float = 0.1) -> List[dict]:
    """Get candidates from spatial index in nearby grid cells."""
    grid_x = int(lon / grid_size_deg)
    grid_y = int(lat / grid_size_deg)
    
    candidates = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            key = (grid_x + dx, grid_y + dy)
            candidates.extend(index.get(key, []))
    
    return candidates
