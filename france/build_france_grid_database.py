"""
France Grid Infrastructure Database Builder

Creates a comprehensive database of:
1. RTE Transmission Substations (~5,000)
2. Postes Sources (~4,800)

Enriched with OSM data for coordinates and voltage details.
"""

import requests
import json
import time
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
import pandas as pd

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Substation:
    """Unified substation record."""
    id: str
    name: str
    network_level: str  # TX (transmission) or DX_SOURCE (poste source)

    # Official source data
    source: str  # RTE_ODRE, AGENCE_ORE
    source_id: str

    # Location
    lat: Optional[float] = None
    lon: Optional[float] = None
    coord_source: str = "MISSING"  # OFFICIAL, OSM, GEOCODED
    coord_quality: str = "MISSING"  # EXACT, APPROXIMATE, COMMUNE_CENTROID

    # Administrative
    commune: Optional[str] = None
    departement: Optional[str] = None
    region: Optional[str] = None
    code_insee: Optional[str] = None

    # Operator
    operator: Optional[str] = None

    # Voltage
    voltage_in_kv: Optional[float] = None
    voltage_out_kv: Optional[float] = None
    voltage_levels: Optional[str] = None  # e.g., "225kV/63kV/20kV"
    voltage_source: str = "MISSING"  # OFFICIAL, OSM

    # Capacity (from CAPARESEAU)
    capacity_total_mw: Optional[float] = None
    capacity_reserved_mw: Optional[float] = None
    capacity_available_mw: Optional[float] = None
    capacity_source: str = "MISSING"

    # Type/Function
    function: Optional[str] = None  # Poste de transformation, etc.
    substation_type: Optional[str] = None  # From OSM

    # OSM enrichment
    osm_id: Optional[str] = None
    osm_match_distance_m: Optional[float] = None
    osm_match_score: Optional[float] = None

    # Metadata
    last_updated: str = ""
    notes: Optional[str] = None


# =============================================================================
# API EXTRACTION FUNCTIONS
# =============================================================================

def fetch_all_rte_substations() -> List[Dict]:
    """Fetch all RTE transmission substations from ODRÉ."""
    print("\n" + "="*60)
    print("EXTRACTING RTE TRANSMISSION SUBSTATIONS")
    print("="*60)

    url = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/postes-electriques-rte/records"
    all_records = []
    offset = 0
    limit = 100

    while True:
        params = {'limit': limit, 'offset': offset}
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        records = data.get('results', [])
        if not records:
            break

        all_records.extend(records)
        print(f"  Fetched {len(all_records)} / {data.get('total_count', '?')} records...")

        offset += limit
        if len(records) < limit:
            break

        time.sleep(0.1)  # Be polite to API

    print(f"Total RTE substations: {len(all_records)}")
    return all_records


def fetch_with_retry(url: str, params: Dict, max_retries: int = 4) -> Dict:
    """Fetch URL with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)  # 2, 4, 8, 16 seconds
                print(f"    Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                raise


def fetch_all_agence_ore_postes() -> List[Dict]:
    """Fetch all postes sources from Agence ORE."""
    print("\n" + "="*60)
    print("EXTRACTING AGENCE ORE POSTES SOURCES")
    print("="*60)

    url = "https://opendata.agenceore.fr/api/explore/v2.1/catalog/datasets/postes-source/records"
    all_records = []
    offset = 0
    limit = 100

    while True:
        params = {'limit': limit, 'offset': offset}
        data = fetch_with_retry(url, params)

        records = data.get('results', [])
        if not records:
            break

        all_records.extend(records)
        print(f"  Fetched {len(all_records)} / {data.get('total_count', '?')} records...")

        offset += limit
        if len(records) < limit:
            break

        time.sleep(0.3)  # Increased delay to be gentler on API

    print(f"Total Agence ORE postes sources: {len(all_records)}")
    return all_records


def fetch_osm_substations_france() -> List[Dict]:
    """Fetch all substations from OSM via Overpass API."""
    print("\n" + "="*60)
    print("EXTRACTING OSM SUBSTATIONS FOR FRANCE")
    print("="*60)

    # Overpass query for French substations
    # Using France bounding box: mainland + Corsica
    query = """
    [out:json][timeout:300];
    (
      // Metropolitan France bounding box
      node["power"="substation"](41.3,-5.2,51.1,9.6);
      way["power"="substation"](41.3,-5.2,51.1,9.6);
      relation["power"="substation"](41.3,-5.2,51.1,9.6);
    );
    out center tags;
    """

    print("  Querying Overpass API (this may take a few minutes)...")
    url = "https://overpass-api.de/api/interpreter"

    resp = requests.post(url, data={'data': query}, timeout=600)
    resp.raise_for_status()
    data = resp.json()

    elements = data.get('elements', [])
    print(f"  Raw OSM elements: {len(elements)}")

    # Process elements
    substations = []
    for elem in elements:
        tags = elem.get('tags', {})

        # Get coordinates (center for ways/relations)
        lat = elem.get('lat') or elem.get('center', {}).get('lat')
        lon = elem.get('lon') or elem.get('center', {}).get('lon')

        if not lat or not lon:
            continue

        # Parse voltage
        voltage_str = tags.get('voltage', '')
        voltages = parse_osm_voltage(voltage_str)

        substation = {
            'osm_id': f"{elem['type']}_{elem['id']}",
            'osm_type': elem['type'],
            'name': tags.get('name', ''),
            'lat': lat,
            'lon': lon,
            'voltage_raw': voltage_str,
            'voltage_max_kv': max(voltages) if voltages else None,
            'voltage_min_kv': min(voltages) if voltages else None,
            'voltage_levels': ';'.join(str(int(v)) for v in sorted(voltages, reverse=True)) if voltages else None,
            'operator': tags.get('operator', ''),
            'substation_type': tags.get('substation', ''),
            'power': tags.get('power', ''),
            'ref': tags.get('ref', ''),
        }
        substations.append(substation)

    print(f"Total OSM substations with coordinates: {len(substations)}")

    # Stats
    with_name = sum(1 for s in substations if s['name'])
    with_voltage = sum(1 for s in substations if s['voltage_max_kv'])
    print(f"  With name: {with_name}")
    print(f"  With voltage: {with_voltage}")

    return substations


def parse_osm_voltage(voltage_str: str) -> List[float]:
    """Parse OSM voltage string (e.g., '400000;225000' or '63000') to list of kV."""
    if not voltage_str:
        return []

    voltages = []
    # Handle semicolon-separated values
    for part in voltage_str.split(';'):
        try:
            v = float(part.strip())
            # Convert V to kV if needed
            if v > 1000:
                v = v / 1000
            voltages.append(v)
        except ValueError:
            continue

    return voltages


# =============================================================================
# TRANSFORMATION FUNCTIONS
# =============================================================================

def transform_rte_records(records: List[Dict]) -> List[Substation]:
    """Transform RTE records to Substation objects."""
    substations = []
    seen_ids = set()

    for idx, rec in enumerate(records):
        code = rec.get('code_poste', '')
        name = rec.get('nom_poste', '')
        departement = rec.get('departement', '')

        # Generate unique key - use code if available, else name+departement
        if code:
            unique_key = code
        elif name:
            unique_key = f"{name}_{departement}"
        else:
            unique_key = f"idx_{idx}"

        # Skip duplicates
        if unique_key in seen_ids:
            continue
        seen_ids.add(unique_key)

        # Parse voltage from tension field
        tension = rec.get('tension', '')
        voltages = parse_rte_voltage(tension)

        sub = Substation(
            id=f"RTE_{code}",
            name=name,
            network_level="TX",
            source="RTE_ODRE",
            source_id=code,
            departement=rec.get('departement'),
            operator="RTE",
            voltage_levels=tension,
            voltage_in_kv=max(voltages) if voltages else None,
            voltage_out_kv=min(voltages) if len(voltages) > 1 else None,
            voltage_source="OFFICIAL",
            function=rec.get('fonction'),
            last_updated=datetime.now().isoformat(),
        )
        substations.append(sub)

    return substations


def parse_rte_voltage(tension: str) -> List[float]:
    """Parse RTE voltage string (e.g., '400kV / 225kV' or '225kV')."""
    if not tension:
        return []

    voltages = []
    matches = re.findall(r'(\d+)\s*kV', tension, re.IGNORECASE)
    for m in matches:
        voltages.append(float(m))

    return voltages


def transform_ore_records(records: List[Dict]) -> List[Substation]:
    """Transform Agence ORE records to Substation objects."""
    substations = []
    seen_ids = set()

    for rec in records:
        name = rec.get('nom_ps') or rec.get('nom', '') or ''

        # Extract coordinates
        geo = rec.get('geo_point_2d') or {}
        lat = geo.get('lat')
        lon = geo.get('lon')

        # Generate unique ID - use coordinates hash when name is missing
        commune = rec.get('commune') or ''
        code_insee = rec.get('code_insee') or ''

        if name:
            name_safe = name.replace(' ', '_')[:30]
        elif lat and lon:
            # Use coordinate hash for unnamed stations
            coord_hash = f"{lat:.5f}_{lon:.5f}".replace('.', '').replace('-', 'm')
            name_safe = f"loc_{coord_hash}"
        else:
            name_safe = f"idx_{len(substations)}"

        sub_id = f"ORE_{code_insee}_{name_safe}"

        # Ensure uniqueness
        if sub_id in seen_ids:
            sub_id = f"{sub_id}_{len(substations)}"
        seen_ids.add(sub_id)

        # Postes sources by definition are HTB/HTA transformers
        # Default: 63kV -> 20kV (most common configuration)
        # Output is always 20kV (HTA standard in France)
        default_voltage_in = 63.0
        default_voltage_out = 20.0

        sub = Substation(
            id=sub_id,
            name=name,
            network_level="DX_SOURCE",
            source="AGENCE_ORE",
            source_id=code_insee,
            lat=lat,
            lon=lon,
            coord_source="OFFICIAL" if lat else "MISSING",
            coord_quality="EXACT" if lat else "MISSING",
            commune=commune,
            departement=rec.get('departement') or '',
            region=rec.get('region') or '',
            code_insee=code_insee,
            operator=rec.get('nom_grd') or '',
            voltage_in_kv=default_voltage_in,
            voltage_out_kv=default_voltage_out,
            voltage_levels="63kV/20kV",
            voltage_source="DEFAULT",
            last_updated=rec.get('date_maj') or datetime.now().isoformat(),
        )
        substations.append(sub)

    return substations


# =============================================================================
# ENRICHMENT FUNCTIONS
# =============================================================================

def normalize_name(name: str) -> str:
    """Normalize substation name for matching."""
    if not name:
        return ""

    # Lowercase
    n = name.lower()

    # Remove common prefixes (OSM style)
    prefixes = [
        'poste électrique de la ', 'poste électrique du ', 'poste électrique de l\'',
        'poste électrique d\'', 'poste électrique de ', 'poste electrique de ',
        'sous-station sncf de ', 'sous-station de ', 'sous station ',
        'poste source ', 'poste de ', 'poste ', 'ps ',
        'portique électrique de ', 'portique de ',
    ]
    for p in prefixes:
        if n.startswith(p):
            n = n[len(p):]
            break

    # Handle RTE-style suffixes like "(LA)", "(LE)", "(LES)"
    # Convert "DURANNE (LA)" to "LA DURANNE"
    import re
    suffix_match = re.search(r'\s*\((la|le|les|l\')\)\s*$', n, re.IGNORECASE)
    if suffix_match:
        article = suffix_match.group(1)
        n = article + ' ' + n[:suffix_match.start()]

    # Remove other parenthetical content like "(CLIENT)", "(S.N.C.F.)"
    n = re.sub(r'\s*\([^)]*\)\s*', ' ', n)

    # Remove accents (simple approach)
    replacements = {
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'à': 'a', 'â': 'a', 'ä': 'a',
        'ù': 'u', 'û': 'u', 'ü': 'u',
        'î': 'i', 'ï': 'i',
        'ô': 'o', 'ö': 'o',
        'ç': 'c',
        '-': ' ', '_': ' ', "'": ' ',
    }
    for old, new in replacements.items():
        n = n.replace(old, new)

    # Remove extra whitespace
    n = ' '.join(n.split())

    return n


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in meters."""
    from math import radians, sin, cos, sqrt, atan2

    R = 6371000  # Earth radius in meters

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c


def name_similarity(name1: str, name2: str) -> float:
    """Calculate similarity between two names (0-1)."""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)

    if not n1 or not n2:
        return 0.0

    # Exact match
    if n1 == n2:
        return 1.0

    # One contains the other
    if n1 in n2 or n2 in n1:
        return 0.9

    # Word overlap
    words1 = set(n1.split())
    words2 = set(n2.split())

    if not words1 or not words2:
        return 0.0

    overlap = len(words1 & words2)
    total = len(words1 | words2)

    return overlap / total if total > 0 else 0.0


def build_name_index(osm_substations: List[Dict]) -> Dict[str, List[Dict]]:
    """Build index of OSM substations by normalized name tokens."""
    index = {}
    for osm in osm_substations:
        name = normalize_name(osm.get('name', ''))
        ref = normalize_name(osm.get('ref', ''))

        # Index by each word in name
        for word in name.split():
            if len(word) >= 3:  # Skip short words
                if word not in index:
                    index[word] = []
                index[word].append(osm)

        # Also index by ref
        for word in ref.split():
            if len(word) >= 3:
                if word not in index:
                    index[word] = []
                index[word].append(osm)

    return index


def geocode_commune(name: str, departement: str = None) -> Optional[Tuple[float, float]]:
    """Try to geocode a location using French government API."""
    try:
        # Clean up name for geocoding
        search_name = normalize_name(name)

        # Try the French address API
        url = "https://api-adresse.data.gouv.fr/search/"
        params = {
            'q': search_name,
            'type': 'municipality',
            'limit': 1
        }

        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            features = data.get('features', [])
            if features:
                coords = features[0]['geometry']['coordinates']
                return (coords[1], coords[0])  # lat, lon
    except:
        pass
    return None


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Calculate Levenshtein similarity ratio (0-1)."""
    if not s1 or not s2:
        return 0.0

    # Simple implementation
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    # Create distance matrix
    distances = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        distances[i][0] = i
    for j in range(len2 + 1):
        distances[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            distances[i][j] = min(
                distances[i-1][j] + 1,
                distances[i][j-1] + 1,
                distances[i-1][j-1] + cost
            )

    distance = distances[len1][len2]
    max_len = max(len1, len2)
    return 1.0 - (distance / max_len)


def enrich_rte_with_osm(rte_substations: List[Substation], osm_substations: List[Dict]) -> List[Substation]:
    """Enrich RTE substations with OSM coordinates using multiple strategies."""
    print("\n" + "="*60)
    print("ENRICHING RTE SUBSTATIONS WITH OSM COORDINATES")
    print("="*60)

    # Filter OSM to likely transmission substations (>= 63kV or no voltage but named)
    osm_tx = [s for s in osm_substations
              if (s['voltage_max_kv'] and s['voltage_max_kv'] >= 63)
              or (s['name'] and s['substation_type'] in ('transmission', 'converter', ''))]

    print(f"  OSM candidates (>= 63kV or transmission): {len(osm_tx)}")

    # Build name index for fast lookup
    print("  Building name index...")
    name_index = build_name_index(osm_tx)
    print(f"  Index contains {len(name_index)} unique terms")

    # STRATEGY 1: Word-based index matching (fast, high precision)
    matched_word = 0
    matched_fuzzy = 0
    matched_geocode = 0
    unmatched_rte = []

    print("\n  Strategy 1: Word-based matching...")
    for i, rte in enumerate(rte_substations):
        if (i + 1) % 1000 == 0:
            print(f"    Processing {i + 1}/{len(rte_substations)}...")

        # Get candidate OSM substations via index
        rte_name = normalize_name(rte.name)
        candidates = set()
        for word in rte_name.split():
            if len(word) >= 3 and word in name_index:
                for osm in name_index[word]:
                    candidates.add(id(osm))

        # Convert to actual objects and find best match
        candidate_list = [osm for osm in osm_tx if id(osm) in candidates]

        best_match = None
        best_score = 0

        for osm in candidate_list:
            sim = name_similarity(rte.name, osm['name'])
            if osm.get('ref'):
                ref_sim = name_similarity(rte.name, osm['ref'])
                sim = max(sim, ref_sim)

            if sim > best_score:
                best_score = sim
                best_match = osm

        # Accept match if score >= 0.7
        if best_match and best_score >= 0.7:
            rte.lat = best_match['lat']
            rte.lon = best_match['lon']
            rte.coord_source = "OSM"
            rte.coord_quality = "EXACT"
            rte.osm_id = best_match['osm_id']
            rte.osm_match_score = best_score
            rte.substation_type = best_match.get('substation_type')

            if best_match['voltage_levels'] and not rte.voltage_levels:
                rte.voltage_levels = best_match['voltage_levels']

            matched_word += 1
        else:
            unmatched_rte.append(rte)

    print(f"    Matched via word index: {matched_word}")

    # STRATEGY 2: Fuzzy Levenshtein matching for remaining (optimized)
    print(f"\n  Strategy 2: Fuzzy matching for {len(unmatched_rte)} remaining...")
    still_unmatched = []

    # Pre-filter and pre-normalize OSM names (massive speedup)
    osm_named = []
    for osm in osm_tx:
        osm_name = osm.get('name', '')
        if osm_name:
            osm_named.append({
                **osm,
                '_norm_name': normalize_name(osm_name),
                '_norm_ref': normalize_name(osm.get('ref', '')) if osm.get('ref') else None
            })
    print(f"    Pre-filtered to {len(osm_named)} named OSM substations")

    # Build first-letter buckets for fast lookup
    letter_buckets = {}
    for osm in osm_named:
        first_letter = osm['_norm_name'][:1] if osm['_norm_name'] else ''
        if first_letter not in letter_buckets:
            letter_buckets[first_letter] = []
        letter_buckets[first_letter].append(osm)

    for i, rte in enumerate(unmatched_rte):
        if (i + 1) % 500 == 0:
            print(f"    Processing {i + 1}/{len(unmatched_rte)}...")

        rte_name_norm = normalize_name(rte.name)
        best_match = None
        best_score = 0

        # Check same-letter bucket and variations
        first_letter = rte_name_norm[:1] if rte_name_norm else ''
        candidates = list(letter_buckets.get(first_letter, []))

        # Extended letter mappings for French names
        nearby = {
            'a': ['e', 'h'],  # article variations (a->e), silent h
            'b': ['p'],       # p/b confusion
            'c': ['k', 's'],  # c/k/s variations
            'd': ['t'],       # d/t confusion
            'e': ['a', 'i'],
            'f': ['v'],       # f/v confusion
            'g': ['j'],       # g/j variations
            'h': ['a', 'e'],  # silent h
            'i': ['y', 'e'],
            'j': ['g'],
            'k': ['c', 'q'],
            'l': ['r'],       # common French confusions
            'm': ['n'],       # m/n confusion
            'n': ['m'],
            'o': ['u'],
            'p': ['b'],
            'q': ['k', 'c'],
            'r': ['l'],
            's': ['z', 'c'],
            't': ['d'],
            'u': ['o'],
            'v': ['f'],
            'w': ['v'],
            'x': ['s'],
            'y': ['i'],
            'z': ['s'],
        }
        for alt_letter in nearby.get(first_letter, []):
            candidates.extend(letter_buckets.get(alt_letter, []))

        for osm in candidates:
            lev_score = levenshtein_ratio(rte_name_norm, osm['_norm_name'])
            if osm['_norm_ref']:
                ref_score = levenshtein_ratio(rte_name_norm, osm['_norm_ref'])
                lev_score = max(lev_score, ref_score)

            if lev_score > best_score:
                best_score = lev_score
                best_match = osm

        # Skip full scan - bucket matching should catch most variations

        # Accept fuzzy match if >= 0.75 Levenshtein similarity
        if best_match and best_score >= 0.75:
            rte.lat = best_match['lat']
            rte.lon = best_match['lon']
            rte.coord_source = "OSM_FUZZY"
            rte.coord_quality = "EXACT"
            rte.osm_id = best_match['osm_id']
            rte.osm_match_score = best_score
            rte.substation_type = best_match.get('substation_type')
            matched_fuzzy += 1
        else:
            still_unmatched.append(rte)

    print(f"    Matched via fuzzy: {matched_fuzzy}")

    # STRATEGY 3: Geocode by commune name for remaining
    print(f"\n  Strategy 3: Geocoding {len(still_unmatched)} remaining by name...")
    geocode_count = 0

    for i, rte in enumerate(still_unmatched):
        if (i + 1) % 100 == 0:
            print(f"    Geocoding {i + 1}/{len(still_unmatched)}...")

        # Try to geocode using the name
        coords = geocode_commune(rte.name, rte.departement)
        if coords:
            rte.lat, rte.lon = coords
            rte.coord_source = "GEOCODED"
            rte.coord_quality = "COMMUNE_CENTROID"
            matched_geocode += 1
            geocode_count += 1

        # Rate limit geocoding
        if geocode_count % 50 == 0:
            time.sleep(1)

    print(f"    Matched via geocoding: {matched_geocode}")

    total_matched = matched_word + matched_fuzzy + matched_geocode
    print(f"\n  TOTAL RTE substations with coordinates: {total_matched}/{len(rte_substations)} ({100*total_matched/len(rte_substations):.1f}%)")
    print(f"    - Word matching: {matched_word}")
    print(f"    - Fuzzy matching: {matched_fuzzy}")
    print(f"    - Geocoding: {matched_geocode}")

    return rte_substations


def enrich_ore_with_osm(ore_substations: List[Substation], osm_substations: List[Dict]) -> List[Substation]:
    """Enrich Agence ORE postes sources with OSM voltage data."""
    print("\n" + "="*60)
    print("ENRICHING POSTES SOURCES WITH OSM VOLTAGE DATA")
    print("="*60)

    # Build spatial index (simple grid-based)
    # Group OSM by rounded coordinates for faster lookup
    osm_grid = {}
    for osm in osm_substations:
        key = (round(osm['lat'], 1), round(osm['lon'], 1))
        if key not in osm_grid:
            osm_grid[key] = []
        osm_grid[key].append(osm)

    matched = 0
    voltage_enriched = 0

    for ore in ore_substations:
        if not ore.lat or not ore.lon:
            continue

        # Get nearby OSM candidates
        key = (round(ore.lat, 1), round(ore.lon, 1))
        candidates = []
        for dk_lat in [-0.1, 0, 0.1]:
            for dk_lon in [-0.1, 0, 0.1]:
                nearby_key = (round(ore.lat + dk_lat, 1), round(ore.lon + dk_lon, 1))
                candidates.extend(osm_grid.get(nearby_key, []))

        # Find best match within 500m
        best_match = None
        best_distance = float('inf')

        for osm in candidates:
            dist = haversine_distance(ore.lat, ore.lon, osm['lat'], osm['lon'])

            if dist < 500 and dist < best_distance:
                # Also check name similarity for confirmation
                name_sim = name_similarity(ore.name, osm['name'])
                if name_sim >= 0.3 or dist < 100:  # Close enough or name matches
                    best_distance = dist
                    best_match = osm

        if best_match:
            matched += 1
            ore.osm_id = best_match['osm_id']
            ore.osm_match_distance_m = best_distance
            ore.substation_type = best_match.get('substation_type')

            # Enrich voltage_in from OSM only if it provides more specific info
            # (e.g., 90kV or 225kV instead of default 63kV)
            # Keep voltage_out at 20kV (standard HTA output for all postes sources)
            if best_match['voltage_max_kv'] and best_match['voltage_max_kv'] != ore.voltage_in_kv:
                ore.voltage_in_kv = best_match['voltage_max_kv']
                # Keep voltage_out at 20kV unless OSM has a different MV level
                osm_min = best_match.get('voltage_min_kv')
                if osm_min and osm_min >= 10 and osm_min <= 33:  # Valid HTA range
                    ore.voltage_out_kv = osm_min
                ore.voltage_levels = f"{int(ore.voltage_in_kv)}kV/{int(ore.voltage_out_kv)}kV"
                ore.voltage_source = "OSM"
                voltage_enriched += 1

    print(f"  Postes sources matched to OSM: {matched}/{len(ore_substations)} ({100*matched/len(ore_substations):.1f}%)")
    print(f"  Postes sources with voltage upgraded from OSM: {voltage_enriched}")

    return ore_substations


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_to_csv(substations: List[Substation], filepath: str):
    """Export substations to CSV."""
    records = [asdict(s) for s in substations]
    df = pd.DataFrame(records)

    # Reorder columns
    priority_cols = [
        'id', 'name', 'network_level', 'source',
        'lat', 'lon', 'coord_source', 'coord_quality',
        'voltage_in_kv', 'voltage_out_kv', 'voltage_levels', 'voltage_source',
        'capacity_available_mw', 'capacity_reserved_mw', 'capacity_total_mw',
        'operator', 'commune', 'departement', 'region',
        'function', 'substation_type',
        'osm_id', 'osm_match_score', 'osm_match_distance_m',
    ]

    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[[c for c in priority_cols if c in df.columns] + other_cols]

    df.to_csv(filepath, index=False)
    print(f"\nExported {len(df)} records to {filepath}")

    return df


def export_to_geojson(substations: List[Substation], filepath: str):
    """Export substations to GeoJSON."""
    features = []

    for sub in substations:
        if sub.lat is None or sub.lon is None:
            continue

        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [sub.lon, sub.lat]
            },
            'properties': asdict(sub)
        }
        features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'features': features,
        'metadata': {
            'generated': datetime.now().isoformat(),
            'total_features': len(features),
            'description': 'France Grid Infrastructure - Transmission & Distribution Source Substations'
        }
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(features)} features to {filepath}")


def print_summary(rte_subs: List[Substation], ore_subs: List[Substation]):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\n RTE TRANSMISSION SUBSTATIONS:")
    print(f"   Total: {len(rte_subs)}")
    with_coords = sum(1 for s in rte_subs if s.lat)
    with_voltage = sum(1 for s in rte_subs if s.voltage_in_kv)
    print(f"   With coordinates: {with_coords} ({100*with_coords/len(rte_subs):.1f}%)")
    print(f"   With voltage: {with_voltage} ({100*with_voltage/len(rte_subs):.1f}%)")

    print("\n POSTES SOURCES:")
    print(f"   Total: {len(ore_subs)}")
    with_coords = sum(1 for s in ore_subs if s.lat)
    with_voltage = sum(1 for s in ore_subs if s.voltage_in_kv)
    print(f"   With coordinates: {with_coords} ({100*with_coords/len(ore_subs):.1f}%)")
    print(f"   With voltage: {with_voltage} ({100*with_voltage/len(ore_subs):.1f}%)")

    # Voltage distribution
    print("\n VOLTAGE DISTRIBUTION (where available):")
    all_subs = rte_subs + ore_subs
    voltage_counts = {}
    for s in all_subs:
        if s.voltage_in_kv:
            v = int(s.voltage_in_kv)
            voltage_counts[v] = voltage_counts.get(v, 0) + 1

    for v in sorted(voltage_counts.keys(), reverse=True):
        print(f"   {v} kV: {voltage_counts[v]}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    print("\n" + "#"*60)
    print("# FRANCE GRID INFRASTRUCTURE DATABASE BUILDER")
    print("#"*60)

    # Step 1: Extract from official sources
    rte_raw = fetch_all_rte_substations()
    ore_raw = fetch_all_agence_ore_postes()

    # Step 2: Transform to unified format
    print("\n" + "="*60)
    print("TRANSFORMING DATA")
    print("="*60)

    rte_subs = transform_rte_records(rte_raw)
    print(f"  Transformed {len(rte_subs)} RTE records")

    ore_subs = transform_ore_records(ore_raw)
    print(f"  Transformed {len(ore_subs)} Agence ORE records")

    # Step 3: Extract OSM for enrichment
    osm_subs = fetch_osm_substations_france()

    # Step 4: Enrich with OSM
    rte_subs = enrich_rte_with_osm(rte_subs, osm_subs)
    ore_subs = enrich_ore_with_osm(ore_subs, osm_subs)

    # Summary
    print_summary(rte_subs, ore_subs)

    # Export
    print("\n" + "="*60)
    print("EXPORTING DATA")
    print("="*60)

    # Combined export
    all_subs = rte_subs + ore_subs
    export_to_csv(all_subs, '/home/user/grid_infrastructure/france_grid_database.csv')
    export_to_geojson(all_subs, '/home/user/grid_infrastructure/france_grid_database.geojson')

    # Separate exports
    export_to_csv(rte_subs, '/home/user/grid_infrastructure/france_rte_transmission.csv')
    export_to_csv(ore_subs, '/home/user/grid_infrastructure/france_postes_sources.csv')

    print("\n" + "#"*60)
    print("# COMPLETE")
    print("#"*60)

    return rte_subs, ore_subs


if __name__ == '__main__':
    main()
