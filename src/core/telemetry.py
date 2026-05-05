import math
from utils import haversine_distance

def find_closest_telemetry(telemetry_data, ref_lat, ref_lon, ref_alt):
    """Finds the timestamp in the new video closest in 3D space to the reference frame."""
    best_dist = float('inf')
    best_entry = None
    
    for entry in telemetry_data:
        lat = entry.get('latitude')
        lon = entry.get('longitude')
        alt = entry.get('rel_alt', entry.get('abs_alt', ref_alt))
        
        if lat is None or lon is None:
            continue
            
        h_dist = haversine_distance(ref_lat, ref_lon, lat, lon)
        v_dist = abs(ref_alt - alt)
        
        total_dist = math.sqrt(h_dist**2 + v_dist**2)
        
        if total_dist < best_dist:
            best_dist = total_dist
            best_entry = entry
            
    return best_entry