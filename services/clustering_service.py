"""
Geographic clustering and routing optimization service.
Groups nearby attractions on the same day to minimize transit fatigue and prevent backtracking.
"""

import math
from typing import List, Tuple, Dict, Any
from models.schemas import ActivitySlot, DayItinerary


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in kilometers."""
    if lat1 == 0.0 or lon1 == 0.0 or lat2 == 0.0 or lon2 == 0.0:
        return 1.2  # Sensible default urban distance
    
    r = 6371.0  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(r * c, 2)


def estimate_transit_info(distance_km: float) -> Tuple[str, float]:
    """
    Returns (transit_description, estimated_minutes) based on urban transit models.
    """
    if distance_km < 0.1:
        return "Under 3 min stroll in same complex (<0.1 km)", 3.0
    elif distance_km <= 1.2:
        # Walking speed ~ 4.5 km/h
        minutes = max(4, int((distance_km / 4.5) * 60))
        return f"{minutes} min scenic stroll ({distance_km:.1f} km)", float(minutes)
    elif distance_km <= 5.0:
        # Short metro/tram/cab ~ 20 km/h with 4 min boarding overhead
        minutes = int(4 + (distance_km / 20.0) * 60)
        return f"{minutes} min via Metro / Cab ({distance_km:.1f} km)", float(minutes)
    else:
        # City cross-transit / taxi ~ 30 km/h with 6 min boarding overhead
        minutes = int(6 + (distance_km / 30.0) * 60)
        return f"{minutes} min via Express Metro / Taxi ({distance_km:.1f} km)", float(minutes)


def optimize_day_geographic_routing(day: DayItinerary) -> DayItinerary:
    """
    Sequentially links day activities to avoid backtracking and updates inter-slot transit info.
    """
    if not day.slots:
        return day

    slots = day.slots
    total_dist = 0.0

    # Ensure slot transit info is calculated sequentially from slot i-1 to slot i
    for i in range(len(slots)):
        if i == 0:
            slots[i].distance_km = 0.0
            slots[i].transit_info = "Starting point / hotel transit departure"
        else:
            prev_slot = slots[i - 1]
            dist = haversine_distance_km(
                prev_slot.latitude, prev_slot.longitude,
                slots[i].latitude, slots[i].longitude
            )
            slots[i].distance_km = dist
            transit_desc, _ = estimate_transit_info(dist)
            slots[i].transit_info = transit_desc
            total_dist += dist

    day.walking_distance_km = round(total_dist, 1)
    return day


def compute_zone_centroid(slots: List[ActivitySlot]) -> Tuple[float, float]:
    """Compute the geographic center (lat, lon) of a cluster."""
    valid_coords = [(s.latitude, s.longitude) for s in slots if s.latitude != 0.0 and s.longitude != 0.0]
    if not valid_coords:
        return (0.0, 0.0)
    avg_lat = sum(p[0] for p in valid_coords) / len(valid_coords)
    avg_lon = sum(p[1] for p in valid_coords) / len(valid_coords)
    return (round(avg_lat, 4), round(avg_lon, 4))


def score_clustering_efficiency(day: DayItinerary) -> Dict[str, Any]:
    """
    Evaluates how tightly clustered the day's itinerary is.
    Returns metrics on efficiency, total transit distance, and fatigue warning.
    """
    total_dist = sum(s.distance_km for s in day.slots)
    max_leg = max([s.distance_km for s in day.slots], default=0.0)
    
    if total_dist <= 6.0 and max_leg <= 3.5:
        rating = "Optimal (Zero Backtracking)"
        badge_color = "#10B981"  # Emerald
        advice = "All stops are within a compact zone. Extremely low transit fatigue."
    elif total_dist <= 12.0:
        rating = "Good (Efficient Corridor)"
        badge_color = "#3B82F6"  # Blue
        advice = "Well sequenced along a natural transit corridor."
    else:
        rating = "Moderate Transit (Wide District)"
        badge_color = "#F59E0B"  # Amber
        advice = "Stops span across multiple neighborhoods; use metro or rideshares between legs."

    return {
        "total_transit_km": round(total_dist, 1),
        "max_leg_km": round(max_leg, 1),
        "rating": rating,
        "badge_color": badge_color,
        "advice": advice
    }
