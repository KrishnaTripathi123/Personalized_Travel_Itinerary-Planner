"""
Smart Destination Recommender Service (Tab 1 / Epic 3).
Analyzes user mood/vibe, budget level, party type, and weather season to compute top 3 matching cities.
"""

from typing import List, Optional
from datetime import datetime
from models.schemas import DestinationRecommendation, TripPreferences
from config.settings import CURATED_DESTINATIONS, CURRENCY_RATES
from services.llm_service import LLMService


class RecommenderService:
    @classmethod
    def get_month_name(cls, date_str: Optional[str] = None) -> str:
        """Returns the month name for the given date or current month."""
        try:
            if date_str:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                return dt.strftime("%B")
        except Exception:
            pass
        return datetime.now().strftime("%B")

    @classmethod
    def recommend_destinations(
        cls,
        vibe_preference: str,
        budget_level: str = "Moderate",
        travel_party: str = "Solo Traveler 🎒",
        travel_month: Optional[str] = None,
        provider: str = "demo",
        api_key: Optional[str] = None
    ) -> List[DestinationRecommendation]:
        """
        Recommends top 3 matching destinations.
        Uses algorithmic scoring against curated database with optional LLM reasoning.
        """
        active_month = travel_month or cls.get_month_name()
        
        # Algorithmic scoring
        scored_destinations = []
        clean_vibe = vibe_preference.split("🏖️")[0].split("🏔️")[0].split("🏛️")[0].split("🏙️")[0].split("🍲")[0].split("🌿")[0].split("🏄‍♂️")[0].split("🕯️")[0].strip().rstrip(",-")
        clean_party = travel_party.split("🎒")[0].split("💑")[0].split("👨‍👩‍👧‍👦")[0].split("👥")[0].split("💼")[0].strip()

        for dest in CURATED_DESTINATIONS:
            score = 50  # Base compatibility

            # 1. Vibe matching
            vibe_keywords = vibe_preference.lower()
            matching_vibes = [v for v in dest["vibes"] if any(w in v.lower() for w in vibe_keywords.split() if len(w) > 3)]
            if matching_vibes:
                score += 25
            elif any(word in [t.lower() for t in dest["highlight_tags"]] for word in vibe_keywords.split() if len(word) > 3):
                score += 15

            # 2. Budget compatibility
            if dest["budget_tier"].lower() == budget_level.lower():
                score += 15
            elif (budget_level == "Budget" and dest["budget_tier"] == "Moderate") or \
                 (budget_level == "Luxury" and dest["budget_tier"] == "Moderate"):
                score += 5
            elif budget_level == "Budget" and dest["budget_tier"] == "Luxury":
                score -= 15

            # 3. Weather / Season compatibility
            if active_month in dest.get("best_months", []):
                score += 10
            else:
                score += 2  # Still visitable

            # 4. Travel party compatibility
            party_lower = travel_party.lower()
            if "solo" in party_lower:
                if dest["name"] in ["Tokyo", "Kyoto", "Barcelona", "Manali & Solang", "Jaipur", "Rome"]:
                    score += 10
                else:
                    score += 4
            elif "couple" in party_lower or "romantic" in party_lower:
                if any("romantic" in v.lower() for v in dest["vibes"]) or dest["name"] in ["Amalfi Coast", "Kyoto", "Paris", "Manali & Solang", "Rome", "Bali"]:
                    score += 10
                else:
                    score += 4
            elif "family" in party_lower or "kid" in party_lower:
                if dest["name"] in ["Tokyo", "Zurich & Interlaken", "Bali", "Paris", "Cape Town"]:
                    score += 10
                else:
                    score += 4
            elif "friend" in party_lower or "group" in party_lower:
                if any(v in ["Beach & Coastal Escapes 🏖️", "Urban Skylines & Nightlife 🏙️", "Thrilling Outdoor Adventure 🏄‍♂️"] for v in dest["vibes"]):
                    score += 10
                else:
                    score += 4
            elif "business" in party_lower:
                if dest["name"] in ["Tokyo", "New York City", "Dubai", "Paris"]:
                    score += 10
                else:
                    score += 4

            # Normalize to max 98%
            final_score = min(98, max(60, score))
            scored_destinations.append((final_score, dest))

        # Sort descending by score
        scored_destinations.sort(key=lambda x: x[0], reverse=True)
        top_3 = scored_destinations[:3]

        recommendations: List[DestinationRecommendation] = []
        for score, dest in top_3:
            daily_usd = float(dest["avg_daily_usd"])
            daily_inr = round(daily_usd * CURRENCY_RATES["INR"], 1)

            why = (
                f"Exceptional match for {clean_vibe} seekers traveling as {clean_party}. "
                f"{dest['name']} in {dest['country']} boasts world-renowned {dest['highlight_tags'][0]} and {dest['highlight_tags'][1]}. "
                f"Visiting in {active_month} aligns perfectly with its {dest['climate_season']} season."
            )

            # Top attractions from clusters or highlights
            attractions = dest["highlight_tags"][:3]

            rec = DestinationRecommendation(
                name=dest["name"],
                country=dest["country"],
                vibe_match_score=score,
                highlight_tags=dest["highlight_tags"],
                estimated_daily_usd=daily_usd,
                estimated_daily_inr=daily_inr,
                best_season=f"{dest['climate_season']} (Prime: {', '.join(dest['best_months'][:3])})",
                why_recommended=why,
                top_attractions=attractions,
                weather_tag=f"Ideal Season ({active_month})",
                safety_badge="High Tourist Safety & Friendly Atmosphere"
            )
            recommendations.append(rec)

        return recommendations
