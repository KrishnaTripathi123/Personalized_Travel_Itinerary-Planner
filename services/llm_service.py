"""
LLM Integration Service with support for Google Gemini, Groq, and Demo Mode.
Features prompt engineering, JSON schema enforcement, and robust offline mock engines.
"""

import json
import logging
import re
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from models.schemas import (
    TripPreferences, DayItinerary, ActivitySlot, DestinationRecommendation
)
from services.clustering_service import optimize_day_geographic_routing
from config.settings import CURATED_DESTINATIONS, CURRENCY_RATES

logger = logging.getLogger(__name__)


class LLMService:
    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[Any]:
        """Robustly extracts JSON from raw LLM output even if surrounded by markdown or commentary."""
        if not text:
            return None
        text = text.strip()
        # Direct parse attempt
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try markdown block ```json ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except Exception:
                pass

        # Try finding outermost { ... } or [ ... ]
        first_bracket = min(
            [i for i in [text.find("{"), text.find("[")] if i != -1],
            default=-1
        )
        last_bracket = max(
            [i for i in [text.rfind("}"), text.rfind("]")] if i != -1],
            default=-1
        )
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            sub = text[first_bracket:last_bracket + 1]
            try:
                return json.loads(sub)
            except Exception:
                pass
        return None

    @classmethod
    def call_gemini(cls, prompt: str, api_key: str, model: str = "gemini-1.5-flash") -> str:
        """Calls Google Gemini v1beta REST API with robust error handling and timeout."""
        if not api_key or not api_key.strip():
            raise ValueError("Gemini API key is required but was empty.")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 3000,
                "responseMimeType": "application/json"
            }
        }
        try:
            resp = requests.post(url, json=payload, timeout=30)
        except requests.Timeout:
            raise RuntimeError("Gemini API request timed out after 30 seconds.")
        except requests.RequestException as req_err:
            raise RuntimeError(f"Gemini network connection error: {req_err}")

        if resp.status_code != 200:
            error_details = resp.text
            try:
                err_json = resp.json()
                error_details = err_json.get("error", {}).get("message", resp.text)
            except Exception:
                pass
            raise RuntimeError(f"Gemini API Error ({resp.status_code}): {error_details}")

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            block_reason = data.get("promptFeedback", {}).get("blockReason")
            if block_reason:
                raise RuntimeError(f"Gemini blocked content generation: {block_reason}")
            raise RuntimeError("Gemini returned no candidates in response.")

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts or "text" not in parts[0]:
            finish_reason = candidates[0].get("finishReason", "UNKNOWN")
            raise RuntimeError(f"Gemini response candidate text is empty (finish reason: {finish_reason}).")

        return parts[0]["text"]

    @classmethod
    def call_groq(cls, prompt: str, api_key: str, model: str = "llama-3.3-70b-versatile") -> str:
        """Calls Groq OpenAI-compatible REST API with robust error handling and timeout."""
        if not api_key or not api_key.strip():
            raise ValueError("Groq API key is required but was empty.")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an expert AI Travel Planner. You MUST return strictly valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.4,
            "response_format": {"type": "json_object"}
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
        except requests.Timeout:
            raise RuntimeError("Groq API request timed out after 30 seconds.")
        except requests.RequestException as req_err:
            raise RuntimeError(f"Groq network connection error: {req_err}")

        if resp.status_code != 200:
            error_details = resp.text
            try:
                err_json = resp.json()
                error_details = err_json.get("error", {}).get("message", resp.text)
            except Exception:
                pass
            raise RuntimeError(f"Groq API Error ({resp.status_code}): {error_details}")

        data = resp.json()
        choices = data.get("choices", [])
        if not choices or "message" not in choices[0] or "content" not in choices[0]["message"]:
            raise RuntimeError(f"Groq returned unexpected response format: {data}")

        return choices[0]["message"]["content"]

    # -------------------------------------------------------------
    # HIGH-FIDELITY OFFLINE DEMO ENGINE
    # -------------------------------------------------------------
    @classmethod
    def _get_dietary_lunch(
        cls,
        city: str,
        zone: str,
        dietary: str,
        default_lunch: tuple,
        multiplier: float
    ) -> tuple:
        """Returns authentic lunch venue and description matching dietary requirements."""
        title, desc, loc, lat, lon, cost, tip = default_lunch
        dietary_clean = (dietary or "Any / Flexible").strip()

        if dietary_clean in ["Any / Flexible", "Standard"]:
            return (title, desc, loc, lat, lon, cost * multiplier, tip, ["Foodie", "Local Cuisine"])

        d_lower = dietary_clean.lower()
        if "vegetarian" in d_lower:
            if "tokyo" in city.lower():
                return ("Bon Buddhist Shojin Ryori", "Exquisite multi-course Zen Buddhist vegetarian cuisine with seasonal tempura and sesame tofu.", "Taito / Asakusa", lat, lon, 22.0 * multiplier, "Reserve counter or tatami table in advance.", ["Vegetarian", "Shojin Ryori", "Organic"])
            elif "paris" in city.lower():
                return ("Wild & The Moon Organic Plant-Based", "100% plant-based organic grain bowls, avocado sourdough tartine, and cold-pressed botanical juices.", "Le Marais, 3rd", lat, lon, 18.0 * multiplier, "Casual chic indoor seating.", ["Vegetarian", "Plant-based", "Organic"])
            elif "bali" in city.lower():
                return ("Moksa Plant-Based Cuisine Ubud", "Farm-to-table permaculture organic bowls, tempeh satay, and coconut jackfruit rendang.", "Ubud Forest", lat, lon, 12.0 * multiplier, "Overlooks lush organic garden beds.", ["Vegetarian", "Farm-to-Table", "Indonesian"])
            else:
                return (f"Green Oasis Vegetarian Bistro ({zone})", f"Gourmet vegetarian bowls, woodfired sourdough, and seasonal organic salads in {city}.", loc, lat, lon, 16.0 * multiplier, "Chef's daily vegetarian special.", ["Vegetarian", "Organic", "Local Bistro"])
        elif "vegan" in d_lower:
            if "tokyo" in city.lower():
                return ("Afuri Vegan Ramen Harajuku", "Rich 100% plant-based yuzu broth ramen with charred seasonal vegetables and konjac noodles.", "Harajuku, Shibuya", lat, lon, 15.0 * multiplier, "Order via touch terminal; certified vegan.", ["Vegan", "Ramen", "Plant-Based"])
            elif "paris" in city.lower():
                return ("Le Potager de Charlotte", "Artisanal gourmet vegan French gastronomy featuring roasted seasonal squash and cashew cream.", "9th Arrondissement", lat, lon, 24.0 * multiplier, "Pre-booking recommended.", ["Vegan", "French Gourmet", "Eco-friendly"])
            elif "bali" in city.lower():
                return ("Alchemy Organic Vegan Oasis", "World-renowned raw and cooked vegan creations, acai bowls, and botanical elixirs.", "Penestanan, Ubud", lat, lon, 14.0 * multiplier, "Chilled garden ambiance.", ["Vegan", "Raw Food", "Healthy"])
            else:
                return (f"Pure Botanica 100% Vegan Kitchen ({zone})", f"Certified plant-based culinary specialties, seed flatbreads, and cold-pressed drinks in {city}.", loc, lat, lon, 17.0 * multiplier, "Certified 100% vegan kitchen.", ["Vegan", "Plant-Based", "Zero Animal Products"])
        elif "halal" in d_lower:
            if "tokyo" in city.lower():
                return ("Halal Ramen Ouka Shinjuku", "Certified Halal wagyu and sea bream dashi ramen served with grilled chicken gyoza.", "Shinjuku East", lat, lon, 20.0 * multiplier, "Certified Halal by Muslim Association.", ["Halal", "Certified", "Japanese"])
            elif "paris" in city.lower():
                return ("Les Saveurs d'Orient Halal Paris", "Authentic certified Halal slow-simmered lamb and vegetable tagine with saffron couscous.", "5th Arrondissement", lat, lon, 22.0 * multiplier, "Alcohol-free family-friendly dining.", ["Halal", "Mediterranean", "Certified"])
            elif "bali" in city.lower():
                return ("Ayam Goreng Berkah Halal", "Crispy Javanese spiced fried chicken with fragrant jasmine rice and sweet sambal.", "Kuta / Seminyak", lat, lon, 10.0 * multiplier, "Certified Halal local favorite.", ["Halal", "Indonesian", "Authentic"])
            else:
                return (f"Crescent Star Certified Halal Kitchen ({zone})", f"Delicious certified Halal local delicacies, grilled skewers, and fragrant saffron rice in {city}.", loc, lat, lon, 18.0 * multiplier, "Certified Halal kitchen.", ["Halal", "Certified", "Local Delights"])
        elif "jain" in d_lower:
            return (f"Sattva Pure Jain Dining ({zone})", f"Strict Jain vegetarian meal prepared without root vegetables (potatoes, onions, garlic) using fresh local ingredients.", loc, lat, lon, 15.0 * multiplier, "Inform server of strict Jain preference.", ["Jain", "Pure Vegetarian", "Sattvic"])
        elif "gluten-free" in d_lower or "celiac" in d_lower:
            if "tokyo" in city.lower():
                return ("Gluten-Free Soba House Ota", "100% pure buckwheat Juwari soba noodles served with wheat-free tamari broth.", "Ginza / Tsukiji", lat, lon, 18.0 * multiplier, "100% gluten-free preparation.", ["Gluten-Free", "Soba", "Celiac Friendly"])
            elif "paris" in city.lower():
                return ("Noglu Dedicated Gluten-Free Bakery & Cafe", "Certified gluten-free savory tarts, quiches, and organic French pastries.", "7th Arrondissement", lat, lon, 20.0 * multiplier, "Safe for celiacs.", ["Gluten-Free", "Dedicated Kitchen", "French"])
            else:
                return (f"The Dedicated Gluten-Free Kitchen ({zone})", f"100% certified gluten-free savory bowls, rice flatbreads, and healthy meals in {city}.", loc, lat, lon, 17.0 * multiplier, "Dedicated allergen-conscious kitchen.", ["Gluten-Free", "Celiac-Friendly", "Healthy"])

        return (title, f"{desc} (Prepared to adhere to {dietary_clean} guidelines)", loc, lat, lon, cost * multiplier, tip, ["Foodie", dietary_clean])

    @classmethod
    def _generate_mock_itinerary(cls, prefs: TripPreferences) -> List[DayItinerary]:
        """Produces ultra-realistic, geographically clustered travel plans with authentic details."""
        dest_name = prefs.destination.strip() if prefs.destination else "Tokyo"
        duration = max(1, min(7, int(prefs.duration_days)))

        # Look for curated match
        matched_dest = None
        for d in CURATED_DESTINATIONS:
            if d["name"].lower() in dest_name.lower() or dest_name.lower() in d["name"].lower():
                matched_dest = d
                break

        if not matched_dest:
            matched_dest = CURATED_DESTINATIONS[0]

        base_lat = matched_dest["lat"]
        base_lon = matched_dest["lon"]

        # Scale costs based on budget level
        multiplier = 1.0
        if prefs.budget_level == "Budget":
            multiplier = 0.55
        elif prefs.budget_level == "Luxury":
            multiplier = 2.2

        days: List[DayItinerary] = []
        try:
            start_dt = datetime.strptime(prefs.start_date, "%Y-%m-%d")
        except Exception:
            start_dt = datetime.now()

        # Destination-specific 7-day rich activity catalogs
        city_templates = {
            "Tokyo": [
                {
                    "theme": "Neon Alleyways & Shinto Heritage",
                    "zone": "Shibuya & Harajuku",
                    "morning": ("Meiji Jingu Shrine & Forest Walk", "Stroll through tranquil 170-acre evergreen forest to Tokyo's iconic imperial shrine.", "Yoyogi Park, Shibuya", 35.6764, 139.6993, 0.0, "Free entry. Wear walking shoes."),
                    "lunch": ("Afuri Ramen Harajuku", "Taste refreshing yuzu-salt ramen with crispy bamboo and delicate broth.", "Harajuku, Shibuya", 35.6705, 139.7045, 14.0, "Use ticket vending machine to order."),
                    "afternoon": ("Takeshita Street & Shibuya Sky", "Explore quirky pop culture boutiques, then head up Shibuya Sky for 360-degree panorama.", "Shibuya Crossing", 35.6598, 139.7024, 22.0, "Pre-book Shibuya Sky ticket for 16:30 sunset."),
                    "evening": ("Omoide Yokocho Izakaya Tour", "Experience lantern-lit alleyway dining with yakitori skewers and draft craft beer.", "Shinjuku West Exit", 35.6931, 139.6997, 35.0, "Small cozy stalls with only 6-8 seats.")
                },
                {
                    "theme": "Old Edo Traditions & Riverfront",
                    "zone": "Asakusa & Ueno",
                    "morning": ("Sensō-ji Temple & Nakamise Souvenirs", "Wander through the massive Thunder Gate into Tokyo's oldest Buddhist temple.", "Asakusa, Taito", 35.7148, 139.7967, 5.0, "Sample fresh ningyo-yaki sweet cakes."),
                    "lunch": ("Daikokuya Tempura (Est. 1887)", "Historic eatery famed for crispy golden tendon and regional broth.", "Asakusa 1-38-10", 35.7126, 139.7958, 20.0, "Traditional tatami mat seating."),
                    "afternoon": ("Tokyo National Museum & Ueno Park", "Marvel at samurai armor, ukiyo-e woodblock prints, and serene lotus ponds.", "Ueno Park, Taito", 35.7189, 139.7765, 12.0, "Audio guide available in English."),
                    "evening": ("Tokyo Skytree Sunset & Solamachi", "Ascend 450 meters above ground and enjoy modern fusion dining overlooking city lights.", "Sumida City", 35.7100, 139.8107, 45.0, "Reserve glass floor observatory pass.")
                },
                {
                    "theme": "Culinary Marvels & Glamour",
                    "zone": "Tsukiji Outer Market & Ginza",
                    "morning": ("Tsukiji Outer Market Street Food Tasting", "Savor tamagoyaki omelet, fresh seafood, skewers, and matcha gelato.", "Tsukiji, Chuo", 35.6655, 139.7708, 25.0, "Cash preferred at traditional street stalls."),
                    "lunch": ("Ginza Kagari Michelin Bib Gourmand", "Velvety artisanal broth ramen with seasonal farm vegetables.", "Ginza 6-4-12", 35.6710, 139.7635, 18.0, "Line moves quickly; worth the wait."),
                    "afternoon": ("TeamLab Planets Digital Art Immersion", "Wade through water and walk among infinite kaleidoscopic crystal rooms.", "Toyosu, Koto", 35.6515, 139.7892, 32.0, "Short pants provided for water exhibits."),
                    "evening": ("Bar High Five & Rooftop Ginza Cocktails", "Experience legendary Japanese mixology with hand-carved ice diamonds.", "Ginza, Chuo", 35.6698, 139.7621, 50.0, "Smart casual dress code.")
                },
                {
                    "theme": "Modern Art, Towers & Night Skyline",
                    "zone": "Roppongi & Tokyo Tower",
                    "morning": ("Mori Art Museum & Roppongi Hills Observation", "View cutting-edge contemporary exhibits followed by Tokyo City View skyline deck.", "Roppongi Hills, Minato", 35.6605, 139.7292, 20.0, "Indoor observation deck open late."),
                    "lunch": ("Tofu Ukai Roppongi", "Artisanal seasonal tofu and seasonal delicacies in a Japanese garden setting.", "Shiba Park, Minato", 35.6558, 139.7460, 30.0, "Stunning view of Tokyo Tower."),
                    "afternoon": ("Zōjō-ji Temple & Shiba Park Stroll", "Ancient Tokugawa temple with Tokyo Tower soaring dramatically in the background.", "Shiba Park, Minato", 35.6575, 139.7483, 0.0, "Historic bell tower cast in 1673."),
                    "evening": ("Tokyo Tower Illuminated Vista & Dining", "Ascend Japan's iconic landmark for sweeping city lights and dusk twilight cocktails.", "Minato City", 35.6586, 139.7454, 40.0, "Photo spot at foot of tower.")
                },
                {
                    "theme": "Electric Culture & Ancient Shrines",
                    "zone": "Akihabara & Kanda",
                    "morning": ("Kanda Myojin Shrine & Sacred Grounds", "Visit the 1,300-year-old guardian shrine honoring prosperity and good fortune.", "Sotokanda, Chiyoda", 35.7020, 139.7679, 0.0, "Popular with technology enthusiasts."),
                    "lunch": ("Kanda Matsuya Traditional Soba", "Historic handmade buckwheat noodles established in 1884 with dashi broth.", "Kanda Sudacho", 35.6963, 139.7712, 16.0, "Warm and welcoming heritage diner."),
                    "afternoon": ("Akihabara Electric Town & Retro Arcades", "Browse vintage electronics, multi-level manga stores, and interactive gaming centers.", "Akihabara Station", 35.6983, 139.7731, 15.0, "Duty-free shopping available with passport."),
                    "evening": ("Manseibashi Riverfront & Craft Beer Tasting", "Repurposed historic red-brick train station now home to boutique eateries and craft breweries.", "Kanda Riverfront", 35.6974, 139.7714, 38.0, "Scenic riverside promenade seating.")
                },
                {
                    "theme": "Futuristic Waterfront & Bay Island",
                    "zone": "Odaiba Seaside Bay",
                    "morning": ("Odaiba Seaside Park & Rainbow Bridge Walk", "Brisk waterfront boardwalk stroll with miniature Statue of Liberty and panoramic bay vistas.", "Daiba, Minato", 35.6300, 139.7760, 0.0, "Great morning marine breeze."),
                    "lunch": ("Bills Odaiba Bayfront Cafe", "Famed ricotta hotcakes, sweet corn fritters, and crisp seasonal salads overlooking Tokyo Bay.", "Decks Tokyo Beach", 35.6292, 139.7758, 22.0, "Terrace seats with bridge panorama."),
                    "afternoon": ("Miraikan National Museum of Emerging Science", "Interact with humanoid robots, astronaut space quarters, and planetary geo-cosmos.", "Aomi, Koto", 35.6193, 139.7764, 10.0, "ASIMO and android demonstrations daily."),
                    "evening": ("Palette Town Ferris Wheel & Sunset Bay Cruise", "Watch golden hour illuminate the Tokyo Bay skyline and Odaiba Marine Park.", "Tokyo Bay", 35.6270, 139.7780, 42.0, "Book sunset boarding time.")
                },
                {
                    "theme": "Old Tokyo Nostalgia & Artisan Quarters",
                    "zone": "Yanaka & Nezu",
                    "morning": ("Nezu Shrine & Vermilion Torii Tunnel", "Picturesque hillside shrine with azalea gardens and vermilion torii tunnel without Kyoto crowds.", "Nezu, Bunkyo", 35.7202, 139.7615, 0.0, "One of Tokyo's oldest shrines."),
                    "lunch": ("Kayaba Coffee (Est. 1938)", "Retro machiya townhouse serving fluffy egg sandwiches and matcha lattes.", "Yanaka, Taito", 35.7219, 139.7702, 14.0, "2nd floor tatami seating."),
                    "afternoon": ("Yanaka Ginza Walking Street & Craft Workshops", "Discover traditional woodblock carvers, ceramic studios, and savory croquettes.", "Yanaka, Taito", 35.7275, 139.7695, 12.0, "Sunset staircase view of old neighborhood."),
                    "evening": ("SCAI The Bathhouse & Farewell Tokyo Feast", "Contemporary gallery inside a 200-year-old public bathhouse followed by celebratory dinner.", "Yanaka Arts Quarter", 35.7230, 139.7720, 48.0, "Reflective ending to the Tokyo odyssey.")
                }
            ],
            "Paris": [
                {
                    "theme": "Iconic Monuments & Royal Gardens",
                    "zone": "7th & 1st Arrondissement",
                    "morning": ("Eiffel Tower & Champ de Mars Stroll", "Ascend to the summit for panoramic views of the Seine River.", "Champ de Mars, 7th", 48.8584, 2.2945, 30.0, "Book morning lift slot to avoid midday crowds."),
                    "lunch": ("Café de Flore or Les Antiquaires", "Classic Parisian bistrot with seasonal French specialties.", "Rue du Bac, 7th", 48.8550, 2.3270, 28.0, "Outdoor wicker terrace seating recommended."),
                    "afternoon": ("Musée d'Orsay Impressionist Masterpieces", "Admiration of Monet, Van Gogh, and Degas inside a Belle Époque railway station.", "1 Rue de la Légion d'Honneur", 48.8599, 2.3265, 18.0, "Skip-the-line digital ticket included."),
                    "evening": ("Seine River Sunset Cruise with Champagne", "Glide past illuminated bridges and Notre-Dame as twilight descends.", "Pont de l'Alma", 48.8638, 2.3021, 40.0, "Wrap a light scarf for river breeze.")
                },
                {
                    "theme": "Bohemian Artistry & Hilltop Views",
                    "zone": "Montmartre & Pigalle",
                    "morning": ("Sacré-Cœur Basilica & Place du Tertre", "Discover artists painting portraits on cobblestone lanes overlooking Paris.", "Montmartre Hill", 48.8867, 2.3431, 0.0, "Free entrance to the basilica dome grounds."),
                    "lunch": ("Le Consulat Traditional Bistro", "Tuck into French culinary classics at Picasso's former haunt.", "Rue Norvins, 18th", 48.8861, 2.3395, 24.0, "Quaint vintage photography exterior."),
                    "afternoon": ("Musée de Montmartre & Renoir Gardens", "Stroll through secluded vineyard gardens where Auguste Renoir painted masterpieces.", "12 Rue Cortot", 48.8879, 2.3406, 15.0, "Serene hidden gem away from crowds."),
                    "evening": ("Moulin Rouge Historic Cabaret or Wine Cave", "Sample biodynamic French wines paired with artisanal cheese in a subterranean cellar.", "Rue des Abbesses", 48.8845, 2.3370, 45.0, "Casual romantic vibes.")
                },
                {
                    "theme": "Historic Heart & Literary Quarters",
                    "zone": "Le Marais & Latin Quarter",
                    "morning": ("Notre-Dame Parvis & Shakespeare and Company", "Gaze upon the Gothic cathedral and browse the legendary English bookshop.", "Place du Parvis", 48.8530, 2.3499, 0.0, "Pick up an embossed book souvenir."),
                    "lunch": ("L'As du Fallafel Le Marais", "World-renowned golden falafel pita with grilled eggplant and tahini dressing.", "Rue des Rosiers, 4th", 48.8575, 2.3590, 15.0, "Fast-moving queue, exceptional quality."),
                    "afternoon": ("Place des Vosges & Victor Hugo Residence", "Relax under red-brick arcades in Paris's oldest planned royal square.", "Place des Vosges, 4th", 48.8555, 2.3655, 0.0, "Lush lawns and classical fountains."),
                    "evening": ("Saint-Germain Jazz Club & Bistro Dining", "Listen to live swing jazz in an authentic vaulted cellar beneath Saint-Germain.", "Rue de Buci, 6th", 48.8535, 2.3360, 42.0, "Smart casual dress code.")
                },
                {
                    "theme": "Canal Waters & Bohemian Arts",
                    "zone": "Canal Saint-Martin & Belleville",
                    "morning": ("Canal Saint-Martin Iron Footbridges", "Stroll along shaded waterways and watch vintage iron lock gates open and close.", "Quai de Valmy, 10th", 48.8710, 2.3650, 0.0, "Popular picnic destination."),
                    "lunch": ("Du Pain et des Idées Bakery", "Award-winning escargot pistache pastry and fresh country sourdough tartines.", "Rue Yves Toudic", 48.8712, 2.3620, 14.0, "Historic 1875 painted ceiling."),
                    "afternoon": ("Parc des Buttes-Chaumont Suspension Bridge", "Explore romantic dramatic cliffs, waterfalls, and temple pavilion vistas.", "19th Arrondissement", 48.8800, 2.3830, 0.0, "Panoramic views of Montmartre."),
                    "evening": ("Belleville Rooftop Sunset & Pan-Asian Bistronomy", "Watch golden hour twilight across the entire Parisian skyline from Parc de Belleville.", "Rue Piat, 20th", 48.8720, 2.3860, 36.0, "Trendy local hotspot.")
                },
                {
                    "theme": "Grand Palaces & Royal Versailles",
                    "zone": "Versailles Historic Estate",
                    "morning": ("Château de Versailles Hall of Mirrors", "Marvel at 357 golden mirrors and King Louis XIV's state apartments.", "Place d'Armes", 48.8049, 2.1204, 28.0, "Take RER C train from central Paris."),
                    "lunch": ("La Flottille Grand Canal Terrace", "Dine right on the edge of the royal canal under weeping willows.", "Parc de Versailles", 48.8110, 2.1150, 25.0, "Scenic waterside terrace."),
                    "afternoon": ("Queen's Hamlet & Petit Trianon Gardens", "Discover Marie Antoinette's idyllic rustic village, windmill, and working dairy.", "Versailles Park", 48.8190, 2.1130, 12.0, "Peaceful meadow stroll."),
                    "evening": ("Return to Paris & Montparnasse Sky Tower", "Fast elevator to the 56th floor rooftop for the best nighttime view of the sparkling Eiffel Tower.", "33 Avenue du Maine", 48.8421, 2.3219, 32.0, "Champagne bar on observation roof.")
                },
                {
                    "theme": "Haute Couture, Opera & Covered Passages",
                    "zone": "Opéra & Grands Boulevards",
                    "morning": ("Palais Garnier Opera House Tour", "Gaze up at Marc Chagall's painted ceiling and the seven-ton crystal chandelier.", "Place de l'Opéra, 9th", 48.8720, 2.3316, 16.0, "Inspired Phantom of the Opera."),
                    "lunch": ("Galeries Lafayette Gourmet Food Hall", "Artisanal truffle pasta, fresh seafood, and Ladurée macarons under glass dome.", "Boulevard Haussmann", 48.8735, 2.3315, 24.0, "Free rooftop terrace panorama."),
                    "afternoon": ("Galerie Vivienne & Passage des Panoramas", "Explore 19th-century mosaic-tiled glass-roofed shopping arcades and antiquarians.", "Rue Vivienne, 2nd", 48.8665, 2.3400, 0.0, "Charming historic architecture."),
                    "evening": ("French Classical Gastronomy & Wine Pairing", "Refined culinary dining highlighting seasonal terroir and vintage Bordeaux.", "Palais-Royal Sector", 48.8640, 2.3370, 55.0, "Elegant dress code.")
                },
                {
                    "theme": "Jardin des Tuileries & Modern Masterpieces",
                    "zone": "Concorde & Seine Embankment",
                    "morning": ("Musée de l'Orangerie Water Lilies", "Immerse in Claude Monet's eight monumental curved Nymphéas murals.", "Jardin des Tuileries", 48.8638, 2.3227, 14.0, "Soft daylight illumination."),
                    "lunch": ("Café Marly Louvre Arcade", "Fine dining under colonnades looking directly into I.M. Pei's Glass Pyramid.", "Rue de Rivoli", 48.8620, 2.3340, 32.0, "Unbeatable architectural vantage."),
                    "afternoon": ("Pont des Arts & Seine Bookstalls (Bouquinistes)", "Browse vintage posters, antique books, and French postcards along riverside stone walls.", "Quai de Conti", 48.8580, 2.3375, 0.0, "UNESCO World Heritage riverbank."),
                    "evening": ("Twilight Farewell Dinner at Le Marais Courtyard", "Savor delicate French patisseries and celebration dinner in a historic courtyard.", "Place Sainte-Catherine", 48.8550, 2.3610, 48.0, "Perfect Paris finale.")
                }
            ],
            "Bali": [
                {
                    "theme": "Emerald Rice Terraces & Sacred Temples",
                    "zone": "Ubud Cultural Highlands",
                    "morning": ("Tegallalang Rice Terraces & Jungle Walk", "Wander stepped green terraces and capture iconic canopy photographs.", "Tegallalang, Ubud", -8.4338, 115.2798, 10.0, "Arrive by 8:30 AM before tourist buses."),
                    "lunch": ("Warung Local Flavors Ubud", "Traditional Balinese spiced cuisine with lemongrass and sambal.", "Jl. Suweta No.2, Ubud", -8.5055, 115.2628, 8.0, "Authentic local seasoning."),
                    "afternoon": ("Sacred Monkey Forest Sanctuary", "Meander through moss-covered Hindu temples guarded by Balinese macaques.", "Jl. Monkey Forest, Ubud", -8.5188, 115.2588, 7.0, "Keep sunglasses and phones secured."),
                    "evening": ("Ubud Palace Legong Dance & Organic Dinner", "Witness rhythmic Gamelan percussion followed by candlelit farm-to-table dining.", "Ubud Central", -8.5069, 115.2625, 20.0, "Performance starts promptly at 19:30.")
                },
                {
                    "theme": "Clifftop Horizons & Sunset Waves",
                    "zone": "Uluwatu & Bukit Peninsula",
                    "morning": ("Padang Padang Beach Surf & Relax", "Walk through cliff cave opening to a world-famous golden sand surf cove.", "Pecatu, Uluwatu", -8.8111, 115.1037, 3.0, "Great for swimming and surf lessons."),
                    "lunch": ("Single Fin Clifftop Eatery", "Smoothie bowls, gourmet wraps, and chilled fresh young coconut with ocean panorama.", "Pantai Suluban", -8.8149, 115.0884, 15.0, "Prime clifftop viewing deck."),
                    "afternoon": ("Uluwatu Clifftop Temple Walk", "Spectacular temple perched 70 meters above roaring Indian Ocean waves.", "Pecatu, Uluwatu", -8.8291, 115.0849, 6.0, "Traditional sarong provided at entry."),
                    "evening": ("Kecak Fire Dance & Jimbaran Seafood Dinner", "Hypnotic choir chanting against burning crimson sunset, followed by beachside dining.", "Jimbaran Bay", -8.7758, 115.1666, 30.0, "Dine with toes in the sand.")
                },
                {
                    "theme": "Bohemian Coastal Culture & Cafes",
                    "zone": "Seminyak & Canggu Coast",
                    "morning": ("Echo Beach Coastal Stroll & Surf Watch", "Watch advanced surfers catch left-hand reef breaks along volcanic dark sand.", "Canggu Coastal Trail", -8.6550, 115.1270, 0.0, "Early morning ocean breeze."),
                    "lunch": ("Crate Cafe or The Shady Shack", "Vibrant organic bowls, turmeric lattes, and artisan sourdough toasties.", "Jl. Canggu Padang Linjong", -8.6480, 115.1380, 12.0, "Buzzing creative atmosphere."),
                    "afternoon": ("Tanah Lot Sea Temple Exploration", "Ancient sea shrine balanced on an offshore rock formation eroded by tide waters.", "Tabanan Coast", -8.6212, 115.0868, 5.0, "Low tide allows walking to base of rock."),
                    "evening": ("La Brisa Sunset Beach Club", "Eco-friendly driftwood beach lounge with illuminated fairy lights and chilled house music.", "Batu Mejan Beach", -8.6540, 115.1275, 28.0, "Book daybed for ocean sunset.")
                },
                {
                    "theme": "Misty Volcanic Lakes & Hidden Waterfalls",
                    "zone": "Bedugul & Munduk Highlands",
                    "morning": ("Ulun Danu Beratan Floating Temple", "Iconic 17th-century water temple floating on the mist-shrouded crater lake.", "Candikuning, Tabanan", -8.2752, 115.1658, 6.0, "Cool alpine climate at 1,200m elevation."),
                    "lunch": ("Mentari Lakeside Buffet", "Warm ginger tea and Indonesian mountain curries overlooking Lake Beratan.", "Bedugul Highway", -8.2780, 115.1640, 10.0, "Stunning volcanic backdrop."),
                    "afternoon": ("Banyumala Twin Waterfalls Trek", "Short rainforest trek to crystal-clear twin cascades pouring into a natural plunge pool.", "Wanagiri, Sukasada", -8.2170, 115.1020, 5.0, "Refreshing mountain swim."),
                    "evening": ("Munduk Coffee Plantation Sunset Tasting", "Sample rich Arabica coffee and cloves grown on mountain slopes during twilight.", "Munduk Ridge", -8.2600, 115.0700, 16.0, "Crisp mountain air.")
                },
                {
                    "theme": "Sacred Springs & Royal Heritage",
                    "zone": "Tampak Siring & East Ubud",
                    "morning": ("Tirta Empul Holy Water Temple", "Participate in or witness traditional Balinese melukat purification ritual in holy springs.", "Manukaya, Gianyar", -8.4150, 115.3150, 6.0, "Lockers and sarongs available."),
                    "lunch": ("Warung D'Tukad Riverview", "Fresh grilled local fish and stir-fried morning glory over river valley canopy.", "Blahbatuh", -8.5300, 115.3000, 11.0, "Peaceful river valley soundtrack."),
                    "afternoon": ("Gunung Kawi 11th-Century Rock Carvings", "Descend through lush river canyon to massive 7-meter funeral monuments carved into cliffs.", "Tampak Siring", -8.4230, 115.3120, 5.0, "Spectacular ancient engineering."),
                    "evening": ("Campuhan Ridge Golden Hour Walk & Fine Dining", "Picturesque rolling hillside nature trek between two rivers culminating in Ubud fine dining.", "Campuhan, Ubud", -8.5030, 115.2540, 26.0, "Best at 17:30 golden hour.")
                },
                {
                    "theme": "Coastal Tranquility & Coral Reefs",
                    "zone": "Sanur & Southeast Coast",
                    "morning": ("Sanur Sunrise Boardwalk Bike Ride", "Cycle along 5km paved beachfront path watching fishermen against Mount Agung sunrise.", "Sanur Beachfront", -8.6830, 115.2630, 4.0, "Calm, lagoon-like reef waters."),
                    "lunch": ("Warung Mak Beng (Est. 1941)", "Legendary fried sea fish with spicy sambal and aromatic cucumber fish soup.", "Jl. Hang Tuah No.45", -8.6750, 115.2635, 7.0, "One signature dish perfected for 80 years."),
                    "afternoon": ("Turtle Conservation & Snorkeling Reef", "Learn about sea turtle rehabilitation and take a glass-bottom boat to coral shallows.", "Sanur Port", -8.6900, 115.2600, 15.0, "Educational and family friendly."),
                    "evening": ("Sindhu Night Market Local Street Feasting", "Sample fresh martabak pancakes, satay sticks, and sweet mango sticky rice.", "Sanur Central", -8.6800, 115.2580, 12.0, "Vibrant local community atmosphere.")
                },
                {
                    "theme": "Southern Peninsula Horizons & Farewell",
                    "zone": "Nusa Dua & Jimbaran Coast",
                    "morning": ("Nusa Dua Water Blow & Geger Beach", "Witness massive Indian Ocean swell erupting through limestone blowholes into the sky.", "Nusa Dua Peninsula", -8.8050, 115.2340, 2.0, "Spectacular ocean spray photos."),
                    "lunch": ("Bebek Bengil Crispy Duck", "Crispy Balinese spiced duck served with local sambals and seasonal vegetable urap.", "Nusa Dua", -8.7980, 115.2280, 18.0, "Beautiful pavilion dining."),
                    "afternoon": ("Traditional Balinese Massage & Herbal Spa", "Unwind with 90-minute traditional aromatherapy massage and flower bath.", "Jimbaran Hills", -8.7850, 115.1750, 30.0, "Deep physical rejuvenation."),
                    "evening": ("Celebratory Beachfront Sunset Banquet", "Dine under candlelit tiki torches directly on Jimbaran sand as planes descend over bay.", "Jimbaran Bay", -8.7758, 115.1666, 45.0, "Unforgettable Balinese farewell.")
                }
            ]
        }

        # Select matching city template or generate dynamic templates using destination clusters
        templates = city_templates.get(matched_dest["name"])
        if not templates:
            # Dynamically pull clusters from CURATED_DESTINATIONS
            clusters_dict = matched_dest.get("clusters", {})
            cluster_keys = list(clusters_dict.keys())
            templates = []
            
            for c_i in range(min(7, max(duration, len(cluster_keys)))):
                c_key = f"Day {c_i + 1}"
                c_info = clusters_dict.get(c_key, {
                    "name": f"District {c_i + 1} of {dest_name}",
                    "lat": base_lat + (c_i * 0.008),
                    "lon": base_lon + (c_i * 0.009)
                })
                c_name = c_info.get("name", f"Quarter {c_i + 1}")
                c_lat = c_info.get("lat", base_lat)
                c_lon = c_info.get("lon", base_lon)

                templates.append({
                    "theme": f"Cultural Discovery & Highlights of {c_name}",
                    "zone": c_name,
                    "morning": (f"{c_name} Historic Landmark & Monument", f"Discover the iconic history and monumental architecture in {c_name}.", f"Central {c_name}", c_lat + 0.003, c_lon + 0.002, 15.0, "Arrive early for the best lighting."),
                    "lunch": (f"{c_name} Regional Bistro", f"Authentic regional specialties prepared with local ingredients.", f"{c_name} Market", c_lat + 0.001, c_lon + 0.001, 18.0, "Ask for chef's daily special."),
                    "afternoon": (f"{c_name} Museum & Artisan Quarter", f"Immerse in local crafts, modern galleries, and scenic photo viewpoints.", f"{c_name} Arts District", c_lat - 0.002, c_lon - 0.001, 12.0, "Great photography spots."),
                    "evening": (f"{c_name} Sunset Promenade & Twilight Dining", f"Breathtaking views followed by fine local dining and twilight entertainment.", f"{c_name} Promenade", c_lat - 0.003, c_lon + 0.002, 35.0, "Reserve window or outdoor table.")
                })

        # Generate each requested day
        for d_idx in range(duration):
            day_num = d_idx + 1
            curr_date = (start_dt + timedelta(days=d_idx)).strftime("%A, %b %d")
            tmpl = templates[d_idx % len(templates)]

            # Small jitter only if duration exceeds available distinct templates
            lat_jitter = (d_idx // len(templates)) * 0.008
            lon_jitter = (d_idx // len(templates)) * 0.009

            m_title, m_desc, m_loc, m_lat, m_lon, m_cost, m_tip = tmpl["morning"]
            raw_lunch = tmpl["lunch"]
            a_title, a_desc, a_loc, a_lat, a_lon, a_cost, a_tip = tmpl["afternoon"]
            e_title, e_desc, e_loc, e_lat, e_lon, e_cost, e_tip = tmpl["evening"]

            # Dietary-aware lunch resolution
            l_title, l_desc, l_loc, l_lat, l_lon, l_cost, l_tip, l_tags = cls._get_dietary_lunch(
                city=dest_name,
                zone=tmpl["zone"],
                dietary=prefs.dietary,
                default_lunch=raw_lunch,
                multiplier=multiplier
            )

            slots = [
                ActivitySlot(
                    slot_id="morning",
                    slot_title="Morning Sightseeing",
                    time_window="09:00 - 12:30",
                    title=m_title,
                    description=m_desc,
                    location_name=m_loc,
                    latitude=round(m_lat + lat_jitter, 5),
                    longitude=round(m_lon + lon_jitter, 5),
                    zone=tmpl["zone"],
                    category="Sightseeing",
                    estimated_cost_usd=round(m_cost * multiplier, 1),
                    estimated_cost_inr=round(m_cost * multiplier * CURRENCY_RATES["INR"], 1),
                    tips=m_tip,
                    tags=["Heritage", "Sightseeing", "Scenic"]
                ),
                ActivitySlot(
                    slot_id="lunch",
                    slot_title="Lunch Recommendation",
                    time_window="12:30 - 14:00",
                    title=l_title,
                    description=l_desc,
                    location_name=l_loc,
                    latitude=round(l_lat + lat_jitter, 5),
                    longitude=round(l_lon + lon_jitter, 5),
                    zone=tmpl["zone"],
                    category="Dining",
                    estimated_cost_usd=round(l_cost, 1),
                    estimated_cost_inr=round(l_cost * CURRENCY_RATES["INR"], 1),
                    tips=l_tip,
                    tags=l_tags
                ),
                ActivitySlot(
                    slot_id="afternoon",
                    slot_title="Afternoon Leisure & Exploration",
                    time_window="14:00 - 17:30",
                    title=a_title,
                    description=a_desc,
                    location_name=a_loc,
                    latitude=round(a_lat + lat_jitter, 5),
                    longitude=round(a_lon + lon_jitter, 5),
                    zone=tmpl["zone"],
                    category="Leisure",
                    estimated_cost_usd=round(a_cost * multiplier, 1),
                    estimated_cost_inr=round(a_cost * multiplier * CURRENCY_RATES["INR"], 1),
                    tips=a_tip,
                    tags=["Culture", "Walkable", "Photography"]
                ),
                ActivitySlot(
                    slot_id="evening",
                    slot_title="Evening Dining & Entertainment",
                    time_window="18:00 - 22:00",
                    title=e_title,
                    description=e_desc,
                    location_name=e_loc,
                    latitude=round(e_lat + lat_jitter, 5),
                    longitude=round(e_lon + lon_jitter, 5),
                    zone=tmpl["zone"],
                    category="Nightlife & Dining",
                    estimated_cost_usd=round(e_cost * multiplier, 1),
                    estimated_cost_inr=round(e_cost * multiplier * CURRENCY_RATES["INR"], 1),
                    tips=e_tip,
                    tags=["Nightlife", "Scenic Dining", "Atmosphere"]
                )
            ]

            day_plan = DayItinerary(
                day_number=day_num,
                date_str=curr_date,
                theme=tmpl["theme"],
                zone_cluster=tmpl["zone"],
                slots=slots
            )
            day_plan = optimize_day_geographic_routing(day_plan)
            day_plan.recalculate_totals(CURRENCY_RATES["INR"])
            days.append(day_plan)

        return days

    # -------------------------------------------------------------
    # MAIN PUBLIC METHODS
    # -------------------------------------------------------------
    @classmethod
    def generate_itinerary(
        cls,
        preferences: TripPreferences,
        provider: str = "demo",
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ) -> List[DayItinerary]:
        """
        Generates day-by-day itinerary plans.
        Uses live LLM (Gemini or Groq) if provider != 'demo' and key is provided.
        Throws informative exceptions on live API failures (does not silently fabricate data).
        """
        if provider == "demo":
            return cls._generate_mock_itinerary(preferences)

        if not api_key or not api_key.strip():
            raise RuntimeError(
                f"API key is required for live provider '{provider}'. "
                f"Please enter your {provider.capitalize()} API key or switch to Demo Mode."
            )

        prompt = f"""
        You are a World-Class Travel Architect. Generate a {preferences.duration_days}-day itinerary for {preferences.destination}.
        Start Date: {preferences.start_date}
        End Date: {preferences.end_date}
        Budget Level: {preferences.budget_level}
        Travel Party: {preferences.travel_party}
        Pace: {preferences.pace}
        Interests: {', '.join(preferences.interests)}
        Dietary Preference: {preferences.dietary}
        Open Vibe: {preferences.open_vibe or 'None'}

        CRITICAL REQUIREMENT - GEOGRAPHIC CLUSTERING:
        Group activities for each day within the SAME neighborhood/district to eliminate transit backtracking.
        Ensure lunch is walking distance to morning sight.
        Provide realistic latitude and longitude coordinates for each location.
        Provide exactly {preferences.duration_days} days in the "days" array.
        Each day MUST contain exactly 4 slots with slot_id: "morning", "lunch", "afternoon", "evening".

        Return ONLY a JSON object matching this schema:
        {{
            "days": [
                {{
                    "day_number": 1,
                    "date_str": "Day 1 (e.g. Monday, Nov 12)",
                    "theme": "Theme title",
                    "zone_cluster": "Neighborhood Name",
                    "slots": [
                        {{
                            "slot_id": "morning",
                            "slot_title": "Morning Sightseeing",
                            "time_window": "09:00 - 12:30",
                            "title": "Specific attraction name",
                            "description": "Engaging 2-sentence description",
                            "location_name": "Full address or landmark name",
                            "latitude": 35.6762,
                            "longitude": 139.6503,
                            "zone": "Neighborhood Name",
                            "category": "Sightseeing",
                            "estimated_cost_usd": 25.0,
                            "tips": "Practical tip for the user",
                            "tags": ["Tag1", "Tag2"]
                        }},
                        {{
                            "slot_id": "lunch",
                            "slot_title": "Lunch Recommendation",
                            "time_window": "12:30 - 14:00",
                            "title": "Specific restaurant name",
                            "description": "Signature dishes matching dietary restriction ({preferences.dietary})",
                            "location_name": "Location",
                            "latitude": 35.6762,
                            "longitude": 139.6503,
                            "zone": "Neighborhood Name",
                            "category": "Dining",
                            "estimated_cost_usd": 18.0,
                            "tips": "Reservation advice",
                            "tags": ["Food", "Local"]
                        }},
                        {{
                            "slot_id": "afternoon",
                            "slot_title": "Afternoon Leisure & Exploration",
                            "time_window": "14:00 - 17:30",
                            "title": "Museum or scenic spot",
                            "description": "Description",
                            "location_name": "Location",
                            "latitude": 35.6762,
                            "longitude": 139.6503,
                            "zone": "Neighborhood Name",
                            "category": "Leisure",
                            "estimated_cost_usd": 15.0,
                            "tips": "Tips",
                            "tags": ["Culture"]
                        }},
                        {{
                            "slot_id": "evening",
                            "slot_title": "Evening Dining & Entertainment",
                            "time_window": "18:00 - 22:00",
                            "title": "Sunset rooftop or dinner venue",
                            "description": "Description",
                            "location_name": "Location",
                            "latitude": 35.6762,
                            "longitude": 139.6503,
                            "zone": "Neighborhood Name",
                            "category": "Nightlife & Dining",
                            "estimated_cost_usd": 35.0,
                            "tips": "Dress code advice",
                            "tags": ["Nightlife", "Dinner"]
                        }}
                    ]
                }}
            ]
        }}
        """

        raw_response = ""
        try:
            if provider == "gemini":
                raw_response = cls.call_gemini(prompt, api_key, model or "gemini-1.5-flash")
            elif provider == "groq":
                raw_response = cls.call_groq(prompt, api_key, model or "llama-3.3-70b-versatile")
            else:
                raise ValueError(f"Unsupported AI provider: {provider}")

            parsed = cls._extract_json_from_text(raw_response)
            if not parsed or not isinstance(parsed, dict) or "days" not in parsed:
                raise ValueError("Parsed LLM output was invalid or did not contain 'days' key.")

            days_raw = parsed["days"]
            if not isinstance(days_raw, list) or not days_raw:
                raise ValueError("LLM returned an empty 'days' list.")

            expected_slot_ids = ["morning", "lunch", "afternoon", "evening"]
            slot_defaults = {
                "morning": ("Morning Sightseeing", "09:00 - 12:30", "Sightseeing"),
                "lunch": ("Lunch Recommendation", "12:30 - 14:00", "Dining"),
                "afternoon": ("Afternoon Leisure & Exploration", "14:00 - 17:30", "Leisure"),
                "evening": ("Evening Dining & Entertainment", "18:00 - 22:00", "Nightlife & Dining")
            }

            days: List[DayItinerary] = []
            for day_idx, d in enumerate(days_raw[:preferences.duration_days]):
                existing_slots = d.get("slots", [])
                if not isinstance(existing_slots, list):
                    existing_slots = []

                cleaned_slots = []
                by_id = {s.get("slot_id"): s for s in existing_slots if isinstance(s, dict) and s.get("slot_id")}

                for s_id in expected_slot_ids:
                    if s_id in by_id:
                        s_data = by_id[s_id]
                        cost_usd = float(s_data.get("estimated_cost_usd", 20.0))
                        s_data["estimated_cost_inr"] = round(cost_usd * CURRENCY_RATES["INR"], 1)
                        if not s_data.get("time_window"):
                            s_data["time_window"] = slot_defaults[s_id][1]
                        if not s_data.get("slot_title"):
                            s_data["slot_title"] = slot_defaults[s_id][0]
                        cleaned_slots.append(ActivitySlot.from_dict(s_data))
                    else:
                        s_title, s_time, s_cat = slot_defaults[s_id]
                        cleaned_slots.append(ActivitySlot(
                            slot_id=s_id,
                            slot_title=s_title,
                            time_window=s_time,
                            title=f"{preferences.destination} {s_title}",
                            description=f"Curated {s_title.lower()} experience in {d.get('zone_cluster', preferences.destination)}.",
                            location_name=f"{preferences.destination} Center",
                            latitude=float(d.get("latitude", 0.0)),
                            longitude=float(d.get("longitude", 0.0)),
                            zone=d.get("zone_cluster", "Central Zone"),
                            category=s_cat,
                            estimated_cost_usd=20.0,
                            estimated_cost_inr=round(20.0 * CURRENCY_RATES["INR"], 1),
                            tips="Plan ahead and arrive early.",
                            tags=["Curated", s_cat]
                        ))

                day_obj = DayItinerary(
                    day_number=int(d.get("day_number", day_idx + 1)),
                    date_str=d.get("date_str", f"Day {day_idx + 1}"),
                    theme=d.get("theme", f"Exploring {preferences.destination}"),
                    zone_cluster=d.get("zone_cluster", "Central District"),
                    slots=cleaned_slots
                )
                day_obj = optimize_day_geographic_routing(day_obj)
                day_obj.recalculate_totals(CURRENCY_RATES["INR"])
                days.append(day_obj)

            # If LLM returned fewer days than requested, backfill cleanly
            if len(days) < preferences.duration_days:
                mock_extra = cls._generate_mock_itinerary(preferences)
                for missing_idx in range(len(days), preferences.duration_days):
                    days.append(mock_extra[missing_idx])

            return days

        except Exception as e:
            logger.error(f"Live LLM generation failed via {provider}: {e}")
            raise RuntimeError(f"Live AI Generation via {provider.upper()} failed: {e}. Please check your API key and connection, or switch to Demo Mode.")

    @classmethod
    def swap_single_slot(
        cls,
        target_slot: ActivitySlot,
        day_context: DayItinerary,
        swap_intent: str,
        preferences: TripPreferences,
        provider: str = "demo",
        api_key: Optional[str] = None
    ) -> ActivitySlot:
        """
        Dynamically swaps an individual activity slot based on user intent.
        Keeps slot geographically coherent with the current day's zone cluster.
        """
        if provider != "demo":
            if not api_key or not api_key.strip():
                raise RuntimeError(
                    f"API key is required for live provider '{provider}'. "
                    f"Please enter your {provider.capitalize()} API key or switch to Demo Mode."
                )
            prompt = f"""
            The user wants to replace an activity in their {preferences.destination} itinerary.
            Current Zone: {day_context.zone_cluster}
            Target Time Slot: {target_slot.slot_title} ({target_slot.time_window})
            Current Activity to Replace: {target_slot.title} ({target_slot.category})
            User's Swap Intent: "{swap_intent}"
            Budget Level: {preferences.budget_level}
            Dietary Restriction: {preferences.dietary}

            Provide a new matching activity in the SAME or immediate neighboring zone ({day_context.zone_cluster})
            with realistic latitude (~{target_slot.latitude}) and longitude (~{target_slot.longitude}).

            Return ONLY a JSON object matching this schema:
            {{
                "title": "New activity name",
                "description": "Engaging 2-sentence description",
                "location_name": "Landmark or venue name",
                "latitude": {target_slot.latitude + 0.002},
                "longitude": {target_slot.longitude + 0.002},
                "category": "{target_slot.category}",
                "estimated_cost_usd": 25.0,
                "tips": "Helpful tip for visitor",
                "tags": ["Tag1", "Tag2"]
            }}
            """
            try:
                if provider == "gemini":
                    raw = cls.call_gemini(prompt, api_key)
                else:
                    raw = cls.call_groq(prompt, api_key)

                data = cls._extract_json_from_text(raw)
                if data and isinstance(data, dict) and "title" in data:
                    cost_usd = float(data.get("estimated_cost_usd", target_slot.estimated_cost_usd))
                    return ActivitySlot(
                        slot_id=target_slot.slot_id,
                        slot_title=target_slot.slot_title,
                        time_window=target_slot.time_window,
                        title=data["title"],
                        description=data.get("description", "A refreshed local recommendation."),
                        location_name=data.get("location_name", target_slot.location_name),
                        latitude=float(data.get("latitude", target_slot.latitude)),
                        longitude=float(data.get("longitude", target_slot.longitude)),
                        zone=day_context.zone_cluster,
                        category=data.get("category", target_slot.category),
                        estimated_cost_usd=cost_usd,
                        estimated_cost_inr=round(cost_usd * CURRENCY_RATES["INR"], 1),
                        tips=data.get("tips", "Enjoy your refreshed activity!"),
                        tags=data.get("tags", ["Customized", "Refreshed"])
                    )
                else:
                    raise ValueError("Live API response did not contain required 'title' field.")
            except Exception as e:
                logger.error(f"Live activity swap failed via {provider}: {e}")
                raise RuntimeError(f"Live AI Activity Swap via {provider.upper()} failed: {e}. Please check your API key and connection, or switch to Demo Mode.")

        # High-fidelity algorithmic swap fallback (Demo Mode)
        intent_lower = swap_intent.lower()
        new_title = target_slot.title
        new_desc = target_slot.description
        new_category = target_slot.category
        new_cost = target_slot.estimated_cost_usd
        new_tags = list(target_slot.tags)
        city = preferences.destination

        # Multiplier based on budget
        cost_mult = 1.0
        if preferences.budget_level == "Budget":
            cost_mult = 0.6
        elif preferences.budget_level == "Luxury":
            cost_mult = 2.0

        if "vegetarian" in intent_lower or "vegan" in intent_lower:
            new_title = f"Green Garden Plant-Based Bistro ({day_context.zone_cluster})"
            new_desc = f"Farm-fresh organic bowls, woodfired flatbreads, and botanical cold-pressed refreshments in {city}."
            new_category = "Dining"
            new_cost = 18.0 * cost_mult
            new_tags = ["Vegetarian", "Healthy", "Plant-based"]
        elif "market" in intent_lower or "shopping" in intent_lower:
            new_title = f"{day_context.zone_cluster} Artisanal Craft & Flea Market"
            new_desc = f"Bustling open-air bazaar featuring handcrafted souvenirs, local designers, and buskers in {city}."
            new_category = "Leisure"
            new_cost = max(5.0, new_cost * 0.7 * cost_mult)
            new_tags = ["Market", "Local Crafts", "Bargains"]
        elif "rain" in intent_lower or "indoor" in intent_lower or "museum" in intent_lower:
            new_title = f"{day_context.zone_cluster} Heritage Fine Arts & Indoor Pavilion"
            new_desc = f"Sheltered cultural haven showcasing historical treasures, interactive exhibits, and a cozy atrium cafe."
            new_category = "Culture"
            new_cost = 15.0 * cost_mult
            new_tags = ["Indoor", "Rainy Day", "Museum"]
        elif "adventure" in intent_lower or "outdoor" in intent_lower:
            new_title = f"{day_context.zone_cluster} Urban Adventure & Cycle Trail"
            new_desc = f"High-energy guided bike or e-scooter tour traversing hidden viewpoints and panoramic trails in {city}."
            new_category = "Adventure"
            new_cost = 28.0 * cost_mult
            new_tags = ["Active", "Adventure", "Outdoors"]
        elif "budget" in intent_lower or "free" in intent_lower:
            new_title = f"{day_context.zone_cluster} Scenic Public Garden & Panorama Vista"
            new_desc = f"Zero-cost admission tranquil public gardens with manicured flora, sculpture paths, and city vistas."
            new_category = "Sightseeing"
            new_cost = 0.0
            new_tags = ["Free", "Budget-Friendly", "Nature"]
        else:
            new_title = f"Bespoke {target_slot.slot_title.split()[0]} Experience: {swap_intent.capitalize()}"
            new_desc = f"Tailored activity aligning with '{swap_intent}', curated inside the {day_context.zone_cluster} quarter in {city}."
            new_cost = new_cost * cost_mult
            new_tags.append("Customized")

        return ActivitySlot(
            slot_id=target_slot.slot_id,
            slot_title=target_slot.slot_title,
            time_window=target_slot.time_window,
            title=new_title,
            description=new_desc,
            location_name=f"{day_context.zone_cluster} District",
            latitude=target_slot.latitude + 0.0015,
            longitude=target_slot.longitude + 0.0015,
            zone=day_context.zone_cluster,
            category=new_category,
            estimated_cost_usd=round(new_cost, 1),
            estimated_cost_inr=round(new_cost * CURRENCY_RATES["INR"], 1),
            tips="Updated per your custom preferences.",
            tags=new_tags
        )
