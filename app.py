"""
Personalized Travel Itinerary Planner - Production Streamlit Application
Skill Wallet Capstone Project 4
"""

import os
import json
import uuid
from datetime import datetime, date, timedelta
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt

from config.settings import (
    LLM_PROVIDERS, CURRENCY_RATES, TIME_SLOTS, VIBE_OPTIONS,
    BUDGET_TIERS, PARTY_TYPES, PACE_LEVELS, INTEREST_TAGS,
    CURATED_DESTINATIONS
)
from models.schemas import (
    TripPreferences, ActivitySlot, DayItinerary, TripRecord, DestinationRecommendation
)
from services.llm_service import LLMService
from services.clustering_service import (
    optimize_day_geographic_routing, score_clustering_efficiency, compute_zone_centroid
)
from services.recommender_service import RecommenderService
from services.storage_service import (
    init_db, save_trip, get_all_trips, get_trip_by_id, delete_trip,
    export_trip_json, import_trip_json
)
from services.pdf_service import generate_itinerary_pdf

# -------------------------------------------------------------
# STREAMLIT PAGE CONFIG & CSS
# -------------------------------------------------------------
st.set_page_config(
    page_title="Personalized Travel Itinerary Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Wanderlust CSS
css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize database
init_db()

# -------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------
if "active_trip" not in st.session_state:
    st.session_state.active_trip = None

if "trip_preferences" not in st.session_state:
    today = date.today()
    default_end = today + timedelta(days=3)
    st.session_state.trip_preferences = TripPreferences(
        destination="Tokyo",
        duration_days=3,
        start_date=str(today),
        end_date=str(default_end),
        budget_level="Moderate",
        travel_party=PARTY_TYPES[0],
        pace="Moderate",
        interests=["Historical Landmarks", "Local Street Food", "Scenic Photography"],
        dietary="Any / Flexible"
    )

if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

if "provider" not in st.session_state:
    st.session_state.provider = "demo"

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemini-1.5-flash"


# Helper to auto-generate default Tokyo trip on first boot for instant delight without duplicating
if st.session_state.active_trip is None:
    existing_trips = get_all_trips()
    if existing_trips:
        st.session_state.active_trip = existing_trips[0]
        st.session_state.trip_preferences = existing_trips[0].preferences
    else:
        demo_days = LLMService.generate_itinerary(st.session_state.trip_preferences, provider="demo")
        demo_trip = TripRecord(
            id="demo_tokyo",
            trip_name="Tokyo Highlights & Hidden Alleys",
            destination="Tokyo",
            duration_days=3,
            start_date=st.session_state.trip_preferences.start_date,
            end_date=st.session_state.trip_preferences.end_date,
            preferences=st.session_state.trip_preferences,
            days=demo_days
        )
        demo_trip.recalculate_totals(CURRENCY_RATES["INR"])
        st.session_state.active_trip = demo_trip
        save_trip(demo_trip)


# -------------------------------------------------------------
# HERO HEADER BANNER
# -------------------------------------------------------------
st.markdown("""
<div class="travel-hero">
    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div>
            <h1>Personalized Travel Itinerary Planner</h1>
            <p>Full-stack AI Travel Architect with geographic clustering, zero-backtracking routing, dynamic vibe customization, and publication-ready offline guides.</p>
        </div>
        <div style="text-align: right;">
            <span class="tag-badge amber">Skill Wallet Capstone 4</span><br/>
            <span style="font-size: 0.8rem; color: #CCFBF1; margin-top: 4px; display: inline-block;">Production Ready • Zero DB Setup</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# SIDEBAR: CONFIGURATION & USER PREFERENCES (EPIC 3)
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ AI Engine & Security")
    provider_choice = st.selectbox(
        "AI Provider",
        options=list(LLM_PROVIDERS.keys()),
        format_func=lambda k: LLM_PROVIDERS[k],
        index=0,
        help="Select Demo Mode for instant offline simulation, or provide Gemini / Groq API key."
    )
    st.session_state.provider = provider_choice

    api_key_input = ""
    if provider_choice == "gemini":
        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.api_key,
            help="Your key is held strictly in session memory and never persisted."
        )
        st.session_state.selected_model = st.selectbox(
            "Gemini Model",
            ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
        )
        st.session_state.api_key = api_key_input
        if api_key_input:
            st.success("🔑 Gemini Key Connected")
        else:
            st.info("💡 Enter key or switch to Demo Mode")

    elif provider_choice == "groq":
        api_key_input = st.text_input(
            "Groq API Key",
            type="password",
            value=st.session_state.api_key,
            help="Groq cloud inference key."
        )
        st.session_state.selected_model = st.selectbox(
            "Groq Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        )
        st.session_state.api_key = api_key_input
        if api_key_input:
            st.success("⚡ Groq Key Connected")
        else:
            st.info("💡 Enter key or switch to Demo Mode")
    else:
        st.success("🟢 Demo Mode Active (Instant & Offline)")

    st.markdown("---")
    st.markdown("### 🧭 Trip Preferences Form")

    # Trip Mode
    pref_mode = st.radio(
        "Planning Intent",
        ["Fixed Destination 🏙️", "Open Vibe / Mood 🏖️"],
        horizontal=True
    )

    dest_val = st.session_state.trip_preferences.destination
    vibe_val = None

    if pref_mode == "Fixed Destination 🏙️":
        popular_cities = [d["name"] for d in CURATED_DESTINATIONS]
        dest_val = st.selectbox(
            "Select or Type Destination",
            options=popular_cities + ["Other (Custom)"],
            index=popular_cities.index(dest_val) if dest_val in popular_cities else 0
        )
        if dest_val == "Other (Custom)":
            dest_val = st.text_input("Enter City Name", value="Kyoto")
    else:
        vibe_val = st.selectbox("Select Your Travel Vibe", VIBE_OPTIONS)
        dest_val = vibe_val.split()[0]  # Temporary anchor

    # Duration & Dates synchronized with preferences
    try:
        curr_start = datetime.strptime(st.session_state.trip_preferences.start_date, "%Y-%m-%d").date()
    except Exception:
        curr_start = date.today()

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_d = st.date_input("Start Date", value=curr_start)
    with col_d2:
        duration_input = st.slider("Duration (Days)", min_value=1, max_value=7, value=int(st.session_state.trip_preferences.duration_days))

    end_d = start_d + timedelta(days=duration_input - 1)
    st.caption(f"Trip Period: **{start_d.strftime('%b %d')}** to **{end_d.strftime('%b %d, %Y')}**")

    # Budget Tier
    budget_lvl = st.select_slider(
        "Budget Tier",
        options=["Budget", "Moderate", "Luxury"],
        value=st.session_state.trip_preferences.budget_level
    )
    b_info = BUDGET_TIERS[budget_lvl]
    st.caption(f"Estimated: **${b_info['min']} - ${b_info['max']}/day** ({b_info['desc']})")

    # Travel Party & Pace synchronized
    party_idx = PARTY_TYPES.index(st.session_state.trip_preferences.travel_party) if st.session_state.trip_preferences.travel_party in PARTY_TYPES else 0
    travel_party = st.selectbox("Travel Party", PARTY_TYPES, index=party_idx)

    pace_options = ["Relaxed", "Moderate", "Fast-Paced"]
    p_idx = pace_options.index(st.session_state.trip_preferences.pace) if st.session_state.trip_preferences.pace in pace_options else 1
    pace_choice = st.selectbox(
        "Travel Pace",
        pace_options,
        index=p_idx,
        help="Controls number of stops and transit buffers."
    )

    # Interests & Dietary synchronized
    initial_interests = [i for i in st.session_state.trip_preferences.interests if i in INTEREST_TAGS]
    if not initial_interests:
        initial_interests = INTEREST_TAGS[:3]

    interests_chosen = st.multiselect(
        "Core Interests (Multi-select)",
        INTEREST_TAGS,
        default=initial_interests
    )

    dietary_options = ["Any / Flexible", "Vegetarian", "Vegan", "Halal", "Gluten-Free", "Jain"]
    d_idx = dietary_options.index(st.session_state.trip_preferences.dietary) if st.session_state.trip_preferences.dietary in dietary_options else 0
    dietary_choice = st.selectbox(
        "Dietary & Lifestyle",
        dietary_options,
        index=d_idx
    )

    # Action Button
    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("🚀 Generate Optimized Itinerary", type="primary", use_container_width=True):
        updated_prefs = TripPreferences(
            destination=dest_val if pref_mode == "Fixed Destination 🏙️" else (vibe_val.split()[0] + " Haven"),
            duration_days=duration_input,
            start_date=str(start_d),
            end_date=str(end_d),
            budget_level=budget_lvl,
            travel_party=travel_party,
            pace=pace_choice,
            interests=interests_chosen if interests_chosen else ["Sightseeing", "Local Food"],
            open_vibe=vibe_val if pref_mode != "Fixed Destination 🏙️" else None,
            dietary=dietary_choice
        )
        st.session_state.trip_preferences = updated_prefs

        with st.spinner(f"Architecting clustered itinerary for {updated_prefs.destination}..."):
            try:
                days = LLMService.generate_itinerary(
                    updated_prefs,
                    provider=st.session_state.provider,
                    api_key=st.session_state.api_key,
                    model=st.session_state.selected_model
                )
                new_trip = TripRecord(
                    id=str(uuid.uuid4())[:8],
                    trip_name=f"{updated_prefs.destination} Odyssey",
                    destination=updated_prefs.destination,
                    duration_days=updated_prefs.duration_days,
                    start_date=updated_prefs.start_date,
                    end_date=updated_prefs.end_date,
                    preferences=updated_prefs,
                    days=days
                )
                new_trip.recalculate_totals(CURRENCY_RATES["INR"])
                st.session_state.active_trip = new_trip
                save_trip(new_trip)
                st.toast(f"Plan generated for {updated_prefs.destination}!", icon="✨")
                st.rerun()
            except Exception as e:
                st.error(f"Error generating plan: {e}")


# -------------------------------------------------------------
# MAIN APPLICATION TABS (EPIC 3, 4, 5)
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Tab 1: Smart Destination Recommender",
    "📅 Tab 2: Day-by-Day Clustered Itinerary",
    "✨ Tab 3: Vibe Customization & Slot Swap",
    "💾 Tab 4: Trip Management & PDF Export"
])


# =============================================================
# TAB 1: SMART DESTINATION RECOMMENDER (EPIC 3)
# =============================================================
with tab1:
    st.markdown("### 🌍 Discover Your Next Journey")
    st.markdown("Unsure where to travel? Let our recommender match top destinations against your budget, vibe mood, party type, and the ideal weather season.")

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        tab1_vibe = st.selectbox("Preferred Vibe", VIBE_OPTIONS, index=0, key="t1_vibe")
    with col_r2:
        tab1_budget = st.selectbox("Budget Tier", ["Budget", "Moderate", "Luxury"], index=1, key="t1_budget")
    with col_r3:
        tab1_party = st.selectbox("Party Type", PARTY_TYPES, index=0, key="t1_party")
    with col_r4:
        tab1_month = st.selectbox(
            "Travel Month",
            ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
            index=datetime.now().month - 1,
            key="t1_month"
        )

    if st.button("✨ Match Top 3 Destinations", type="primary"):
        with st.spinner("Analyzing seasonal weather patterns and compatibility scoring..."):
            st.session_state.recommendations = RecommenderService.recommend_destinations(
                vibe_preference=tab1_vibe,
                budget_level=tab1_budget,
                travel_party=tab1_party,
                travel_month=tab1_month,
                provider=st.session_state.provider,
                api_key=st.session_state.api_key
            )

    # Display recommendations
    recs = st.session_state.recommendations
    if not recs:
        # Pre-populate on initial render
        recs = RecommenderService.recommend_destinations(tab1_vibe, tab1_budget, tab1_party, tab1_month)
        st.session_state.recommendations = recs

    st.markdown("#### 🏆 Top Recommended Destinations For You")
    rec_cols = st.columns(3)

    for idx, rec in enumerate(recs[:3]):
        with rec_cols[idx]:
            st.markdown(f"""
            <div class="travel-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <span class="match-score-pill">{rec.vibe_match_score}% Match</span>
                    <span style="font-size: 0.85rem; color: #64748B;">📍 {rec.country}</span>
                </div>
                <h3 style="margin: 0 0 0.5rem 0; color: #0B2545;">{rec.name}</h3>
                <div style="margin-bottom: 0.8rem;">
                    {" ".join([f"<span class='tag-badge'>{tag}</span>" for tag in rec.highlight_tags[:3]])}
                </div>
                <p style="font-size: 0.9rem; color: #334155; line-height: 1.45; min-height: 70px;">
                    {rec.why_recommended}
                </p>
                <div style="background: #F8FAFC; padding: 0.75rem; border-radius: 8px; margin: 0.8rem 0; font-size: 0.85rem;">
                    <div>💵 <b>Est. Daily:</b> ${rec.estimated_daily_usd:.0f} USD (₹{rec.estimated_daily_inr:,.0f})</div>
                    <div>🌤️ <b>Season:</b> {rec.best_season}</div>
                    <div>🛡️ <b>Safety:</b> {rec.safety_badge}</div>
                </div>
                <div style="font-size: 0.82rem; color: #475569; margin-bottom: 1rem;">
                    <b>Must-Visit:</b> {", ".join(rec.top_attractions)}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"👉 Plan Trip to {rec.name}", key=f"btn_plan_{rec.name}_{idx}", use_container_width=True):
                # Update preferences and generate new trip
                new_prefs = TripPreferences(
                    destination=rec.name,
                    duration_days=st.session_state.trip_preferences.duration_days,
                    start_date=st.session_state.trip_preferences.start_date,
                    end_date=st.session_state.trip_preferences.end_date,
                    budget_level=tab1_budget,
                    travel_party=tab1_party,
                    pace="Moderate",
                    interests=st.session_state.trip_preferences.interests,
                    dietary=st.session_state.trip_preferences.dietary
                )
                st.session_state.trip_preferences = new_prefs
                with st.spinner(f"Building {rec.name} itinerary..."):
                    days = LLMService.generate_itinerary(new_prefs, provider=st.session_state.provider, api_key=st.session_state.api_key)
                    new_trip = TripRecord(
                        id=str(uuid.uuid4())[:8],
                        trip_name=f"{rec.name} Discovery Journey",
                        destination=rec.name,
                        duration_days=new_prefs.duration_days,
                        start_date=new_prefs.start_date,
                        end_date=new_prefs.end_date,
                        preferences=new_prefs,
                        days=days
                    )
                    new_trip.recalculate_totals(CURRENCY_RATES["INR"])
                    st.session_state.active_trip = new_trip
                    save_trip(new_trip)
                    st.success(f"Trip created for {rec.name}! Switch to Tab 2 to view your day-by-day plan.")
                    st.rerun()


# =============================================================
# TAB 2: DAY-BY-DAY CLUSTERED ITINERARY (EPIC 4)
# =============================================================
with tab2:
    active_trip: TripRecord = st.session_state.active_trip
    if not active_trip:
        st.info("No active trip found. Please use the sidebar form to generate a new itinerary.")
    else:
        # Header Metrics & Trip Overview
        st.markdown(f"### 📍 {active_trip.destination}: {active_trip.trip_name}")
        st.markdown(f"**Dates:** {active_trip.start_date} to {active_trip.end_date} ({active_trip.duration_days} Days) • **Party:** {active_trip.preferences.travel_party.split()[0]} • **Pace:** {active_trip.preferences.pace.split()[0]}")

        # Metrics row
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Total Estimated Budget", f"${active_trip.total_cost_usd:,.0f} USD", f"₹{active_trip.total_cost_inr:,.0f} INR")
        with m_col2:
            avg_daily = active_trip.total_cost_usd / max(1, active_trip.duration_days)
            st.metric("Avg Daily Spend", f"${avg_daily:,.0f} / day", f"₹{avg_daily * CURRENCY_RATES['INR']:,.0f} / day")
        with m_col3:
            total_stops = sum(len(d.slots) for d in active_trip.days)
            st.metric("Curated Stops", f"{total_stops} Activities", f"{active_trip.duration_days} Day Clusters")
        with m_col4:
            b_status = "On Track (Moderate)"
            if active_trip.preferences.budget_level == "Budget":
                b_status = "Eco Value (Budget)"
            elif active_trip.preferences.budget_level == "Luxury":
                b_status = "Premium Tier"
            st.metric("Budget Health", b_status, "Zero Backtracking")

        st.markdown("---")

        # Geographic Clustering & Interactive Map Section
        st.markdown("#### 🗺️ Geographic Zone Map & Transit Sequencing")
        st.markdown("Activities are clustered geographically by neighborhood to eliminate erratic city backtracking.")

        # Prepare Map Data
        map_points = []
        for d in active_trip.days:
            for s in d.slots:
                if s.latitude != 0.0 and s.longitude != 0.0:
                    map_points.append({
                        "day": f"Day {d.day_number}",
                        "slot": s.slot_title,
                        "title": s.title,
                        "lat": s.latitude,
                        "lon": s.longitude,
                        "zone": s.zone,
                        "cost": f"${s.estimated_cost_usd:.0f}"
                    })

        if map_points:
            df_map = pd.DataFrame(map_points)
            center_lat = df_map["lat"].mean()
            center_lon = df_map["lon"].mean()

            # Pydeck interactive scatter layer
            layer = pdk.Layer(
                "ScatterplotLayer",
                df_map,
                get_position=["lon", "lat"],
                get_color="[13, 148, 136, 200]",
                get_radius=220,
                pickable=True,
                auto_highlight=True
            )

            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=12.2,
                pitch=25
            )

            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"html": "<b>{day} - {slot}</b><br/>{title}<br/>Zone: {zone}<br/>Est. Cost: {cost}", "style": {"color": "white"}}
            )
            st.pydeck_chart(r, use_container_width=True)
        else:
            st.info("No geographic map coordinates available for this trip.")

        st.markdown("---")

        # Spending Breakdown Charts
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("##### 📊 Daily Spending Progression")
            daily_chart_data = pd.DataFrame([
                {"Day": f"Day {d.day_number}", "Cost (USD)": d.daily_cost_usd, "Cost (INR)": d.daily_cost_inr}
                for d in active_trip.days
            ])
            if daily_chart_data.empty:
                st.info("No daily spending data available.")
            else:
                daily_chart = alt.Chart(daily_chart_data).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#0D9488").encode(
                    x=alt.X("Day:N", title="Trip Day"),
                    y=alt.Y("Cost (USD):Q", title="Spend ($ USD)"),
                    tooltip=["Day", "Cost (USD)", "Cost (INR)"]
                ).properties(height=240)
                st.altair_chart(daily_chart, use_container_width=True)

        with col_c2:
            st.markdown("##### 🥧 Category Spend Breakdown")
            cat_totals = {}
            for d in active_trip.days:
                for s in d.slots:
                    cat = s.category or "Sightseeing"
                    cat_totals[cat] = cat_totals.get(cat, 0.0) + s.estimated_cost_usd
            cat_chart_data = pd.DataFrame([{"Category": k, "Spend ($ USD)": v} for k, v in cat_totals.items()])
            if cat_chart_data.empty or cat_chart_data["Spend ($ USD)"].sum() == 0:
                st.info("No categorical spend recorded for this trip.")
            else:
                cat_chart = alt.Chart(cat_chart_data).mark_arc(innerRadius=45).encode(
                    theta=alt.Theta(field="Spend ($ USD)", type="quantitative"),
                    color=alt.Color(field="Category", type="nominal", scale=alt.Scale(scheme="tealblues")),
                    tooltip=["Category", "Spend ($ USD)"]
                ).properties(height=240)
                st.altair_chart(cat_chart, use_container_width=True)

        st.markdown("---")

        # Day-by-Day Accordion Schedule
        st.markdown("#### 📆 Structured Day-Wise Itinerary")
        
        for day in active_trip.days:
            # Efficiency score
            eff = score_clustering_efficiency(day)
            
            with st.expander(f"📍 Day {day.day_number}: {day.theme} — {day.zone_cluster} (Est: ${day.daily_cost_usd:.0f} / ₹{day.daily_cost_inr:,.0f})", expanded=True):
                st.markdown(f"""
                <div style="display: flex; gap: 1rem; align-items: center; margin-bottom: 0.8rem;">
                    <span class="tag-badge blue">Cluster: {day.zone_cluster}</span>
                    <span class="tag-badge" style="background-color: {eff['badge_color']}20; color: {eff['badge_color']}; border-color: {eff['badge_color']};">
                        {eff['rating']}
                    </span>
                    <span style="font-size: 0.85rem; color: #64748B;">Total Leg Transit: ~{day.walking_distance_km} km</span>
                </div>
                """, unsafe_allow_html=True)

                for slot_idx, slot in enumerate(day.slots):
                    # Category container color
                    card_cls = "slot-container"
                    if slot.slot_id == "lunch":
                        card_cls += " dining"
                    elif slot.slot_id == "afternoon":
                        card_cls += " leisure"
                    elif slot.slot_id == "evening":
                        card_cls += " evening"

                    st.markdown(f"""
                    <div class="{card_cls}">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <div class="slot-time">{slot.time_window} • {slot.slot_title}</div>
                                <div class="slot-title">{slot.title}</div>
                            </div>
                            <div style="text-align: right;">
                                <span style="font-weight: 700; color: #0B2545;">${slot.estimated_cost_usd:.0f} USD</span><br/>
                                <span style="font-size: 0.8rem; color: #64748B;">₹{slot.estimated_cost_inr:.0f}</span>
                            </div>
                        </div>
                        <div class="slot-desc">{slot.description}</div>
                        <div class="slot-meta">
                            <span>📍 <b>Location:</b> {slot.location_name}</span>
                            <span>🏷️ <b>Category:</b> {slot.category}</span>
                            <span>💡 <b>Pro Tip:</b> {slot.tips}</span>
                        </div>
                        <div class="transit-chip">
                            <span>🚶‍♂️/🚇</span> {slot.transit_info}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# =============================================================
# TAB 3: VIBE CUSTOMIZATION & DYNAMIC ACTIVITY SWAP (EPIC 4)
# =============================================================
with tab3:
    st.markdown("### ✨ Vibe Customization & Dynamic Slot Swap")
    st.markdown("Easily customize your trip on the fly. Swap out any individual activity or regenerate single days without wiping out the rest of your itinerary.")

    active_trip: TripRecord = st.session_state.active_trip
    if not active_trip or not active_trip.days:
        st.info("No active trip to customize. Please generate an itinerary in the sidebar first.")
    else:
        col_s1, col_s2 = st.columns([1, 2])

        with col_s1:
            st.markdown("#### 1. Select Target Slot")
            day_options = [f"Day {d.day_number} ({d.zone_cluster})" for d in active_trip.days]
            selected_day_idx = st.selectbox("Select Day to Modify", range(len(day_options)), format_func=lambda i: day_options[i])
            chosen_day: DayItinerary = active_trip.days[selected_day_idx]

            slot_labels = [f"{s.slot_title} ({s.time_window})" for s in chosen_day.slots]
            selected_slot_idx = st.selectbox("Select Time Slot to Swap", range(len(slot_labels)), format_func=lambda i: slot_labels[i])
            chosen_slot: ActivitySlot = chosen_day.slots[selected_slot_idx]

            st.markdown("---")
            st.markdown("#### Current Slot Details")
            st.markdown(f"""
            <div style="background: #F1F5F9; padding: 1rem; border-radius: 8px; font-size: 0.88rem;">
                <b>Slot:</b> {chosen_slot.slot_title}<br/>
                <b>Activity:</b> {chosen_slot.title}<br/>
                <b>Current Zone:</b> {chosen_slot.zone}<br/>
                <b>Est. Cost:</b> ${chosen_slot.estimated_cost_usd:.0f} (₹{chosen_slot.estimated_cost_inr:.0f})<br/>
                <b>Category:</b> {chosen_slot.category}
            </div>
            """, unsafe_allow_html=True)

        with col_s2:
            st.markdown("#### 2. Define Your Swap Intent")
            
            # Preset Prompts with state retention
            if "swap_intent_text" not in st.session_state:
                st.session_state.swap_intent_text = "Swap for an authentic local artisanal market and coffee experience"

            st.markdown("**Quick Preset Prompts (Click to apply):**")
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            if p_col1.button("🥗 Vegetarian / Vegan"):
                st.session_state.swap_intent_text = "Find a top-rated organic vegetarian or vegan dining spot with local specialties"
            if p_col2.button("🌧️ Rainy Day Indoor"):
                st.session_state.swap_intent_text = "Sheltered indoor museum, art gallery, or artisan market safe from rain"
            if p_col3.button("💰 Budget / Free"):
                st.session_state.swap_intent_text = "Free scenic walking route, historic public park, or budget viewpoint"
            if p_col4.button("⚡ High Energy"):
                st.session_state.swap_intent_text = "High-energy adventure, bike tour, watersport, or adrenaline experience"

            swap_query = st.text_area(
                "Or Describe Custom Intent:",
                value=st.session_state.swap_intent_text,
                height=90,
                help="E.g., 'Swap museum for outdoor artisanal market', 'Find a rooftop cocktail bar', etc."
            )

            st.markdown("<br/>", unsafe_allow_html=True)
            col_b1, col_b2 = st.columns(2)
            
            with col_b1:
                if st.button("🔄 Swap This Single Activity", type="primary", use_container_width=True):
                    with st.spinner("Swapping slot while preserving geographic cluster..."):
                        try:
                            new_slot = LLMService.swap_single_slot(
                                target_slot=chosen_slot,
                                day_context=chosen_day,
                                swap_intent=swap_query,
                                preferences=active_trip.preferences,
                                provider=st.session_state.provider,
                                api_key=st.session_state.api_key
                            )
                            chosen_day.slots[selected_slot_idx] = new_slot
                            chosen_day = optimize_day_geographic_routing(chosen_day)
                            chosen_day.recalculate_totals(CURRENCY_RATES["INR"])
                            active_trip.recalculate_totals(CURRENCY_RATES["INR"])
                            save_trip(active_trip)
                            st.success(f"Slot '{chosen_slot.slot_title}' successfully replaced with: **{new_slot.title}**!")
                            st.rerun()
                        except Exception as swap_err:
                            st.error(f"Failed to swap slot: {swap_err}")

            with col_b2:
                if st.button("🔁 Regenerate Entire Day Plan", use_container_width=True):
                    with st.spinner(f"Regenerating Day {chosen_day.day_number} plan..."):
                        try:
                            try:
                                base_start = datetime.strptime(active_trip.start_date, "%Y-%m-%d")
                                day_date_str = (base_start + timedelta(days=selected_day_idx)).strftime("%Y-%m-%d")
                            except Exception:
                                day_date_str = str(date.today())

                            single_day_prefs = TripPreferences(
                                destination=active_trip.destination,
                                duration_days=1,
                                start_date=day_date_str,
                                end_date=day_date_str,
                                budget_level=active_trip.preferences.budget_level,
                                travel_party=active_trip.preferences.travel_party,
                                pace=active_trip.preferences.pace,
                                interests=active_trip.preferences.interests,
                                dietary=active_trip.preferences.dietary
                            )
                            fresh_days = LLMService.generate_itinerary(
                                single_day_prefs,
                                provider=st.session_state.provider,
                                api_key=st.session_state.api_key
                            )
                            if fresh_days:
                                fresh_day = fresh_days[0]
                                fresh_day.day_number = chosen_day.day_number
                                fresh_day.date_str = chosen_day.date_str
                                active_trip.days[selected_day_idx] = fresh_day
                                active_trip.recalculate_totals(CURRENCY_RATES["INR"])
                                save_trip(active_trip)
                                st.success(f"Day {chosen_day.day_number} regenerated with fresh cluster: **{fresh_day.zone_cluster}**!")
                                st.rerun()
                            else:
                                st.error("No itinerary generated for the selected day.")
                        except Exception as reg_err:
                            st.error(f"Failed to regenerate day: {reg_err}")


# =============================================================
# TAB 4: TRIP MANAGEMENT & OFFLINE PDF EXPORT (EPIC 5)
# =============================================================
with tab4:
    st.markdown("### 💾 Trip Management & Offline PDF Export")
    st.markdown("Manage multiple saved itineraries with zero-cloud local persistence, and generate publication-ready offline PDF travel guides.")

    col_m1, col_m2 = st.columns([1, 1])

    # Left Column: PDF Guide Export
    with col_m1:
        st.markdown("#### 📄 1-Click Offline PDF Travel Guide")
        active_trip: TripRecord = st.session_state.active_trip
        if active_trip:
            st.markdown(f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;">
                <h4 style="margin: 0 0 0.4rem 0; color: #0B2545;">✈️ {active_trip.trip_name}</h4>
                <div style="font-size: 0.88rem; color: #475569; margin-bottom: 0.6rem;">
                    <b>Destination:</b> {active_trip.destination} • <b>Days:</b> {active_trip.duration_days} Days<br/>
                    <b>Total Cost:</b> ${active_trip.total_cost_usd:,.0f} USD (₹{active_trip.total_cost_inr:,.0f} INR)
                </div>
                <div style="font-size: 0.82rem; color: #64748B;">
                    <b>Included in PDF:</b> Day-by-Day Clustered Schedule, Inter-slot Transit Guidance, Destination Emergency Numbers (Police, Medical, Embassy), and Essential Travel Packing Checklist.
                </div>
            </div>
            """, unsafe_allow_html=True)

            try:
                pdf_bytes = generate_itinerary_pdf(active_trip)
                clean_dest = "".join([c if c.isalnum() else "_" for c in active_trip.destination.lower()]).strip("_")
                clean_filename = f"{clean_dest}_travel_guide_{active_trip.id}.pdf"
                st.download_button(
                    label="📥 Download Offline PDF Travel Guide",
                    data=pdf_bytes,
                    file_name=clean_filename,
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating PDF: {e}")
        else:
            st.info("No active trip to export.")

    # Right Column: Trip History & Persistence
    with col_m2:
        st.markdown("#### 🗄️ Save Current Trip & JSON Backup")
        if active_trip:
            trip_custom_name = st.text_input("Trip Nickname", value=active_trip.trip_name)
            col_save1, col_save2 = st.columns(2)
            with col_save1:
                if st.button("💾 Save / Update Trip", use_container_width=True):
                    active_trip.trip_name = trip_custom_name
                    save_trip(active_trip)
                    st.success("Trip saved successfully in database!")
                    st.rerun()
            with col_save2:
                json_data = export_trip_json(active_trip)
                st.download_button(
                    "📥 Export Trip JSON",
                    data=json_data,
                    file_name=f"{active_trip.destination}_trip_{active_trip.id}.json",
                    mime="application/json",
                    use_container_width=True
                )

        st.markdown("---")
        st.markdown("#### 📂 Import Trip from JSON")
        uploaded_file = st.file_uploader("Upload previously exported Trip JSON", type=["json"])
        if uploaded_file is not None:
            try:
                content = uploaded_file.read().decode("utf-8")
                imported_trip = import_trip_json(content)
                st.session_state.active_trip = imported_trip
                st.session_state.trip_preferences = imported_trip.preferences
                st.success(f"Trip '{imported_trip.trip_name}' restored successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to import JSON: {e}")

    st.markdown("---")
    st.markdown("#### 📚 Saved Trips Repository (SQLite Local Database)")
    saved_trips = get_all_trips()
    if not saved_trips:
        st.caption("No trips saved yet in database.")
    else:
        for t in saved_trips:
            t_col1, t_col2, t_col3, t_col4 = st.columns([3, 2, 1, 1])
            with t_col1:
                st.markdown(f"**{t.trip_name}** ({t.destination})")
                st.caption(f"{t.start_date} to {t.end_date} • {t.duration_days} Days • Created: {t.created_at}")
            with t_col2:
                st.markdown(f"**${t.total_cost_usd:,.0f} USD** (₹{t.total_cost_inr:,.0f})")
                st.caption(f"Tier: {t.preferences.budget_level} • Party: {t.preferences.travel_party.split()[0]}")
            with t_col3:
                if st.button("📂 Load", key=f"load_{t.id}", use_container_width=True):
                    st.session_state.active_trip = t
                    st.session_state.trip_preferences = t.preferences
                    st.success(f"Loaded '{t.trip_name}'!")
                    st.rerun()
            with t_col4:
                if st.button("🗑️ Del", key=f"del_{t.id}", use_container_width=True):
                    delete_trip(t.id)
                    st.warning("Trip deleted.")
                    st.rerun()
