"""
Offline PDF Travel Itinerary Guide Generator using ReportLab.
Produces a clean, beautifully formatted travel guide with day schedules,
emergency contact checklists, and packing guidelines.
"""

import io
import html
from typing import Optional
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
)
from models.schemas import TripRecord
from config.settings import CURATED_DESTINATIONS, CURRENCY_RATES


def _clean_pdf_text(text: Optional[str]) -> str:
    """Escapes XML entities so ReportLab Paragraph parsers never raise parse errors."""
    if not text:
        return ""
    # Strip emojis that could render as unprintable square boxes in standard Type 1 Helvetica
    cleaned = text.encode("ascii", "ignore").decode("ascii").strip()
    return html.escape(cleaned)


def generate_itinerary_pdf(trip: TripRecord) -> bytes:
    """
    Builds a multi-page, publication-grade PDF itinerary travel guide and returns bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Brand Palette
    c_primary = colors.HexColor("#0B2545")     # Deep oceanic navy
    c_secondary = colors.HexColor("#0D9488")   # Teal
    c_accent = colors.HexColor("#D97706")      # Amber
    c_light_bg = colors.HexColor("#F8FAFC")    # Cool light grey
    c_dark_text = colors.HexColor("#1E293B")   # Charcoal
    c_subtext = colors.HexColor("#64748B")     # Slate grey

    # Custom Typography Styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.white,
        alignment=1  # Center
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#CCFBF1"),
        alignment=1
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=c_primary,
        spaceBefore=12,
        spaceAfter=6
    )

    day_header_style = ParagraphStyle(
        "DayHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=c_secondary,
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=c_dark_text
    )

    bold_body_style = ParagraphStyle(
        "BoldBody_Custom",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    small_style = ParagraphStyle(
        "Small_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=c_subtext
    )

    story = []

    dest_title = _clean_pdf_text(trip.destination.upper())
    trip_title = _clean_pdf_text(trip.trip_name)
    start_d = _clean_pdf_text(trip.start_date)
    end_d = _clean_pdf_text(trip.end_date)
    party_str = _clean_pdf_text(trip.preferences.travel_party.split()[0] if trip.preferences.travel_party else "Solo")
    pace_str = _clean_pdf_text(trip.preferences.pace.split()[0] if trip.preferences.pace else "Moderate")
    diet_str = _clean_pdf_text(trip.preferences.dietary or "Any / Flexible")
    tier_str = _clean_pdf_text(trip.preferences.budget_level or "Moderate")

    # 1. HEADER BANNER
    header_data = [
        [Paragraph(f"{dest_title} TRAVEL GUIDE", title_style)],
        [Paragraph(f"{trip_title}  |  {start_d} to {end_d}  |  {trip.duration_days} Days", subtitle_style)]
    ]
    header_table = Table(header_data, colWidths=[540])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_primary),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('LEFTPADDING', (0, 0), (-1, -1), 16),
        ('RIGHTPADDING', (0, 0), (-1, -1), 16),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 14))

    # 2. TRIP OVERVIEW MATRIX
    interests_display = ", ".join([_clean_pdf_text(i) for i in trip.preferences.interests[:2]]) if trip.preferences.interests else "General Sights"
    overview_data = [
        [
            Paragraph("<b>Destination:</b> " + _clean_pdf_text(trip.destination), body_style),
            Paragraph(f"<b>Duration:</b> {trip.duration_days} Days", body_style),
            Paragraph(f"<b>Travel Party:</b> {party_str}", body_style),
        ],
        [
            Paragraph(f"<b>Pace:</b> {pace_str}", body_style),
            Paragraph(f"<b>Budget Tier:</b> {tier_str}", body_style),
            Paragraph(f"<b>Dietary:</b> {diet_str}", body_style),
        ],
        [
            Paragraph(f"<b>Total Est. Cost:</b> ${trip.total_cost_usd:,.0f} USD (INR {trip.total_cost_inr:,.0f})", bold_body_style),
            Paragraph(f"<b>Interests:</b> {interests_display}", body_style),
            Paragraph(f"<b>Status:</b> Confirmed & Optimized", body_style),
        ]
    ]
    overview_table = Table(overview_data, colWidths=[180, 180, 180])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_light_bg),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 14))

    # 3. DAY-BY-DAY ITINERARY SCHEDULE
    story.append(Paragraph("Day-by-Day Optimized Itinerary", h1_style))
    story.append(Paragraph("Activities are clustered geographically by neighborhood to eliminate transit backtracking and minimize fatigue.", small_style))
    story.append(Spacer(1, 8))

    if not trip.days:
        story.append(Paragraph("No daily activities currently scheduled for this trip.", body_style))
        story.append(Spacer(1, 10))
    else:
        for day in trip.days:
            day_elements = []
            day_theme = _clean_pdf_text(day.theme)
            day_zone = _clean_pdf_text(day.zone_cluster)
            day_title = f"Day {day.day_number} -- {day_theme} ({day_zone})"
            day_elements.append(Paragraph(day_title, day_header_style))

            table_rows = [
                [
                    Paragraph("<b>Time & Slot</b>", bold_body_style),
                    Paragraph("<b>Activity & Experience</b>", bold_body_style),
                    Paragraph("<b>Transit / Distance</b>", bold_body_style),
                    Paragraph("<b>Est. Cost</b>", bold_body_style)
                ]
            ]

            for s in day.slots:
                time_win = _clean_pdf_text(s.time_window)
                slot_t = _clean_pdf_text(s.slot_title)
                s_title = _clean_pdf_text(s.title)
                s_desc = _clean_pdf_text(s.description)
                s_tip = _clean_pdf_text(s.tips)
                s_transit = _clean_pdf_text(s.transit_info)
                s_loc = _clean_pdf_text(s.location_name)

                slot_label = f"<b>{time_win}</b><br/><font color='#0D9488'>{slot_t}</font>"
                activity_desc = f"<b>{s_title}</b><br/>{s_desc}<br/><font color='#64748B'><i>Tip: {s_tip}</i></font>"
                transit = f"{s_transit}<br/><font color='#94A3B8'>{s_loc}</font>"
                cost = f"${s.estimated_cost_usd:.0f} USD<br/>INR {s.estimated_cost_inr:,.0f}"

                table_rows.append([
                    Paragraph(slot_label, body_style),
                    Paragraph(activity_desc, body_style),
                    Paragraph(transit, small_style),
                    Paragraph(cost, body_style)
                ])

            day_table = Table(table_rows, colWidths=[95, 235, 130, 80])
            day_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ('LINEBELOW', (0, 0), (-1, 0), 1, c_secondary),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ]))
            day_elements.append(day_table)

            summary_text = f"<b>Day {day.day_number} Summary:</b> Total Transit: ~{day.walking_distance_km} km | Daily Est. Spend: ${day.daily_cost_usd:.0f} USD (INR {day.daily_cost_inr:,.0f})"
            day_elements.append(Spacer(1, 4))
            day_elements.append(Paragraph(summary_text, small_style))
            day_elements.append(Spacer(1, 10))

            story.append(KeepTogether(day_elements))

    # 4. EMERGENCY CONTACTS & HELPLINES
    story.append(Spacer(1, 10))
    story.append(Paragraph("Emergency Contacts & Safety Checklist", h1_style))

    # Retrieve destination specific numbers if available
    dest_match = next((d for d in CURATED_DESTINATIONS if d["name"].lower() in trip.destination.lower()), None)
    police_no = dest_match["emergency_contacts"]["police"] if dest_match else "112 / 911"
    med_no = dest_match["emergency_contacts"]["ambulance"] if dest_match else "112 / 911"
    tourist_no = dest_match["emergency_contacts"].get("tourist", "National Helpline") if dest_match else "National Tourist Assistance"

    emergency_data = [
        [
            Paragraph("<b>Emergency Service</b>", bold_body_style),
            Paragraph("<b>Contact Dial / Action</b>", bold_body_style),
            Paragraph("<b>Operational Guidance</b>", bold_body_style)
        ],
        [
            Paragraph("Police / Law Enforcement", body_style),
            Paragraph(f"<b>{police_no}</b>", bold_body_style),
            Paragraph("Report thefts, lost passport certificate, or immediate danger.", body_style)
        ],
        [
            Paragraph("Ambulance & Medical Emergency", body_style),
            Paragraph(f"<b>{med_no}</b>", bold_body_style),
            Paragraph("Keep hotel address written in local language.", body_style)
        ],
        [
            Paragraph("Tourist Police / Helpline", body_style),
            Paragraph(f"<b>{tourist_no}</b>", bold_body_style),
            Paragraph("English-speaking operators for traveler assistance and scam disputes.", body_style)
        ],
        [
            Paragraph("Universal Embassy Protocol", body_style),
            Paragraph("Direct Embassy Hotline", body_style),
            Paragraph("Notify embassy immediately in case of passport loss or legal emergencies.", body_style)
        ]
    ]

    emergency_table = Table(emergency_data, colWidths=[140, 120, 280])
    emergency_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#FEE2E2")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#991B1B")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#FCA5A5")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#FEE2E2")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(emergency_table)

    # 5. ESSENTIAL PACKING & DEPARTURE CHECKLIST
    story.append(Spacer(1, 10))
    story.append(Paragraph("Essential Travel Preparation Checklist", h1_style))
    checklist_text = (
        "- <b>Documents:</b> Passport valid for at least 6 months, printed copies of travel visa, travel insurance policy card.<br/>"
        "- <b>Finances:</b> Inform your bank of travel dates; carry at least 2 zero-forex debit/credit cards and a small amount of local currency cash.<br/>"
        "- <b>Connectivity:</b> Download offline Google Maps for the city cluster; download Google Translate offline language pack.<br/>"
        "- <b>Electronics:</b> Universal electrical travel adapter, 10,000mAh+ power bank for day-long sightseeing.<br/>"
        "- <b>Health:</b> Basic first-aid kit with personal prescriptions, hydration electrolytes, blister bandages, and antacids."
    )
    story.append(Paragraph(checklist_text, body_style))
    story.append(Spacer(1, 14))

    # FOOTER
    story.append(Paragraph("Generated by Wanderlust Personalized Travel Planner -- Skill Wallet Capstone Project 4", small_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
