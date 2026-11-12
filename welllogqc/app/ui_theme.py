"""
welllogqc/app/ui_theme.py
=========================
Professional enterprise styling, color palette, and SVG icon system.
Strictly zero emojis. All visual cues use clean SVG icons and executive badges.
"""

import streamlit as st

__all__ = [
    "clean_html",
    "inject_custom_theme",
    "render_page_header",
    "render_kpi",
    "render_section_header",
    "render_sidebar_header",
    "render_advisory_card",
    "ICONS",
    "PALETTE",
    "CUSTOM_CSS",
]

# Enterprise Color Palette
PALETTE = {
    "primary": "#0284c7",       # Sky 600
    "primary_dark": "#0369a1",  # Sky 700
    "secondary": "#334155",    # Slate 700
    "bg_dark": "#0f172a",      # Slate 900
    "card_bg": "#1e293b",      # Slate 800
    "card_border": "#334155",  # Slate 700
    "success": "#10b981",      # Emerald 500
    "warning": "#f59e0b",      # Amber 500
    "danger": "#ef4444",       # Red 500
    "info": "#38bdf8",         # Sky 400
    "text": "#f8fafc",         # Slate 50
    "text_muted": "#94a3b8",   # Slate 400
}

# Professional SVG Icons (Lucide / Feather / Material style)
ICONS = {
    "server": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px;"><rect width="20" height="8" x="2" y="2" rx="2" ry="2"/><rect width="20" height="8" x="2" y="14" rx="2" ry="2"/><line x1="6" x2="6.01" y1="6" y2="6"/><line x1="6" x2="6.01" y1="18" y2="18"/></svg>""",
    "cpu": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px;"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/></svg>""",
    "gpu": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px;"><path d="M2 8h20v8H2z"/><path d="M6 16v2"/><path d="M10 16v2"/><path d="M14 16v2"/><path d="M18 16v2"/><path d="M6 6v2"/><path d="M10 6v2"/><path d="M14 6v2"/><path d="M18 6v2"/></svg>""",
    "activity": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px;"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>""",
    "check": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px;"><path d="M20 6 9 17l-5-5"/></svg>""",
    "alert": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px;"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/></svg>""",
    "layers": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px;"><path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/></svg>""",
    "database": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px;"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/></svg>""",
    "chart": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px;"><line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/></svg>""",
    "download": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>""",
    "shield": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>""",
    "target": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px;"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>""",
    "globe": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px;"><circle cx="12" cy="12" r="10"/><line x1="2" x2="22" y1="12" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>""",
    "compass": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px;"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>"""
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Force wide layout across all views */
.block-container {
    max-width: 96% !important;
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    width: 100% !important;
}

/* Enterprise Header */
.corporate-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 18px 24px;
    margin-bottom: 20px;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    width: 100%;
    box-sizing: border-box;
}

.corporate-header-text {
    flex: 1 1 auto;
    min-width: 320px;
}

.corporate-header-badge {
    flex: 0 0 auto;
    margin-left: auto;
}

.corporate-header h1 {
    font-size: 1.65rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0 0 6px 0;
    letter-spacing: -0.01em;
    line-height: 1.35;
    white-space: normal;
    word-break: normal;
}

.corporate-header p {
    color: #94a3b8;
    margin: 0;
    font-size: 0.90rem;
    line-height: 1.45;
    white-space: normal;
}

/* Buttons: Prevent any vertical word wrapping */
.stButton > button {
    white-space: nowrap !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 10px 24px !important;
    min-height: 44px !important;
}

/* Sidebar navigation links: Prevent vertical word wrapping */
[data-testid="stSidebarNav"] span {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.74rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 6px 14px;
    border-radius: 9999px;
    white-space: nowrap;
}

.status-online {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.status-offline {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.4);
}

.status-gpu {
    background: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.4);
}

.kpi-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
    min-width: 0;
    box-sizing: border-box;
    width: 100%;
}

.kpi-label {
    font-size: 0.78rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.3;
}

.kpi-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.2;
    white-space: nowrap;
}

.kpi-sub {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 4px;
    white-space: normal;
    word-break: normal;
    line-height: 1.35;
}

.advisory-box {
    background: #0f172a;
    border-left: 4px solid #0284c7;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 14px 0;
}

.advisory-title {
    font-size: 0.92rem;
    font-weight: 600;
    color: #38bdf8;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.advisory-body {
    font-size: 0.86rem;
    color: #cbd5e1;
    line-height: 1.5;
}

.decision-approved {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid #10b981;
    color: #10b981;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
}

.decision-rejected {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid #ef4444;
    color: #ef4444;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
}

.decision-relog {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid #f59e0b;
    color: #f59e0b;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
}

.advisory-drilling {
    border-left: 4px solid #f59e0b !important;
}
.advisory-wireline {
    border-left: 4px solid #8b5cf6 !important;
}
.advisory-petro {
    border-left: 4px solid #0284c7 !important;
}
</style>
"""

def clean_html(html: str) -> str:
    """Strip leading indentation and empty lines so CommonMark never parses HTML as a code block."""
    return "\n".join(line.lstrip() for line in html.splitlines() if line.strip())

def inject_custom_theme():
    """Inject corporate CSS stylesheet into Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def render_page_header(title: str, subtitle: str, badge_text: str = None, badge_type: str = "gpu"):
    """Render a clean, corporate header with an optional status pill."""
    inject_custom_theme()
    badge_html = ""
    if badge_text:
        icon = ICONS.get(badge_type, ICONS["gpu"])
        badge_html = f'<div class="corporate-header-badge"><div class="status-pill status-{badge_type}">{icon} {badge_text}</div></div>'
    html = clean_html(f"""
    <div class="corporate-header">
        <div class="corporate-header-text">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        {badge_html}
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)

def render_section_header(title: str, icon_name: str = None, subtitle: str = None):
    """Render a clean section header with an SVG icon without raw HTML leak."""
    icon_svg = ICONS.get(icon_name, "") if icon_name else ""
    sub_html = f"<div style='color: #94a3b8; font-size: 0.84rem; margin-top: 4px; white-space: normal; line-height: 1.4;'>{subtitle}</div>" if subtitle else ""
    html = clean_html(f"""
    <div style="margin-top: 22px; margin-bottom: 12px; border-bottom: 1px solid #334155; padding-bottom: 8px; width: 100%; box-sizing: border-box;">
        <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; display: flex; align-items: center; gap: 10px; width: 100%;">
            <span style="flex-shrink: 0; display: inline-flex;">{icon_svg}</span> <span style="white-space: normal; line-height: 1.3;">{title}</span>
        </div>
        {sub_html}
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)

def render_sidebar_header(title: str, icon_name: str = None):
    """Render a clean sidebar category header."""
    icon_svg = ICONS.get(icon_name, "") if icon_name else ""
    html = clean_html(f"""
    <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; display: flex; align-items: center; gap: 8px; margin-top: 14px; margin-bottom: 10px; border-bottom: 1px solid #334155; padding-bottom: 6px;">
        {icon_svg} <span>{title}</span>
    </div>
    """)
    st.sidebar.markdown(html, unsafe_allow_html=True)

def render_kpi(label: str, value: str, subtext: str = "", delta: str = None):
    """Render an executive KPI card."""
    delta_html = f"<span style='color: #10b981; font-weight: 600;'>{delta}</span>" if delta else ""
    html = clean_html(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{subtext} {delta_html}</div>
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)

def render_advisory_card(title: str, recommendation: str, icon_name: str = "shield", category: str = "petro"):
    """Render an engineering advisory or improvement recommendation."""
    icon = ICONS.get(icon_name, ICONS["shield"])
    cat_class = f"advisory-{category.lower()}"
    html = clean_html(f"""
    <div class="advisory-box {cat_class}">
        <div class="advisory-title">{icon} {title}</div>
        <div class="advisory-body">{recommendation}</div>
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)
