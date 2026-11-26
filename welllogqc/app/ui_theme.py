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
    "get_petroleum_loader_html",
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
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    width: 100% !important;
}

/* ── 1. COMPLETELY REMOVE SIDEBAR ── */
[data-testid="stSidebar"],
section[data-testid="stSidebar"],
[data-testid="collapsedControl"],
.stSidebar {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    pointer-events: none !important;
}

/* ── 2. REMOVE RUNNING MAN / SPORTS ANIMATION & STOP BUTTON; KEEP 3-DOTS ── */
[data-testid="stStatusWidget"],
div[data-testid="stStatusWidget"],
.stStatusWidget,
button[kind="header-action"],
button[data-testid="baseButton-headerAction"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

[data-testid="stMainMenu"],
#MainMenu {
    display: block !important;
    visibility: visible !important;
}

/* ── 3. ENTERPRISE HEADER ── */
.corporate-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px 22px;
    margin-bottom: 16px;
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
    min-width: 300px;
}

.corporate-header-badge {
    flex: 0 0 auto;
    margin-left: auto;
}

.corporate-header h1 {
    font-size: 1.55rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0 0 4px 0;
    letter-spacing: -0.01em;
    line-height: 1.3;
}

.corporate-header p {
    color: #94a3b8;
    margin: 0;
    font-size: 0.88rem;
    line-height: 1.4;
}

/* ── 4. BUTTONS & UI CONTROLS ── */
.stButton > button {
    white-space: nowrap !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 8px 20px !important;
    min-height: 42px !important;
}

/* ── 5. KPI CARDS ── */
.kpi-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
    min-width: 0;
    box-sizing: border-box;
    width: 100%;
}

.kpi-label {
    font-size: 0.76rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.3;
}

.kpi-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.2;
    white-space: nowrap;
}

.kpi-sub {
    font-size: 0.80rem;
    color: #64748b;
    margin-top: 4px;
    line-height: 1.3;
}

/* ── 6. ADVISORY & RECOMMENDATION CARDS ── */
.advisory-box {
    background: #0f172a;
    border-left: 4px solid #0284c7;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    border-top: 1px solid #334155;
    border-right: 1px solid #334155;
    border-bottom: 1px solid #334155;
}

.advisory-title {
    font-size: 0.90rem;
    font-weight: 600;
    color: #38bdf8;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.advisory-body {
    font-size: 0.85rem;
    color: #cbd5e1;
    line-height: 1.5;
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

/* ── 7. LIGHT & DARK THEME ADAPTABILITY (ALL LINES VISIBLE IN BOTH THEMES) ── */
@media (prefers-color-scheme: light) {
    .corporate-header {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
        border: 1px solid #cbd5e1 !important;
    }
    .corporate-header h1 {
        color: #0f172a !important;
    }
    .corporate-header p {
        color: #475569 !important;
    }
    .kpi-card {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-label {
        color: #475569 !important;
    }
    .kpi-value {
        color: #0f172a !important;
    }
    .kpi-sub {
        color: #64748b !important;
    }
    .advisory-box {
        background: #ffffff !important;
        border-top: 1px solid #cbd5e1 !important;
        border-right: 1px solid #cbd5e1 !important;
        border-bottom: 1px solid #cbd5e1 !important;
    }
    .advisory-title {
        color: #0369a1 !important;
    }
    .advisory-body {
        color: #334155 !important;
    }
}

[data-theme="light"] .corporate-header {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
    border: 1px solid #cbd5e1 !important;
}
[data-theme="light"] .corporate-header h1 {
    color: #0f172a !important;
}
[data-theme="light"] .corporate-header p {
    color: #475569 !important;
}
[data-theme="light"] .kpi-card {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
[data-theme="light"] .kpi-label {
    color: #475569 !important;
}
[data-theme="light"] .kpi-value {
    color: #0f172a !important;
}
[data-theme="light"] .kpi-sub {
    color: #64748b !important;
}
[data-theme="light"] .advisory-box {
    background: #ffffff !important;
    border-top: 1px solid #cbd5e1 !important;
    border-right: 1px solid #cbd5e1 !important;
    border-bottom: 1px solid #cbd5e1 !important;
}
[data-theme="light"] .advisory-title {
    color: #0369a1 !important;
}
[data-theme="light"] .advisory-body {
    color: #334155 !important;
}

/* ── 8. FORMATION EVALUATION PETROLEUM LOADER ANIMATION ── */
@keyframes wireline-scan {
    0% { transform: translateY(0px); }
    50% { transform: translateY(40px); }
    100% { transform: translateY(0px); }
}

@keyframes sensor-beam {
    0% { transform: scaleX(0.7); opacity: 0.9; }
    50% { transform: scaleX(1.3); opacity: 0.4; }
    100% { transform: scaleX(0.7); opacity: 0.9; }
}

@keyframes eval-pulse {
    0%, 100% { opacity: 0.85; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.02); }
}

.petro-loader-card {
    background: #1e293b;
    border: 1px solid #38bdf8;
    border-radius: 8px;
    padding: 24px;
    margin: 18px 0;
    text-align: center;
    box-shadow: 0 4px 20px rgba(56, 189, 248, 0.12);
}

.petro-loader-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #38bdf8;
    letter-spacing: 0.10em;
    margin-top: 12px;
    margin-bottom: 4px;
    animation: eval-pulse 1.8s infinite ease-in-out;
}

.petro-loader-sub {
    font-size: 0.88rem;
    color: #94a3b8;
    margin-bottom: 8px;
}
</style>
"""

def clean_html(html: str) -> str:
    """Strip leading indentation and empty lines so CommonMark never parses HTML as a code block."""
    return "\n".join(line.lstrip() for line in html.splitlines() if line.strip())

def inject_custom_theme(analysis_done: bool = False):
    """Inject corporate CSS stylesheet into Streamlit app with conditional print menu control."""
    css = CUSTOM_CSS
    if not analysis_done:
        css += """
        <style>
        /* Hide Print option from 3-dot menu before formation analysis is completed */
        div[data-testid="stMainMenuPopover"] li:nth-child(3),
        div[role="menu"] [role="menuitem"]:has(span:contains("Print")),
        button[role="menuitem"]:has(div:contains("Print")) {
            display: none !important;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

def get_petroleum_loader_html() -> str:
    """Generate high-contrast petroleum formation evaluation loading animation."""
    return clean_html("""
    <div class="petro-loader-card">
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 8px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="280" height="150" viewBox="0 0 280 150" style="overflow: visible;">
                <!-- Geological Strata Background -->
                <rect x="20" y="20" width="100" height="28" fill="#334155" rx="2" opacity="0.8"/>
                <rect x="160" y="20" width="100" height="28" fill="#334155" rx="2" opacity="0.8"/>
                <text x="25" y="38" fill="#94a3b8" font-size="9" font-family="sans-serif">Shale Caprock</text>

                <rect x="20" y="52" width="100" height="28" fill="#475569" rx="2" opacity="0.8"/>
                <rect x="160" y="52" width="100" height="28" fill="#475569" rx="2" opacity="0.8"/>
                <text x="25" y="70" fill="#cbd5e1" font-size="9" font-family="sans-serif">Limestone Matrix</text>

                <rect x="20" y="84" width="100" height="28" fill="#b45309" rx="2" opacity="0.85"/>
                <rect x="160" y="84" width="100" height="28" fill="#b45309" rx="2" opacity="0.85"/>
                <text x="25" y="102" fill="#fde68a" font-size="9" font-family="sans-serif">Porous Sandstone</text>

                <rect x="20" y="116" width="100" height="28" fill="#047857" rx="2" opacity="0.9"/>
                <rect x="160" y="116" width="100" height="28" fill="#047857" rx="2" opacity="0.9"/>
                <text x="25" y="134" fill="#a7f3d0" font-weight="700" font-size="9" font-family="sans-serif">Hydrocarbon Pay Zone</text>

                <!-- Borehole Wellbore Walls -->
                <line x1="124" y1="10" x2="124" y2="150" stroke="#38bdf8" stroke-width="2" stroke-dasharray="3,3"/>
                <line x1="156" y1="10" x2="156" y2="150" stroke="#38bdf8" stroke-width="2" stroke-dasharray="3,3"/>

                <!-- Wireline Cable -->
                <line x1="140" y1="0" x2="140" y2="140" stroke="#94a3b8" stroke-width="1.5"/>

                <!-- Animated Wireline Tool Sonde -->
                <g style="animation: wireline-scan 2.4s infinite ease-in-out;">
                    <!-- Sonde Body -->
                    <rect x="133" y="35" width="14" height="42" rx="4" fill="#0284c7" stroke="#38bdf8" stroke-width="1.5"/>
                    <circle cx="140" cy="45" r="2.5" fill="#38bdf8"/>
                    <circle cx="140" cy="56" r="2.5" fill="#10b981"/>
                    <circle cx="140" cy="67" r="2.5" fill="#f59e0b"/>

                    <!-- Acoustic / Resistivity Sensor Beams radiating into formation -->
                    <ellipse cx="140" cy="56" rx="36" ry="12" fill="none" stroke="#38bdf8" stroke-width="1.5" opacity="0.7" style="animation: sensor-beam 1.4s infinite ease-in-out; transform-origin: 140px 56px;"/>
                    <ellipse cx="140" cy="56" rx="58" ry="18" fill="none" stroke="#10b981" stroke-width="1" opacity="0.5" style="animation: sensor-beam 1.4s infinite ease-in-out; animation-delay: 0.4s; transform-origin: 140px 56px;"/>
                </g>
            </svg>
        </div>
        <div class="petro-loader-title">EVALUATING</div>
        <div class="petro-loader-sub">Multi-Sensor Formation Evaluation & Wireline QC Screening</div>
        <div style="font-size: 0.78rem; color: #64748b; font-weight: 500;">
            Screening Borehole Washout &bull; Compensating Density Standoff &bull; Archie Saturation Synthesis
        </div>
    </div>
    """)

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
