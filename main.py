"""
Assembly Line — entry point.

Each tool is a page under tools/. The folder is deliberately not called `pages/`:
that name triggers Streamlit's automatic page discovery, which would fight with
the explicit st.navigation below and render the nav twice.

Add a tool by dropping a script in tools/ and appending an st.Page here.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Assembly Line", page_icon="🛎️", layout="wide")

# ── Visual identity ──────────────────────────────────────────────────────────
# Restrained functionalism: an off-white ground, black hairlines, square
# corners, and De Stijl primaries used only where they carry meaning — the
# action, the step marker, an error. No colour is decorative; nothing is
# coloured that would read the same in black.
INK, RULE, GROUND, CARD = "#111111", "#111111", "#f7f7f4", "#ffffff"
BLUE, RED, YELLOW = "#1b4d9b", "#c4302b", "#e3b505"
_STEP_COLOURS = [BLUE, RED, YELLOW, BLUE, RED]

_step_css = "".join(
    f'.st-key-step{i+1} > div:first-child::before{{background:{c};}}'
    for i, c in enumerate(_STEP_COLOURS))

st.markdown(f"""<style>
.block-container {{max-width:1180px; padding-top:2.2rem;}}
.stApp {{background:{GROUND};}}

/* step blocks: white fields bounded by black lines, not soft grey cards */
[class*="st-key-step"] {{
  background:{CARD}; border:1px solid {RULE} !important; border-radius:0 !important;
  padding:1.15rem 1.35rem 1.35rem !important; margin-bottom:1.05rem;
}}
/* the one flourish: a small primary square marking each step */
[class*="st-key-step"] > div:first-child {{position:relative; padding-left:1.15rem;}}
[class*="st-key-step"] > div:first-child::before {{
  content:""; position:absolute; left:0; top:.62rem; width:.55rem; height:.55rem;
}}
{_step_css}
[class*="st-key-step"] h3 {{
  font-size:1.02rem !important; font-weight:600 !important; letter-spacing:.01em;
  color:{INK}; padding:0 0 .35rem 0 !important;
}}

/* square everything: Rams geometry, no pill shapes */
.stButton button, .stDownloadButton button, [data-baseweb="select"] > div,
[data-baseweb="input"], .stTextInput input, [data-testid="stExpander"] details {{
  border-radius:0 !important;
}}
[data-testid="stExpander"] details {{border:1px solid #d9d9d2 !important;}}

/* colour reserved for the action */
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {{
  background:{BLUE} !important; border:1px solid {BLUE} !important;
  color:#fff !important; font-weight:600;
}}
.stButton button[kind="primary"]:hover, .stDownloadButton button[kind="primary"]:hover {{
  background:{INK} !important; border-color:{INK} !important;
}}
.stAlert {{border-radius:0 !important;}}

/* tabs: a rule, underlined where you are */
.stTabs [data-baseweb="tab-list"] {{gap:1.4rem; border-bottom:1px solid #d9d9d2;}}
.stTabs [data-baseweb="tab-highlight"] {{background:{INK};}}

/* selection chips: ten saturated pills read as decoration, not data —
   neutral squares with a hairline keep the colour budget for meaning */
[data-baseweb="tag"] {{
  background:#efefe9 !important; border:1px solid #cfcfc6 !important;
  border-radius:0 !important; color:{INK} !important;
}}
[data-baseweb="tag"] span, [data-baseweb="tag"] svg {{color:{INK} !important; fill:{INK} !important;}}

/* sliders and toggles in ink, not the default accent */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
  background:{INK} !important; border-radius:0 !important;
}}
[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
[data-testid="stSlider"] [data-testid="stSliderTickBarMax"] {{color:#8a8a80;}}

/* top nav: squared, current tool marked in ink */
[data-testid="stTopNav"] a, header a[href] {{border-radius:0 !important;}}

h1 {{letter-spacing:-.02em; font-weight:700;}}
</style>""", unsafe_allow_html=True)

TOOLS = [
    st.Page("tools/edge_concierge.py", title="Edge Concierge", default=True),
    st.Page("tools/learned_edge.py", title="Learned Edge"),
]

# Nav sits along the top, not in the sidebar: switching tools and configuring
# the tool you are in are different jobs, and stacking them made the sidebar
# read as one list where "Learned Edge" looked like a setting of Edge Concierge.
st.navigation(TOOLS, position="top").run()
