def get_css(lang: str = "en") -> str:
    rtl_block = ""
    if lang == "ar":
        rtl_block = """
        .main .block-container { direction: rtl; text-align: right; }
        .stTextInput input { direction: rtl; text-align: right; }
        .stSelectbox, .stMultiSelect { direction: rtl; }
        .alt-table th { text-align: right !important; }
        .alt-table td { text-align: right !important; }
        .section-title { padding-left: 0 !important; padding-right: 12px !important; border-left: none !important; border-right: 3px solid #167C73 !important; text-align: right !important; }
        .finding-text { margin-left: 0 !important; margin-right: 15px !important; text-align: right !important; }
        .gap-row { direction: rtl; }
        .pred-main { direction: rtl; }
        .confusion-item { direction: rtl; }
        .sem-row { direction: rtl; }
        .metric-inline { direction: rtl; }
        .kpi-label { direction: rtl; }
        .decision-note { text-align: right !important; }
        """

    ar_font = ""
    if lang == "ar":
        ar_font = """
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;500;600;700&display=swap');
    html, body, .stApp {
        font-family: 'Cairo', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    """

    return f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    {ar_font}

    #MainMenu, footer, header {{visibility: hidden;}}
    .stDeployButton {{display: none;}}

    section[data-testid="stSidebar"] {{
        background: #123832;
        border-right: 1px solid rgba(18,56,50,0.15);
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {{
        font-size: 0.88rem;
        line-height: 1.6;
        color: rgba(243,241,235,0.65);
    }}
    section[data-testid="stSidebar"] .stMarkdown strong {{
        color: rgba(243,241,235,0.8);
    }}
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: #f3f1eb;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(243,241,235,0.1) !important;
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        color: rgba(243,241,235,0.8) !important;
    }}
    section[data-testid="stSidebar"] .stRadio label:hover {{
        color: #f3f1eb !important;
    }}
    section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {{
        color: rgba(243,241,235,0.8) !important;
    }}
    section[data-testid="stSidebar"] .stExpander {{
        border-color: rgba(243,241,235,0.1) !important;
        background: rgba(243,241,235,0.03) !important;
        border-radius: 6px !important;
    }}
    section[data-testid="stSidebar"] .stExpander summary span,
    section[data-testid="stSidebar"] .stExpander [data-testid="stMarkdownContainer"] p {{
        color: rgba(243,241,235,0.6) !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stExpanderToggleIcon"] {{
        color: rgba(243,241,235,0.4) !important;
    }}
    section[data-testid="stSidebar"] .stButton button {{
        background: rgba(243,241,235,0.08) !important;
        border: 1px solid rgba(243,241,235,0.15) !important;
        border-radius: 6px !important;
        color: rgba(243,241,235,0.75) !important;
    }}
    section[data-testid="stSidebar"] .stButton button:hover {{
        background: rgba(243,241,235,0.14) !important;
        color: #f3f1eb !important;
    }}
    section[data-testid="stSidebar"] .stButton button[kind="primary"],
    section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] {{
        background: #167C73 !important;
        border-color: #167C73 !important;
        color: #fff !important;
        font-weight: 600 !important;
    }}
    section[data-testid="stSidebar"] .stButton button[kind="primary"]:hover,
    section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"]:hover {{
        background: #1a9e94 !important;
        border-color: #1a9e94 !important;
    }}

    .main [data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 6px !important;
        overflow: hidden;
    }}

    .kpi-row {{
        display: flex;
        gap: 12px;
        margin: 4px 0 28px;
        flex-wrap: wrap;
    }}
    .kpi-card {{
        flex: 1;
        min-width: 130px;
        background: #f3f1eb;
        border: 1px solid rgba(18,56,50,0.08);
        border-radius: 6px;
        padding: 16px 14px;
        text-align: center;
    }}
    .kpi-value {{
        font-size: 1.35rem;
        font-weight: 700;
        color: #167C73;
        line-height: 1.2;
    }}
    .kpi-label {{
        font-size: 0.72rem;
        font-weight: 500;
        color: rgba(18,56,50,0.6);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 5px;
    }}

    .sarf-header {{
        text-align: center;
        padding: 20px 0 4px;
    }}
    .sarf-header h1 {{
        font-size: 2.2rem;
        font-weight: 700;
        color: #123832;
        margin: 0;
        letter-spacing: 0.12em;
    }}
    .sarf-header .subtitle {{
        font-size: 0.95rem;
        color: rgba(18,56,50,0.65);
        margin-top: 4px;
    }}
    .sarf-badge {{
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 500;
        color: rgba(18,56,50,0.45);
        background: rgba(18,56,50,0.05);
        border: 1px solid rgba(18,56,50,0.1);
        border-radius: 4px;
        padding: 3px 12px;
        margin-top: 8px;
    }}
    .hero-text {{
        text-align: center;
        font-size: 1rem;
        color: rgba(18,56,50,0.65);
        margin: 14px auto 20px;
        max-width: 560px;
        line-height: 1.6;
    }}

    .section-title {{
        font-size: 1rem;
        font-weight: 600;
        color: #123832;
        margin-bottom: 10px;
        padding-left: 12px;
        border-left: 3px solid #167C73;
    }}
    .finding-text {{
        font-size: 0.9rem;
        color: rgba(18,56,50,0.65);
        margin: -2px 0 14px 15px;
        line-height: 1.5;
    }}

    .stTextInput input {{
        background: #f3f1eb !important;
        border: 1px solid rgba(18,56,50,0.12) !important;
        border-radius: 6px !important;
        color: #123832 !important;
        padding: 16px 20px !important;
        font-size: 1.08rem !important;
        transition: border-color 0.2s ease;
    }}
    .stTextInput input:focus {{
        border-color: #167C73 !important;
        box-shadow: 0 0 0 1px rgba(22,124,115,0.15) !important;
    }}
    .stTextInput input::placeholder {{
        color: rgba(18,56,50,0.3) !important;
    }}

    .main .stButton button {{
        background: rgba(18,56,50,0.04) !important;
        border: 1px solid rgba(18,56,50,0.12) !important;
        border-radius: 6px !important;
        color: rgba(18,56,50,0.65) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 6px 14px !important;
        transition: all 0.15s ease;
    }}
    .main .stButton button:hover {{
        background: rgba(18,56,50,0.08) !important;
        color: #123832 !important;
    }}
    .main .stButton button[kind="primary"],
    .main .stButton button[data-testid="stBaseButton-primary"] {{
        background: #167C73 !important;
        border-color: #167C73 !important;
        color: #fff !important;
        font-weight: 600 !important;
    }}
    .main .stButton button[kind="primary"]:hover,
    .main .stButton button[data-testid="stBaseButton-primary"]:hover {{
        background: #123832 !important;
        border-color: #123832 !important;
    }}

    .pred-main {{
        background: rgba(22,124,115,0.06);
        border: 1px solid rgba(22,124,115,0.12);
        border-radius: 6px;
        padding: 28px 24px;
        text-align: center;
        margin-bottom: 20px;
    }}
    .pred-main .understood-label {{
        font-size: 0.88rem;
        color: rgba(18,56,50,0.65);
    }}
    .pred-main .intent-name {{
        font-size: 1.25rem;
        font-weight: 600;
        color: #123832;
        margin: 10px 0 8px;
        line-height: 1.4;
    }}
    .pred-main .score-value {{
        font-size: 1.9rem;
        font-weight: 700;
        color: #167C73;
    }}
    .pred-main .score-suffix {{
        font-size: 0.82rem;
        color: rgba(18,56,50,0.55);
        margin-top: 2px;
    }}

    .decision-note {{
        font-size: 0.85rem;
        color: rgba(18,56,50,0.6);
        margin: 8px 0 4px;
        line-height: 1.5;
    }}

    .gap-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 9px 14px;
        border-radius: 6px;
        background: #f3f1eb;
        margin-bottom: 5px;
        font-size: 0.88rem;
    }}
    .gap-row .g-label {{
        color: rgba(18,56,50,0.7);
    }}
    .gap-row .g-value {{
        font-weight: 600;
        color: #123832;
    }}
    .gap-row .g-value.accent {{
        color: #167C73;
    }}

    .alt-table {{
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 8px;
    }}
    .alt-table th {{
        font-size: 0.78rem;
        font-weight: 600;
        color: rgba(18,56,50,0.55);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 8px 14px;
        border-bottom: 1px solid rgba(18,56,50,0.1);
        text-align: left;
    }}
    .alt-table td {{
        font-size: 0.88rem;
        color: rgba(18,56,50,0.75);
        padding: 10px 14px;
        border-bottom: 1px solid rgba(18,56,50,0.04);
    }}
    .alt-table tr:first-child td {{
        color: #167C73;
        font-weight: 600;
    }}
    .alt-table .score-cell {{
        font-weight: 600;
        font-variant-numeric: tabular-nums;
    }}

    .confusion-item {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 9px 14px;
        border-radius: 6px;
        background: #f3f1eb;
        margin-bottom: 5px;
    }}
    .confusion-item .c-name {{
        font-size: 0.88rem;
        color: #123832;
    }}
    .confusion-item .c-count {{
        font-size: 0.8rem;
        color: rgba(18,56,50,0.55);
    }}

    .sem-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 9px 14px;
        border-radius: 6px;
        background: #f3f1eb;
        margin-bottom: 5px;
    }}
    .sem-name {{
        font-size: 0.88rem;
        color: #123832;
    }}
    .sem-score {{
        font-size: 0.82rem;
        font-weight: 600;
        color: #167C73;
    }}

    .metric-inline {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 9px 14px;
        border-radius: 6px;
        background: #f3f1eb;
        margin-bottom: 5px;
    }}
    .metric-inline .m-label {{
        font-size: 0.88rem;
        color: rgba(18,56,50,0.65);
    }}
    .metric-inline .m-value {{
        font-size: 0.92rem;
        font-weight: 600;
        color: #167C73;
    }}

    .main .stExpander {{
        border: 1px solid rgba(18,56,50,0.08) !important;
        border-radius: 6px !important;
    }}

    .sarf-footer {{
        text-align: center;
        font-size: 0.7rem;
        color: rgba(243,241,235,0.35);
        padding: 8px 0 16px;
    }}

    .sidebar-logo {{
        text-align: center;
        padding: 14px 0 10px;
        border-bottom: 1px solid rgba(243,241,235,0.1);
        margin-bottom: 10px;
    }}
    .sidebar-logo h2 {{
        font-size: 1.3rem;
        font-weight: 700;
        color: #f3f1eb;
        margin: 0;
        letter-spacing: 0.14em;
    }}
    .sidebar-logo .sub {{
        font-size: 0.68rem;
        color: rgba(243,241,235,0.4);
        margin-top: 2px;
    }}

    .stPlotlyChart {{
        border-radius: 6px;
        overflow: hidden;
    }}

    hr {{
        border: none;
        border-top: 1px solid rgba(18,56,50,0.06);
        margin: 18px 0;
    }}

    ::-webkit-scrollbar {{ width: 5px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{
        background: rgba(18,56,50,0.12);
        border-radius: 3px;
    }}

    .results-header {{
        text-align: center;
        padding: 20px 0 8px;
    }}
    .results-header h1 {{
        font-size: 1.6rem;
        font-weight: 600;
        color: #123832;
        margin: 0;
    }}
    .results-header .intro {{
        font-size: 0.92rem;
        color: rgba(18,56,50,0.65);
        margin-top: 6px;
        max-width: 520px;
        margin-left: auto;
        margin-right: auto;
    }}

    .latency-tag {{
        display: inline-block;
        font-size: 0.78rem;
        color: rgba(18,56,50,0.55);
        background: rgba(18,56,50,0.06);
        border: 1px solid rgba(18,56,50,0.1);
        border-radius: 4px;
        padding: 3px 10px;
        margin-top: 8px;
    }}

    .info-tip {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 18px;
        height: 18px;
        font-size: 0.65rem;
        font-weight: 600;
        font-style: normal;
        color: rgba(18,56,50,0.35);
        background: rgba(18,56,50,0.06);
        border: 1px solid rgba(18,56,50,0.1);
        border-radius: 50%;
        margin-left: 8px;
        margin-right: 8px;
        cursor: help;
        position: relative;
        vertical-align: middle;
        line-height: 1;
    }}
    .info-tip:hover {{
        color: #167C73;
        background: rgba(22,124,115,0.08);
        border-color: rgba(22,124,115,0.25);
    }}
    .info-tip:hover::after {{
        content: attr(title);
        position: absolute;
        bottom: calc(100% + 8px);
        left: 50%;
        transform: translateX(-50%);
        background: #123832;
        color: #f3f1eb;
        font-size: 0.78rem;
        font-weight: 400;
        line-height: 1.5;
        padding: 10px 14px;
        border-radius: 6px;
        width: max-content;
        max-width: 320px;
        white-space: normal;
        z-index: 9999;
        box-shadow: 0 4px 16px rgba(18,56,50,0.2);
        pointer-events: none;
    }}
    .info-tip:hover::before {{
        content: "";
        position: absolute;
        bottom: calc(100% + 2px);
        left: 50%;
        transform: translateX(-50%);
        border: 6px solid transparent;
        border-top-color: #123832;
        z-index: 9999;
        pointer-events: none;
    }}

    {rtl_block}
</style>"""
