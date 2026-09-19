import os
import time

import streamlit as st
import pandas as pd

from translations import TEXTS, translate_label
from styles import get_css
import charts

st.set_page_config(
    page_title="SARF | Arabic Banking Intent Classifier",
    page_icon="\U0001f3e6",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "lang" not in st.session_state:
    st.session_state.lang = "en"


def _t(key: str) -> str:
    return TEXTS[st.session_state.lang].get(key, key)


st.markdown(get_css(st.session_state.lang), unsafe_allow_html=True)

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

ARABIC_SAMPLES = [
    "بطاقتي ما تشتغل",
    "اتخصم مني مرتين",
    "نسيت الرقم السري",
    "حوّلتي ما وصلت",
]


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    path = os.path.join(ASSETS, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource
def load_predictor():
    from predictor import predict_unified, retrieve_similar_msa_examples
    return predict_unified, retrieve_similar_msa_examples


@st.cache_data
def measure_latency() -> int:
    predict_unified_fn, _ = load_predictor()
    predict_unified_fn("تجربة")
    t0 = time.time()
    predict_unified_fn("بطاقتي ما تشتغل")
    return round((time.time() - t0) * 1000)


def _set_sample(txt: str):
    st.session_state.main_input = txt


def _reset():
    st.session_state.main_input = ""


def _render_sidebar():
    st.markdown(
        '<div class="sidebar-logo">'
        "<h2>SARF</h2>"
        '<div class="sub">Arabic Banking NLP</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns(2)
    with left_col:
        if st.button(
            "EN",
            use_container_width=True,
            type="primary" if st.session_state.lang == "en" else "secondary",
        ):
            st.session_state.lang = "en"
            st.rerun()
    with right_col:
        if st.button(
            "عربي",
            use_container_width=True,
            type="primary" if st.session_state.lang == "ar" else "secondary",
        ):
            st.session_state.lang = "ar"
            st.rerun()

    st.markdown("---")

    page = st.radio(
        "nav",
        options=["try", "results"],
        format_func=lambda x: _t(f"nav_{x}"),
        label_visibility="collapsed",
    )

    st.markdown("---")

    with st.expander(_t("methodology_title"), expanded=False):
        for key in [
            "meth_dataset",
            "meth_model",
            "meth_checkpoint",
            "meth_conditions",
            "meth_transfer",
            "meth_retrieval",
        ]:
            st.markdown(_t(key))

    with st.expander(_t("limitations_title"), expanded=False):
        for key in [
            "lim_calibration",
            "lim_oos",
            "lim_routing",
            "lim_ensemble",
            "lim_privacy",
        ]:
            st.markdown(f"- {_t(key)}")

    with st.expander(_t("future_work_title"), expanded=False):
        for key in [
            "future_work_1",
            "future_work_2",
            "future_work_3",
            "future_work_4",
        ]:
            st.markdown(f"- {_t(key)}")

    st.markdown("---")
    st.markdown(
        f'<div class="sarf-footer">{_t("footer")}</div>',
        unsafe_allow_html=True,
    )
    return page


def _render_header():
    st.markdown(
        f'<div class="sarf-header">'
        f"<h1>{_t('app_title')}</h1>"
        f'<div class="subtitle">{_t("app_subtitle")}</div>'
        f'<div class="sarf-badge">{_t("badge")}</div>'
        f"</div>"
        f'<div class="hero-text">{_t("hero_text")}</div>',
        unsafe_allow_html=True,
    )


def _render_kpi_row(latency_str: str):
    st.markdown(
        f'<div class="kpi-row">'
        f'<div class="kpi-card">'
        f'<div class="kpi-value">77</div>'
        f'<div class="kpi-label">{_t("kpi_intents")}</div>'
        f"</div>"
        f'<div class="kpi-card">'
        f'<div class="kpi-value">0.731</div>'
        f'<div class="kpi-label">{_t("kpi_f1")}</div>'
        f"</div>"
        f'<div class="kpi-card">'
        f'<div class="kpi-value">E1 / 2026</div>'
        f'<div class="kpi-label">{_t("kpi_model")}</div>'
        f"</div>"
        f'<div class="kpi-card">'
        f'<div class="kpi-value">{latency_str}</div>'
        f'<div class="kpi-label">{_t("kpi_latency")}</div>'
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_input_area():
    _, center_col, _ = st.columns([1, 3, 1])

    with center_col:
        input_c, clear_c = st.columns([7, 1], vertical_alignment="bottom")
        with input_c:
            user_text = st.text_input(
                label="sentence",
                placeholder=_t("input_placeholder"),
                key="main_input",
                label_visibility="collapsed",
            )
        with clear_c:
            st.button(_t("reset_btn"), on_click=_reset, use_container_width=True)

        st.markdown(
            f'<p style="font-size:0.8rem;color:rgba(18,56,50,0.5);'
            f'margin:4px 0 6px;text-align:center;">{_t("try_sample")}</p>',
            unsafe_allow_html=True,
        )

        cols = st.columns(4)
        for i, col in enumerate(cols):
            with col:
                st.button(
                    _t(f"sample_{i + 1}"),
                    on_click=_set_sample,
                    args=(ARABIC_SAMPLES[i],),
                    use_container_width=True,
                )

    return user_text


def _section_header(title_key: str, tip_key: str):
    st.markdown(
        f'<div class="section-title">{_t(title_key)}'
        f'<span class="info-tip" title="{_t(tip_key)}">i</span></div>',
        unsafe_allow_html=True,
    )


def _render_prediction(top3, pred_latency):
    lang = st.session_state.lang
    pct_top = top3[0]["score"] * 100
    top1_display = translate_label(top3[0]["label"], lang)

    _section_header("sec_understood", "tip_understood")

    st.markdown(
        f'<div class="pred-main">'
        f'<div class="understood-label">{_t("model_understood")}</div>'
        f'<div class="intent-name">{top1_display}</div>'
        f'<div class="score-value">{pct_top:.1f}%</div>'
        f'<div class="score-suffix">{_t("model_score")}</div>'
        f'<div class="latency-tag">{_t("response_time")}: {pred_latency}ms</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_decision(top3):
    _section_header("sec_decision", "tip_decision")

    if len(top3) < 2:
        return

    gap = top3[0]["score"] - top3[1]["score"]
    gap_pct = gap * 100

    st.markdown(
        f'<div class="gap-row">'
        f'<span class="g-label">{_t("top_intent_label")}</span>'
        f'<span class="g-value accent">{top3[0]["score"] * 100:.1f}%</span>'
        f"</div>"
        f'<div class="gap-row">'
        f'<span class="g-label">{_t("second_intent_label")}</span>'
        f'<span class="g-value">{top3[1]["score"] * 100:.1f}%</span>'
        f"</div>"
        f'<div class="gap-row">'
        f'<span class="g-label">{_t("gap_label")}</span>'
        f'<span class="g-value">{gap_pct:.1f} {_t("points_label")}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    note_key = "decision_note_clear" if gap >= 0.20 else "decision_note_close"
    st.markdown(
        f'<div class="decision-note">{_t(note_key)}</div>',
        unsafe_allow_html=True,
    )


def _render_alternatives(all_sorted):
    lang = st.session_state.lang

    _section_header("sec_alternatives", "tip_alternatives")

    top5 = all_sorted[:5]
    rows_html = ""
    for i, (label, score) in enumerate(top5):
        display_lbl = translate_label(label, lang)
        rows_html += (
            f"<tr><td>{i + 1}</td><td>{display_lbl}</td>"
            f'<td class="score-cell">{score * 100:.1f}%</td></tr>'
        )

    st.markdown(
        f'<table class="alt-table">'
        f"<thead><tr>"
        f'<th>{_t("col_rank")}</th>'
        f'<th>{_t("col_intent")}</th>'
        f'<th>{_t("col_score")}</th>'
        f"</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table>",
        unsafe_allow_html=True,
    )


def _render_retrieval(retrieved_examples):
    lang = st.session_state.lang

    _section_header("sec_retrieval", "tip_retrieval")

    st.markdown(
        f'<p style="font-size:0.85rem;color:rgba(18,56,50,0.55);'
        f'margin-bottom:10px;">{_t("retrieval_note")}</p>',
        unsafe_allow_html=True,
    )

    if not retrieved_examples:
        st.markdown(
            f'<p style="font-size:0.88rem;color:rgba(18,56,50,0.55);">'
            f'{_t("retrieval_unavailable")}</p>',
            unsafe_allow_html=True,
        )
        return

    rows_html = ""
    for item in retrieved_examples:
        example_display = item["text"]
        intent_display = translate_label(item["label"], lang)
        sim_score = item["score"]
        rows_html += (
            f"<tr>"
            f"<td>{example_display}</td>"
            f"<td>{intent_display}</td>"
            f'<td class="score-cell">{sim_score:.2f}</td>'
            f"</tr>"
        )

    st.markdown(
        f'<table class="alt-table">'
        f"<thead><tr>"
        f'<th>{_t("retrieval_example")}</th>'
        f'<th>{_t("retrieval_intent")}</th>'
        f'<th>{_t("retrieval_similarity")}</th>'
        f"</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table>",
        unsafe_allow_html=True,
    )

    distinct_intents = len(set(item["label"] for item in retrieved_examples))
    if distinct_intents == 1:
        span_text = _t("retrieval_span_one")
    else:
        span_text = _t("retrieval_span_many").format(count=distinct_intents)

    st.markdown(
        f'<p style="font-size:0.78rem;color:rgba(18,56,50,0.5);'
        f'margin-top:6px;">{span_text}</p>',
        unsafe_allow_html=True,
    )


def _render_confusion(top1_label):
    lang = st.session_state.lang

    _section_header("sec_confusion", "tip_confusion")

    df_confusions = load_csv("preselected_candidate_top_20_confusions.csv")
    shown = False

    if not df_confusions.empty:
        cols_lower = {c: c.lower().replace(" ", "_") for c in df_confusions.columns}
        df_c = df_confusions.rename(columns=cols_lower)

        true_col = next(
            (c for c in ["true_label", "true", "actual"] if c in df_c.columns),
            df_c.columns[0],
        )
        pred_col = next(
            (c for c in ["predicted_label", "predicted", "pred"] if c in df_c.columns),
            df_c.columns[1],
        )
        count_col = next(
            (c for c in ["count", "n", "frequency"] if c in df_c.columns),
            df_c.columns[-1],
        )

        mask = (df_c[true_col] == top1_label) | (df_c[pred_col] == top1_label)
        sub = df_c[mask]

        if not sub.empty:
            st.markdown(
                f'<p style="font-size:0.88rem;color:rgba(18,56,50,0.6);'
                f'margin-bottom:10px;">{_t("confusion_intro")}</p>',
                unsafe_allow_html=True,
            )

            partners = []
            for _, row in sub.iterrows():
                partner = (
                    row[pred_col] if row[true_col] == top1_label else row[true_col]
                )
                partners.append((partner, int(row[count_col])))

            partners.sort(key=lambda x: x[1], reverse=True)
            for name, cnt in partners[:5]:
                partner_display = translate_label(name, lang)
                st.markdown(
                    f'<div class="confusion-item">'
                    f'<span class="c-name">{partner_display}</span>'
                    f'<span class="c-count">{cnt} {_t("times")}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            shown = True

    if not shown:
        st.markdown(
            f'<p style="font-size:0.88rem;color:rgba(18,56,50,0.55);">'
            f'{_t("no_confusion")}</p>',
            unsafe_allow_html=True,
        )


def _render_advanced(top1_label, top1_display_label):
    lang = st.session_state.lang

    with st.expander(_t("sec_advanced")):
        st.markdown(
            f'<p style="font-size:0.82rem;color:rgba(18,56,50,0.55);'
            f'margin-bottom:14px;font-style:italic;">{_t("advanced_context")}</p>',
            unsafe_allow_html=True,
        )

        df_metrics = load_csv("preselected_candidate_per_class_metrics.csv")
        if not df_metrics.empty:
            intent_row = df_metrics[df_metrics["label"] == top1_label]
            if not intent_row.empty:
                ir = intent_row.iloc[0]
                st.markdown(
                    f'<div class="section-title">{_t("advanced_f1")}: '
                    f"{top1_display_label}</div>",
                    unsafe_allow_html=True,
                )
                m1, m2, m3, m4 = st.columns(4)
                metric_pairs = [
                    (m1, "f1_label", ir["f1_score"]),
                    (m2, "precision_label", ir["precision"]),
                    (m3, "recall_label", ir["recall"]),
                    (m4, "support_label", int(ir["support"])),
                ]
                for col, label_key, value in metric_pairs:
                    fmt = f"{value:.3f}" if isinstance(value, float) else str(value)
                    with col:
                        st.markdown(
                            f'<div class="metric-inline">'
                            f'<span class="m-label">{_t(label_key)}</span>'
                            f'<span class="m-value">{fmt}</span></div>',
                            unsafe_allow_html=True,
                        )

        st.markdown("<br>", unsafe_allow_html=True)

        df_seeds = load_csv("e1_per_class_seed_f1.csv")
        if not df_seeds.empty:
            row = df_seeds[df_seeds["label"] == top1_label]
            if not row.empty:
                r = row.iloc[0]
                seed_data = {}
                for sc in ["f1_seed_42", "f1_seed_123", "f1_seed_2026"]:
                    if sc in r.index:
                        seed_data[sc.replace("f1_seed_", "")] = float(r[sc])
                if seed_data:
                    st.markdown(
                        f'<div class="section-title">{_t("advanced_seeds")}</div>',
                        unsafe_allow_html=True,
                    )
                    fig_seed = charts.chart_seed_stability(
                        seed_data, top1_display_label, lang=lang
                    )
                    st.plotly_chart(
                        fig_seed,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )


def _render_results_page():
    lang = st.session_state.lang

    st.markdown(
        f'<div class="results-header">'
        f"<h1>{_t('results_title')}</h1>"
        f'<div class="intro">{_t("results_intro")}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    _section_header("sec_model_compare", "tip_model_compare")
    st.markdown(
        f'<div class="finding-text">{_t("model_compare_finding")}</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        df_comp = load_csv("final_model_comparison.csv")
        if not df_comp.empty:
            fig_mc = charts.chart_model_comparison(df_comp, lang=lang)
            st.plotly_chart(
                fig_mc, use_container_width=True, config={"displayModeBar": False}
            )
        else:
            st.caption(_t("no_data"))

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        _section_header("sec_f1_dist", "tip_f1_dist")
        st.markdown(
            f'<div class="finding-text">{_t("f1_dist_finding")}</div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            df_pc = load_csv("preselected_candidate_per_class_metrics.csv")
            if not df_pc.empty:
                fig_f1 = charts.chart_f1_distribution(df_pc, lang=lang)
                st.plotly_chart(
                    fig_f1,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.caption(_t("no_data"))

    with col2:
        _section_header("sec_top_confusions", "tip_confusions")
        st.markdown(
            f'<div class="finding-text">{_t("confusions_finding")}</div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            df_conf = load_csv("preselected_candidate_top_20_confusions.csv")
            if not df_conf.empty:
                fig_cf = charts.chart_top_confusions(df_conf, top_n=8, lang=lang)
                st.plotly_chart(
                    fig_cf,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.caption(_t("no_data"))

    st.markdown("<br>", unsafe_allow_html=True)

    _section_header("sec_intent_dist", "tip_intent_dist")
    st.markdown(
        f'<div class="finding-text">{_t("intent_dist_finding")}</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        df_dist = load_csv("eda_msa_train_intent_counts.csv")
        if not df_dist.empty:
            fig_id = charts.chart_intent_distribution(df_dist, top_n=20, lang=lang)
            st.plotly_chart(
                fig_id, use_container_width=True, config={"displayModeBar": False}
            )
        else:
            st.caption(_t("no_data"))

    st.markdown("<br>", unsafe_allow_html=True)

    _section_header("sec_text_length", "tip_text_length")
    st.markdown(
        f'<div class="finding-text">{_t("text_length_finding")}</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        df_len = load_csv("eda_text_length_summary.csv")
        if not df_len.empty:
            fig_len = charts.chart_text_length(df_len, lang=lang)
            st.plotly_chart(
                fig_len, use_container_width=True, config={"displayModeBar": False}
            )
        else:
            st.caption(_t("no_data"))


with st.sidebar:
    page = _render_sidebar()


if page == "try":

    _render_header()

    try:
        latency_val = measure_latency()
        latency_str = f"{latency_val}ms"
    except Exception:
        latency_str = "~830ms"

    _render_kpi_row(latency_str)

    user_text = _render_input_area()

    if user_text and user_text.strip():
        predict_unified_fn, retrieve_examples_fn = load_predictor()

        with st.spinner(""):
            t0 = time.time()
            top3, all_scores_raw = predict_unified_fn(user_text.strip())
            pred_latency = round((time.time() - t0) * 1000)
            retrieved_examples = retrieve_examples_fn(user_text.strip(), top_k=3)

        all_sorted = sorted(all_scores_raw, key=lambda x: x[1], reverse=True)
        top1_label = top3[0]["label"]
        top1_display_label = translate_label(top1_label, st.session_state.lang)

        st.markdown("<br>", unsafe_allow_html=True)

        _render_prediction(top3, pred_latency)

        _render_decision(top3)

        st.markdown("<br>", unsafe_allow_html=True)

        _render_alternatives(all_sorted)

        st.markdown("<br>", unsafe_allow_html=True)

        _render_retrieval(retrieved_examples)

        st.markdown("<br>", unsafe_allow_html=True)

        _render_confusion(top1_label)

        st.markdown("<br>", unsafe_allow_html=True)

        _render_advanced(top1_label, top1_display_label)


elif page == "results":
    _render_results_page()
