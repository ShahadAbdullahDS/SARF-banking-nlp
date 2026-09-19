import plotly.graph_objects as go
import pandas as pd
import numpy as np
from translations import translate_label

TEAL = "#167C73"
TEAL_DARK = "#123832"
ROSE = "#b97573"
CREAM = "#f3f1eb"
SAGE = "#c9ddd7"

TEXT_PRIMARY = "#123832"
TEXT_MUTED = "rgba(18,56,50,0.7)"

SEQUENCE = [TEAL, ROSE, "#3498DB", "#D4A574", "#9B59B6", "#e67e22",
            "#1abc9c", "#2980b9", "#8e44ad", "#27ae60", "#c0392b", "#f39c12"]

GRID = "rgba(18,56,50,0.08)"
GRID_AXIS = "rgba(18,56,50,0.14)"

_AXIS_LABELS = {
    "en": {
        "saudi_macro_f1": "Saudi Macro F1",
        "f1_score": "F1 Score",
        "num_intents": "Number of Intents",
        "misclass_count": "Misclassification Count",
        "count": "Count",
        "words": "Words",
        "mean": "Mean",
        "median": "Median",
        "p95": "95th percentile",
        "seed": "Seed",
    },
    "ar": {
        "saudi_macro_f1": "Saudi Macro F1",
        "f1_score": "درجة F1",
        "num_intents": "عدد النوايا",
        "misclass_count": "عدد أخطاء التصنيف",
        "count": "العدد",
        "words": "كلمات",
        "mean": "المتوسط",
        "median": "الوسيط",
        "p95": "النسبة المئوية 95",
        "seed": "البذرة",
    },
}


def _ax(key: str, lang: str = "en") -> str:
    return _AXIS_LABELS.get(lang, _AXIS_LABELS["en"]).get(key, key)


_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY, size=14),
    margin=dict(l=48, r=16, t=32, b=40),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID_AXIS,
               tickfont=dict(size=13, color=TEXT_MUTED)),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID_AXIS,
               tickfont=dict(size=13, color=TEXT_MUTED)),
    legend=dict(font=dict(size=13, color=TEXT_PRIMARY), bgcolor="rgba(0,0,0,0)", borderwidth=0),
    hoverlabel=dict(bgcolor=TEAL_DARK, font_size=14, font_color=CREAM, bordercolor=TEAL),
    colorway=SEQUENCE,
)


def _apply_base(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(**_BASE_LAYOUT, height=height)
    return fig


def chart_model_comparison(df: pd.DataFrame, lang: str = "en") -> go.Figure:
    cols_lower = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=cols_lower)

    label_col = "model" if "model" in df.columns else df.columns[0]
    metric_col = "macro_f1_mean" if "macro_f1_mean" in df.columns else None
    std_col = "macro_f1_std" if "macro_f1_std" in df.columns else None

    if metric_col is None:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]) and c != label_col and "change" not in c and "seed" not in c and "id" not in c:
                metric_col = c
                break

    if metric_col is None:
        fig = go.Figure()
        fig.add_annotation(text="No numeric data found", showarrow=False,
                           font=dict(color=TEXT_MUTED, size=13))
        return _apply_base(fig)

    df_sorted = df.sort_values(metric_col, ascending=True)

    fig = go.Figure(go.Bar(
        x=df_sorted[label_col], y=df_sorted[metric_col],
        marker_color=TEAL,
        marker_line=dict(width=0),
        text=[f"{v:.3f}" for v in df_sorted[metric_col]],
        textposition="outside",
        textfont=dict(size=12, color=TEXT_MUTED),
    ))

    if std_col and std_col in df_sorted.columns:
        has_std = df_sorted[std_col] > 0
        if has_std.any():
            sub = df_sorted[has_std]
            fig.add_trace(go.Scatter(
                x=sub[label_col], y=sub[metric_col],
                error_y=dict(type="data", array=sub[std_col].tolist(), visible=True,
                             color="rgba(18,56,50,0.3)", thickness=1.5, width=4),
                mode="markers",
                marker=dict(size=0, color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ))

    fig.update_layout(
        yaxis_title=_ax("saudi_macro_f1", lang),
        xaxis_title="",
        showlegend=False,
        bargap=0.3,
        yaxis=dict(range=[0, df_sorted[metric_col].max() * 1.15]),
    )
    return _apply_base(fig, height=380)


def chart_intent_distribution(df: pd.DataFrame, top_n: int = 20, lang: str = "en") -> go.Figure:
    cols_lower = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=cols_lower)

    label_col = next((c for c in ["label", "intent", "class", "name"]
                      if c in df.columns), df.columns[0])
    count_col = next((c for c in ["count", "frequency", "samples", "n",
                                   "number_of_training_examples"]
                      if c in df.columns), df.columns[1])

    df_sorted = df.sort_values(count_col, ascending=True).tail(top_n)
    display_labels = [translate_label(l, lang) for l in df_sorted[label_col]]

    fig = go.Figure(go.Bar(
        y=display_labels,
        x=df_sorted[count_col],
        orientation="h",
        marker_color=TEAL,
        marker_line=dict(width=0),
    ))
    fig.update_layout(
        xaxis_title=_ax("count", lang),
        yaxis=dict(tickfont=dict(size=12, color=TEXT_MUTED)),
        margin=dict(l=220, r=16, t=20, b=40),
    )
    return _apply_base(fig, height=max(400, top_n * 26))


def chart_f1_distribution(df: pd.DataFrame, lang: str = "en") -> go.Figure:
    cols_lower = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=cols_lower)

    f1_col = next((c for c in ["f1_score", "f1", "f1_mean", "macro_f1"]
                   if c in df.columns), None)
    if f1_col is None:
        fig = go.Figure()
        fig.add_annotation(text="No F1 data found", showarrow=False,
                           font=dict(color=TEXT_MUTED))
        return _apply_base(fig)

    values = df[f1_col].dropna()

    fig = go.Figure(go.Histogram(
        x=values,
        nbinsx=20,
        marker_color=TEAL,
        marker_line=dict(width=1, color=SAGE),
    ))
    mean_val = values.mean()
    mean_label = _ax("mean", lang)
    fig.add_vline(x=mean_val, line_dash="dash", line_color=ROSE,
                  annotation_text=f"{mean_label} {mean_val:.2f}",
                  annotation_font_color=ROSE)
    fig.update_layout(
        xaxis_title=_ax("f1_score", lang),
        yaxis_title=_ax("num_intents", lang),
        bargap=0.05,
    )
    return _apply_base(fig, height=340)


def chart_text_length(df: pd.DataFrame, lang: str = "en") -> go.Figure:
    cols_lower = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=cols_lower)

    split_col = next((c for c in ["split", "dataset", "set"]
                      if c in df.columns), df.columns[0])

    metrics = []
    for candidate in ["mean_words", "median_words", "95th_percentile_words"]:
        if candidate in df.columns:
            metrics.append(candidate)

    if not metrics:
        metrics = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:3]

    nice_names = {
        "mean_words": _ax("mean", lang),
        "median_words": _ax("median", lang),
        "95th_percentile_words": _ax("p95", lang),
    }

    fig = go.Figure()
    for i, m in enumerate(metrics):
        fig.add_trace(go.Bar(
            x=df[split_col],
            y=df[m],
            name=nice_names.get(m, m),
            marker_color=SEQUENCE[i],
            text=[f"{v:.0f}" for v in df[m]],
            textposition="outside",
            textfont=dict(size=12, color=TEXT_MUTED),
        ))

    fig.update_layout(
        barmode="group",
        yaxis_title=_ax("words", lang),
        xaxis_title="",
        bargap=0.25,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return _apply_base(fig, height=320)


def chart_top_confusions(df: pd.DataFrame, top_n: int = 10, lang: str = "en") -> go.Figure:
    cols_lower = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=cols_lower)

    true_col = next((c for c in ["true_label", "true", "actual"] if c in df.columns), df.columns[0])
    pred_col = next((c for c in ["predicted_label", "predicted", "pred"] if c in df.columns), df.columns[1])
    count_col = next((c for c in ["count", "n", "frequency"] if c in df.columns), df.columns[-1])

    df_sorted = df.sort_values(count_col, ascending=True).tail(top_n)
    labels = [
        f"{translate_label(t, lang)}  →  {translate_label(p, lang)}"
        for t, p in zip(df_sorted[true_col], df_sorted[pred_col])
    ]

    fig = go.Figure(go.Bar(
        y=labels,
        x=df_sorted[count_col],
        orientation="h",
        marker_color=ROSE,
        marker_line=dict(width=0),
        text=[str(int(v)) for v in df_sorted[count_col]],
        textposition="outside",
        textfont=dict(size=11, color=TEXT_MUTED),
    ))
    fig.update_layout(
        xaxis_title=_ax("misclass_count", lang),
        yaxis=dict(tickfont=dict(size=12, color=TEXT_MUTED)),
        margin=dict(l=300, r=40, t=20, b=40),
    )
    return _apply_base(fig, height=max(360, top_n * 34))


def chart_seed_stability(seed_data: dict, intent_name: str, lang: str = "en") -> go.Figure:
    seeds = list(seed_data.keys())
    f1_vals = list(seed_data.values())

    colors = [TEAL if s == "2026" else "rgba(22,124,115,0.45)" for s in seeds]

    fig = go.Figure(go.Bar(
        x=[f"{_ax('seed', lang)} {s}" for s in seeds],
        y=f1_vals,
        marker_color=colors,
        marker_line=dict(width=0),
        text=[f"{v:.3f}" for v in f1_vals],
        textposition="outside",
        textfont=dict(size=12, color=TEXT_MUTED),
    ))
    fig.update_layout(
        yaxis_title=_ax("f1_score", lang),
        xaxis_title="",
        yaxis=dict(range=[0, max(f1_vals) * 1.3] if f1_vals else [0, 1]),
        bargap=0.35,
    )
    return _apply_base(fig, height=260)
