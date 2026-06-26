"""Local Streamlit dashboard for World Cup prediction model outputs."""

from __future__ import annotations

import json
import os
from datetime import date
from html import escape
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.live_update import WORLD_CUP_FINAL_DATE, refresh_live_outputs, world_cup_updates_are_active


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
AUTO_REFRESH_TTL_SECONDS = 15 * 60


MODEL_ROWS = [
    {
        "Name": "Elo Model",
        "Kept Model": "Elo baseline",
        "Architecture": "Team-strength rating system",
        "Scope": "2018+",
        "Log Loss": 0.8927,
        "Accuracy": 0.5995,
        "Brier": 0.5257,
        "Takeaway": "Simple and stable benchmark.",
    },
    {
        "Name": "Regression Model",
        "Kept Model": "Logistic regression",
        "Architecture": "Linear classifier",
        "Scope": "2022+",
        "Log Loss": 0.8646,
        "Accuracy": 0.6044,
        "Brier": 0.5093,
        "Takeaway": "Interpretable feature-based baseline.",
    },
    {
        "Name": "Gradient Boosting Model",
        "Kept Model": "GB postprocess current best",
        "Architecture": "Tree-based nonlinear classifier",
        "Scope": "2022+",
        "Log Loss": 0.8632,
        "Accuracy": 0.6074,
        "Brier": 0.5076,
        "Takeaway": "Best practical current model.",
    },
    {
        "Name": "Blended Model",
        "Kept Model": "Logistic/GB blend",
        "Architecture": "Weighted probability blend",
        "Scope": "2022+",
        "Log Loss": 0.8621,
        "Accuracy": 0.6051,
        "Brier": 0.5076,
        "Takeaway": "Best clean log-loss blend.",
    },
    {
        "Name": "Poisson Goal Model",
        "Kept Model": "Dixon-Coles / Poisson goal model",
        "Architecture": "Goal-rate model",
        "Scope": "2018+",
        "Log Loss": 0.9450,
        "Accuracy": 0.5998,
        "Brier": 0.5458,
        "Takeaway": "Useful for score and draw reasoning.",
    },
    {
        "Name": "Market-Adjusted WC Elo Model",
        "Kept Model": "WC Elo + market value",
        "Architecture": "World Cup rating model with squad value",
        "Scope": "WC 2006-2022",
        "Log Loss": 0.9646,
        "Accuracy": 0.5844,
        "Brier": 0.5606,
        "Takeaway": "Best World Cup-specific Elo variant.",
    },
]


MODEL_DESCRIPTIONS = {
    "Elo Model": {
        "summary": "Rates teams from historical results and converts the rating gap into win, draw, and loss probabilities.",
        "how": (
            "Each match updates both teams' ratings based on result, opponent strength, "
            "margin, and contextual adjustments. It is transparent and hard to overfit, "
            "but it cannot learn complex feature interactions."
        ),
    },
    "Regression Model": {
        "summary": "Uses logistic regression to map engineered match features directly into W/D/L probabilities.",
        "how": (
            "The model learns one set of linear weights for each outcome class. Features such as Elo gaps, form, "
            "market value, injuries, phase flags, and context can push probability toward a home/team A win, draw, "
            "or team B win."
        ),
    },
    "Gradient Boosting Model": {
        "summary": "Uses boosted decision trees to learn nonlinear patterns from the engineered features.",
        "how": (
            "It builds many small trees in sequence. Each tree corrects errors left by the earlier trees, which lets "
            "the model capture threshold effects such as strong-favorite guardrails, phase behavior, and interaction "
            "between team quality and match context."
        ),
    },
    "Blended Model": {
        "summary": "Averages probabilities from the regression and gradient boosting models.",
        "how": (
            "Blending reduces dependence on one modeling style. The regression model contributes smoother, more "
            "stable probabilities, while gradient boosting contributes nonlinear corrections."
        ),
    },
    "Poisson Goal Model": {
        "summary": "Predicts expected goals for each team, then derives W/D/L probabilities from score distributions.",
        "how": (
            "The model estimates the chance of scorelines such as 0-0, 1-0, 1-1, and so on. A Dixon-Coles-style "
            "adjustment can shift low-score outcomes, which is useful for reasoning about draws."
        ),
    },
    "Market-Adjusted WC Elo Model": {
        "summary": "A World Cup-only Elo variant that adjusts team strength using squad market value.",
        "how": (
            "It starts from tournament-era Elo ratings, then applies a conservative squad-value adjustment. This helps "
            "avoid clearly wrong picks when a rating signal underrates a squad with substantially stronger players."
        ),
    },
}


FEATURE_NOTES = [
    "StatsBomb data is treated as a feature source, not a standalone model family.",
    "Injuries, market value, altitude, rotation risk, and World Cup priors are also feature inputs.",
    "The current best practical model is still the gradient boosting postprocess version.",
]


MODEL_2026_OUTPUTS = {
    "Blended Model": "blended",
    "Elo Model": "elo",
    "Poisson Goal Model": "poisson_goal",
    "Gradient Boosting Model": "gradient_boosting",
    "Regression Model": "regression",
    "Market-Adjusted WC Elo Model": "market_adjusted_wc_elo",
}


@st.cache_data
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


@st.cache_data
def read_json(path: str) -> dict[str, Any]:
    with (ROOT / path).open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(value: float) -> str:
    return f"{value:.1%}"


def safe_read_csv(path: str) -> pd.DataFrame:
    full_path = ROOT / path
    if not full_path.exists():
        return pd.DataFrame()
    return read_csv(path)


def safe_read_live_csv(path: str) -> pd.DataFrame:
    full_path = ROOT / path
    if not full_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(full_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def safe_read_json(path: str) -> dict[str, Any]:
    full_path = ROOT / path
    if not full_path.exists():
        return {}
    return read_json(path)


@st.cache_data(ttl=AUTO_REFRESH_TTL_SECONDS, show_spinner=False)
def auto_refresh_live_outputs(today_iso: str) -> dict[str, Any]:
    """Refresh live data once per cache window while the tournament is active."""
    today = date.fromisoformat(today_iso)
    if not world_cup_updates_are_active(today):
        return {
            "status": "inactive",
            "message": f"Automatic updates stopped after {WORLD_CUP_FINAL_DATE.isoformat()}.",
        }
    if not os.environ.get("FOOTBALL_DATA_API_KEY"):
        return {
            "status": "missing_key",
            "message": "Set FOOTBALL_DATA_API_KEY to enable automatic match updates.",
        }
    try:
        return refresh_live_outputs(today=today)
    except Exception as exc:  # noqa: BLE001 - surface API/update failures in the dashboard.
        return {"status": "error", "message": str(exc)}


def model_table() -> pd.DataFrame:
    df = pd.DataFrame(MODEL_ROWS)
    return df


def metric_display_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    display["Accuracy"] = display["Accuracy"].map(pct)
    display["Log Loss"] = display["Log Loss"].map(lambda value: f"{value:.4f}")
    display["Brier"] = display["Brier"].map(lambda value: f"{value:.4f}")
    return display


def model_metric_card(label: str, model_name: str, value: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-model">{model_name}</div>
        <div class="metric-value"><span class="best-pill">{value}</span></div>
    </div>
    """


def render_metric_cards(df: pd.DataFrame) -> None:
    best_log_loss = df.loc[df["Log Loss"].idxmin()]
    best_accuracy = df.loc[df["Accuracy"].idxmax()]
    best_brier = df.loc[df["Brier"].idxmin()]

    col1, col2, col3 = st.columns(3)
    col1.markdown(
        model_metric_card("Lowest Log Loss", best_log_loss["Name"], f"{best_log_loss['Log Loss']:.4f}"),
        unsafe_allow_html=True,
    )
    col2.markdown(
        model_metric_card("Highest Accuracy", best_accuracy["Name"], pct(float(best_accuracy["Accuracy"]))),
        unsafe_allow_html=True,
    )
    col3.markdown(
        model_metric_card("Lowest Brier", best_brier["Name"], f"{best_brier['Brier']:.4f}"),
        unsafe_allow_html=True,
    )


def render_model_cards(df: pd.DataFrame, reference_df: pd.DataFrame | None = None) -> None:
    comparison_df = df if reference_df is None else reference_df
    best_log_loss = float(comparison_df["Log Loss"].min())
    best_accuracy = float(comparison_df["Accuracy"].max())
    best_brier = float(comparison_df["Brier"].min())

    def metric_value(metric: str, value: float, best_value: float, lower_is_better: bool = True) -> str:
        is_best = value == best_value
        formatted = pct(value) if metric == "Accuracy" else f"{value:.4f}"
        class_name = "best-pill" if is_best else "metric-plain"
        return f'<strong><span class="{class_name}">{formatted}</span></strong>'

    rows = list(df.to_dict("records"))
    for start in range(0, len(rows), 2):
        cols = st.columns(2)
        for col, row in zip(cols, rows[start : start + 2], strict=False):
            log_loss = metric_value("Log Loss", float(row["Log Loss"]), best_log_loss)
            accuracy = metric_value("Accuracy", float(row["Accuracy"]), best_accuracy, lower_is_better=False)
            brier = metric_value("Brier", float(row["Brier"]), best_brier)
            with col:
                st.markdown(
                    f"""
                    <div class="model-card">
                        <div class="model-card-title">{row['Name']}</div>
                        <div class="model-card-subtitle">{row['Architecture']}</div>
                        <div class="model-card-body">{row['Takeaway']}</div>
                        <div class="model-card-grid">
                            <div><span>Log Loss</span>{log_loss}</div>
                            <div><span>Accuracy</span>{accuracy}</div>
                            <div><span>Brier</span>{brier}</div>
                        </div>
                        <div class="model-card-foot">{row['Kept Model']} · {row['Scope']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def metric_chart(df: pd.DataFrame, metric: str) -> alt.Chart:
    chart_df = df[["Name", metric]].copy()
    if metric == "Accuracy":
        chart_df = chart_df.sort_values(metric, ascending=False)
        value_format = ".1%"
        subtitle = "Raw accuracy values; higher is better."
    else:
        chart_df = chart_df.sort_values(metric, ascending=True)
        value_format = ".4f"
        subtitle = f"Raw {metric.lower()} values; lower is better."

    chart_df["Actual"] = chart_df[metric].map(lambda value: pct(value) if metric == "Accuracy" else f"{value:.4f}")
    min_value = float(chart_df[metric].min())
    max_value = float(chart_df[metric].max())
    padding = max((max_value - min_value) * 0.18, 0.002 if metric == "Accuracy" else 0.01)
    y_min = max(0.0, min_value - padding)
    y_max = max_value + padding
    chart_df["Baseline"] = y_min

    base = alt.Chart(chart_df).encode(
        x=alt.X("Name:N", sort=list(chart_df["Name"]), title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y(
            f"{metric}:Q",
            title=metric.lower(),
            scale=alt.Scale(domain=[y_min, y_max], zero=False),
            axis=alt.Axis(format="%" if metric == "Accuracy" else ".3f"),
        ),
        tooltip=[
            alt.Tooltip("Name:N", title="Model"),
            alt.Tooltip(f"{metric}:Q", title=metric, format=value_format),
        ],
    )
    bars = (
        base
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            y2=alt.Y2("Baseline:Q"),
            color=alt.Color("Name:N", legend=None, scale=alt.Scale(scheme="tableau20")),
        )
    )
    return bars.properties(
        height=360,
        title=alt.TitleParams(text=f"{metric} Comparison", subtitle=subtitle),
    )


def render_model_comparison() -> None:
    st.header("Model Comparison")
    st.caption(
        "The table keeps one representative per model architecture. StatsBomb, injuries, market value, "
        "rotation risk, altitude, and priors are feature sources rather than separate model families."
    )

    df = model_table()
    render_metric_cards(df)

    selected_model = st.selectbox(
        "Model details",
        df.sort_values("Name")["Name"].tolist(),
        index=df.sort_values("Name")["Name"].tolist().index("Gradient Boosting Model"),
    )
    selected_df = df[df["Name"].eq(selected_model)].copy()
    render_model_cards(selected_df, reference_df=df)

    st.subheader("Metric View")
    chart_metric = st.radio(
        "Chart metric",
        ["Log Loss", "Accuracy", "Brier"],
        horizontal=True,
    )
    st.altair_chart(metric_chart(df, chart_metric), width="stretch")


def render_model_descriptions() -> None:
    st.header("How The Models Work")
    st.caption("These descriptions focus on architecture, not which data source was added to the model.")

    for row in MODEL_ROWS:
        name = row["Name"]
        description = MODEL_DESCRIPTIONS[name]
        with st.container(border=True):
            st.subheader(name)
            cols = st.columns([1, 2])
            cols[0].markdown(f"**Representative:** {row['Kept Model']}")
            cols[0].markdown(f"**Architecture:** {row['Architecture']}")
            cols[0].markdown(f"**Best use:** {row['Takeaway']}")
            cols[1].markdown(f"**Summary:** {description['summary']}")
            cols[1].markdown(f"**How it works:** {description['how']}")

    st.info("Feature sources: " + " ".join(FEATURE_NOTES))


def format_kickoff(value: object) -> str:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return "Kickoff TBD"
    local = timestamp.tz_convert("America/New_York")
    return local.strftime("%b %-d, %-I:%M %p ET")


def is_knockout_stage(stage: object) -> bool:
    text = str(stage or "").upper()
    return bool(text and "GROUP" not in text)


def probability_value(value: object) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return 0.0
    return max(0.0, min(1.0, float(numeric)))


def top_outcome(row: pd.Series, team_a: str, team_b: str) -> str:
    outcomes = {
        f"{team_a} win": probability_value(row.get("team_a_win_prob")),
        "Draw": probability_value(row.get("draw_prob")),
        f"{team_b} win": probability_value(row.get("team_b_win_prob")),
    }
    return max(outcomes.items(), key=lambda item: item[1])[0]


def render_probability_bar(value: object) -> None:
    probability = probability_value(value)
    st.progress(probability, text=pct(probability))


def render_todays_matches() -> None:
    st.header("Today's Matches")
    today_iso = date.today().isoformat()
    with st.spinner("Refreshing live match data..."):
        refresh_status = auto_refresh_live_outputs(today_iso)
    if refresh_status.get("status") == "updated":
        st.caption(
            f"Auto-updated from football-data.org. Cached matches: {refresh_status.get('matches', 0)}. "
            f"Prediction rows: {refresh_status.get('predictions', 0)}."
        )
    elif refresh_status.get("status") == "missing_key":
        st.info(str(refresh_status["message"]))
    elif refresh_status.get("status") == "inactive":
        st.caption(str(refresh_status["message"]))
    elif refresh_status.get("status") == "error":
        st.warning(f"Automatic update failed: {refresh_status['message']}")

    predictions = safe_read_live_csv("results/todays_match_predictions.csv")
    if predictions.empty:
        st.info(
            "No saved predictions for today's matches yet. Set `FOOTBALL_DATA_API_KEY` and refresh the app, "
            "or run `python scripts/update_world_cup_live_data.py`."
        )
        return

    probability_columns = [
        "team_a_win_prob",
        "draw_prob",
        "team_b_win_prob",
        "team_a_advancement_prob",
        "team_b_advancement_prob",
    ]
    for column in probability_columns:
        if column in predictions.columns:
            predictions[column] = pd.to_numeric(predictions[column], errors="coerce")

    match_columns = ["match_id", "kickoff_utc", "status", "stage", "group", "team_a", "team_b"]
    matches = predictions[match_columns].drop_duplicates().sort_values(["kickoff_utc", "match_id"])
    for match in matches.itertuples(index=False):
        match_predictions = predictions[predictions["match_id"].astype(str).eq(str(match.match_id))].copy()
        team_a = str(match.team_a)
        team_b = str(match.team_b)
        show_advancement = is_knockout_stage(match.stage)
        with st.container(border=True):
            title_col, status_col = st.columns([3, 1])
            title_col.subheader(f"{team_a} vs {team_b}")
            status_col.caption(str(match.status or "Scheduled"))
            st.caption(
                " · ".join(
                    item
                    for item in [
                        format_kickoff(match.kickoff_utc),
                        str(match.stage or ""),
                        str(match.group or ""),
                    ]
                    if item
                )
            )

            if show_advancement:
                header_weights = [1.35, 1, 1, 1, 1, 1]
                header_labels = [
                    "Model",
                    f"{team_a} win",
                    "Draw",
                    f"{team_b} win",
                    f"{team_a} advances",
                    f"{team_b} advances",
                ]
            else:
                header_weights = [1.35, 1, 1, 1]
                header_labels = ["Model", f"{team_a} win", "Draw", f"{team_b} win"]

            header_cols = st.columns(header_weights)
            for col, label in zip(header_cols, header_labels, strict=False):
                col.markdown(f"**{label}**")

            for index, row in enumerate(match_predictions.itertuples(index=False)):
                row_series = pd.Series(row._asdict())
                cols = st.columns(header_weights)
                cols[0].markdown(f"**{row_series.get('model', 'Model')}**")
                cols[0].caption(f"Top: {top_outcome(row_series, team_a, team_b)}")
                with cols[1]:
                    render_probability_bar(row_series.get("team_a_win_prob"))
                with cols[2]:
                    render_probability_bar(row_series.get("draw_prob"))
                with cols[3]:
                    render_probability_bar(row_series.get("team_b_win_prob"))
                if show_advancement:
                    with cols[4]:
                        render_probability_bar(row_series.get("team_a_advancement_prob"))
                    with cols[5]:
                        render_probability_bar(row_series.get("team_b_advancement_prob"))
                if index < len(match_predictions) - 1:
                    st.divider()


def final_from_temp_snapshot() -> dict[str, Any] | None:
    df = safe_read_csv("results/temp_2026_world_cup_current_model_predictions.csv")
    if df.empty or "phase" not in df.columns:
        return None
    final = df[df["phase"].eq("final")]
    if final.empty:
        return None
    row = final.iloc[0]
    return {
        "Model Output": "Current temporary blended snapshot",
        "Final": f"{row['team_a']} vs {row['team_b']}",
        "Winner": row.get("predicted_advancer", ""),
        "Team A Advance": row.get("team_a_advancement_prob"),
        "Team B Advance": row.get("team_b_advancement_prob"),
    }


def final_from_poisson_json() -> dict[str, Any] | None:
    data = safe_read_json("results/deterministic_poisson_bracket.json")
    final = data.get("final")
    if not final:
        return None
    row = final[0]
    return {
        "Model Output": "Deterministic Poisson bracket",
        "Final": f"{row['team_a']} vs {row['team_b']}",
        "Winner": row.get("winner", data.get("champion", "")),
        "Team A Advance": row.get("team_a_champion_prob"),
        "Team B Advance": row.get("team_b_champion_prob"),
    }


def final_from_sample_json() -> dict[str, Any] | None:
    data = safe_read_json("results/sample_bracket.json")
    final = data.get("final")
    if not final:
        return None
    return {
        "Model Output": "Elo sample bracket",
        "Final": " vs ".join(final),
        "Winner": data.get("champion", ""),
        "Team A Advance": None,
        "Team B Advance": None,
    }


def render_final_cards(finals: list[dict[str, Any]]) -> None:
    cols = st.columns(max(1, len(finals)))
    for col, item in zip(cols, finals, strict=False):
        with col.container(border=True):
            st.caption(item["Model Output"])
            st.subheader(item["Winner"] or "Unavailable")
            st.write(item["Final"])
            a = item.get("Team A Advance")
            b = item.get("Team B Advance")
            if pd.notna(a) and pd.notna(b):
                st.caption(f"Displayed probabilities: {float(a):.1%} / {float(b):.1%}")


def stage_probability_chart(path: str, title: str, key: str) -> None:
    df = safe_read_csv(path)
    if df.empty:
        st.warning(f"Missing {path}")
        return

    top_n = st.slider(f"{title} teams shown", min_value=5, max_value=15, value=10, key=key)
    chart_df = (
        df.sort_values("champion_prob", ascending=False)
        .head(top_n)
        [["team", "champion_prob", "final_prob"]]
    )
    if top_n <= 8:
        label_angle = -25
        label_font_size = 12
        label_limit = 140
    elif top_n <= 12:
        label_angle = -40
        label_font_size = 10
        label_limit = 105
    else:
        label_angle = -55
        label_font_size = 8
        label_limit = 80

    st.subheader(title)
    long = chart_df.melt(
        id_vars="team",
        value_vars=["champion_prob", "final_prob"],
        var_name="Metric",
        value_name="Probability",
    )
    long["Metric"] = long["Metric"].map(
        {
            "champion_prob": "Champion",
            "final_prob": "Final",
        }
    )
    chart = (
        alt.Chart(long)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(
                "team:N",
                sort=list(chart_df["team"]),
                title=None,
                axis=alt.Axis(
                    labelAngle=label_angle,
                    labelFontSize=label_font_size,
                    labelLimit=label_limit,
                    labelOverlap=False,
                ),
            ),
            xOffset=alt.XOffset("Metric:N", sort=["Champion", "Final"]),
            y=alt.Y("Probability:Q", axis=alt.Axis(format="%"), title="probability"),
            color=alt.Color("Metric:N", scale=alt.Scale(range=["#276749", "#2b6cb0"])),
            tooltip=[
                alt.Tooltip("team:N", title="Team"),
                alt.Tooltip("Metric:N"),
                alt.Tooltip("Probability:Q", format=".1%"),
            ],
        )
        .properties(height=390)
    )
    st.altair_chart(chart, width="stretch")
    render_probability_rankings(chart_df)


def render_probability_rankings(chart_df: pd.DataFrame) -> None:
    st.markdown("**Probability ranking**")
    for rank, row in enumerate(chart_df.itertuples(index=False), start=1):
        champion = float(row.champion_prob)
        final = float(row.final_prob)
        with st.container(border=True):
            rank_col, team_col, champion_col, final_col = st.columns([0.5, 1.5, 2, 2])
            rank_col.markdown(f"**#{rank}**")
            team_col.markdown(f"**{row.team}**")
            champion_col.progress(champion, text=f"Champion {champion:.1%}")
            final_col.progress(final, text=f"Final {final:.1%}")


def current_blend_final_chart() -> None:
    bracket = safe_read_csv("results/temp_2026_world_cup_current_model_predictions.csv")
    if bracket.empty:
        st.info("No saved 2026 bracket output yet.")
        return
    final = bracket[bracket["phase"].eq("final")]
    if final.empty:
        st.info("No saved final row in the current bracket output.")
        return
    row = final.iloc[0]
    chart_df = pd.DataFrame(
        [
            {"Team": row["team_a"], "Probability": row["team_a_advancement_prob"]},
            {"Team": row["team_b"], "Probability": row["team_b_advancement_prob"]},
        ]
    ).sort_values("Probability", ascending=False)
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Team:N", sort=list(chart_df["Team"]), title=None),
            y=alt.Y("Probability:Q", axis=alt.Axis(format="%"), title="advancement probability"),
            color=alt.Color("Team:N", legend=None, scale=alt.Scale(scheme="tableau20")),
            tooltip=[alt.Tooltip("Team:N"), alt.Tooltip("Probability:Q", format=".1%")],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, width="stretch")
    render_probability_rankings(
        chart_df.rename(columns={"Team": "team", "Probability": "champion_prob"}).assign(
            final_prob=chart_df["Probability"]
        )
    )


def unavailable_model_graphic(model_name: str) -> None:
    st.markdown(
        f"""
        <div class="empty-graphic">
            <div class="empty-graphic-title">{model_name}</div>
            <div class="empty-graphic-line"></div>
            <div class="empty-graphic-copy">No saved 2026 tournament bracket for this model yet.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_2026_graphics() -> str:
    st.subheader("2026 Outputs By Model Family")
    st.caption("Choose one model family at a time. Teams in probability charts are sorted by champion probability.")
    selected_model = st.selectbox(
        "Model family",
        [
            "Blended Model",
            "Elo Model",
            "Poisson Goal Model",
            "Gradient Boosting Model",
            "Regression Model",
            "Market-Adjusted WC Elo Model",
        ],
    )

    with st.container(border=True):
        model_key = MODEL_2026_OUTPUTS[selected_model]
        probabilities_path = f"results/2026_{model_key}_stage_probabilities.csv"
        if Path(probabilities_path).exists():
            stage_probability_chart(probabilities_path, selected_model, f"{model_key}-chart")
        else:
            st.markdown(f"**{selected_model}**")
            unavailable_model_graphic(selected_model)
    return selected_model


def render_2026_predictions() -> None:
    st.header("2026 Prediction Comparison")
    st.caption(
        "This section uses the saved 2026 outputs already generated by the project. "
        "The temporary blended snapshot is the current main bracket output."
    )

    selected_model = render_model_2026_graphics()

    st.divider()
    st.subheader("Most Likely Bracket")
    render_selected_model_bracket(selected_model)


def deterministic_bracket_frame(model_key: str, model_name: str) -> pd.DataFrame:
    data = safe_read_json(f"results/2026_{model_key}_deterministic_bracket.json")
    if not data:
        return pd.DataFrame()

    rows = []
    phase_map = {
        "round_of_32": "round_of_32",
        "round_of_16": "round_of_16",
        "quarterfinal": "quarterfinal",
        "semifinal": "semifinal",
        "final": "final",
    }
    for key, phase in phase_map.items():
        for match in data.get(key, []):
            rows.append(
                {
                    "model_snapshot": model_name,
                    "phase": phase,
                    "match_number": match.get("match_number"),
                    "team_a": match.get("team_a"),
                    "team_b": match.get("team_b"),
                    "predicted_advancer": match.get("winner"),
                }
            )
    return pd.DataFrame(rows)


def render_selected_model_bracket(selected_model: str) -> None:
    model_key = MODEL_2026_OUTPUTS[selected_model]
    bracket = deterministic_bracket_frame(model_key, selected_model)
    if bracket.empty:
        st.info(f"No deterministic 2026 bracket has been generated yet for {selected_model}.")
        return
    render_two_sided_bracket(bracket, selected_model)


def render_two_sided_bracket(bracket: pd.DataFrame, model_title: str) -> None:
    if bracket.empty:
        st.warning("Missing bracket output.")
        return

    knockout = bracket[bracket["phase"].ne("group")].copy()
    knockout["match_number"] = knockout["match_number"].astype(int)
    matches = {int(row.match_number): row for row in knockout.itertuples(index=False)}
    if 104 not in matches:
        st.info("No saved final row in the current bracket output.")
        return

    champion = str(matches[104].predicted_advancer)

    width = 1240
    height = 820
    box_width = 124
    box_height = 56
    top_y = 104
    step = 82
    left_x = [20, 152, 284, 416]
    right_x = [1096, 964, 832, 700]
    center_x = 558
    final_y = 404

    def truncate(text: str, max_chars: int = 16) -> str:
        text = str(text)
        return text if len(text) <= max_chars else text[: max_chars - 1] + "…"

    def match_positions(side: str) -> dict[int, tuple[float, float]]:
        if side == "left":
            xs = left_x
            r32 = [74, 77, 73, 75, 83, 84, 81, 82]
            r16 = [89, 90, 93, 94]
            qf = [97, 98]
            sf = [101]
        else:
            xs = right_x
            r32 = [76, 78, 79, 80, 86, 88, 85, 87]
            r16 = [91, 92, 95, 96]
            qf = [99, 100]
            sf = [102]

        y_r32 = [top_y + index * step for index in range(8)]
        y_r16 = [(y_r32[i] + y_r32[i + 1]) / 2 for i in range(0, 8, 2)]
        y_qf = [(y_r16[i] + y_r16[i + 1]) / 2 for i in range(0, 4, 2)]
        y_sf = [(y_qf[0] + y_qf[1]) / 2]

        positions: dict[int, tuple[float, float]] = {}
        for number, y in zip(r32, y_r32, strict=False):
            positions[number] = (xs[0], y)
        for number, y in zip(r16, y_r16, strict=False):
            positions[number] = (xs[1], y)
        for number, y in zip(qf, y_qf, strict=False):
            positions[number] = (xs[2], y)
        for number, y in zip(sf, y_sf, strict=False):
            positions[number] = (xs[3], y)
        return positions

    positions = {
        **match_positions("left"),
        **match_positions("right"),
        104: (center_x, final_y),
    }

    connector_pairs = [
        (74, 89),
        (77, 89),
        (73, 90),
        (75, 90),
        (83, 93),
        (84, 93),
        (81, 94),
        (82, 94),
        (89, 97),
        (90, 97),
        (93, 98),
        (94, 98),
        (97, 101),
        (98, 101),
        (101, 104),
        (76, 91),
        (78, 91),
        (79, 92),
        (80, 92),
        (86, 95),
        (88, 95),
        (85, 96),
        (87, 96),
        (91, 99),
        (92, 99),
        (95, 100),
        (96, 100),
        (99, 102),
        (100, 102),
        (102, 104),
    ]

    def connector(start: int, end: int) -> str:
        if start not in positions or end not in positions:
            return ""
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        side = "left" if x1 < center_x else "right"
        start_x = x1 + box_width if side == "left" else x1
        end_x = x2 if side == "left" else x2 + box_width
        start_y = y1 + box_height / 2
        end_y = y2 + box_height / 2
        mid_x = (start_x + end_x) / 2
        return (
            f'<path d="M {start_x:.1f} {start_y:.1f} '
            f'L {mid_x:.1f} {start_y:.1f} '
            f'L {mid_x:.1f} {end_y:.1f} '
            f'L {end_x:.1f} {end_y:.1f}" />'
        )

    def match_svg(match_number: int) -> str:
        match = matches[match_number]
        team_a = str(match.team_a)
        team_b = str(match.team_b)
        winner = str(match.predicted_advancer)
        x, y = positions[match_number]
        team_a_class = "winner" if team_a == winner else "team"
        team_b_class = "winner" if team_b == winner else "team"
        return f"""
        <g class="match" transform="translate({x:.1f} {y:.1f})">
          <rect class="match-bg" width="{box_width}" height="{box_height}" rx="7" />
          <rect class="match-head" width="{box_width}" height="18" rx="7" />
          <text class="match-number" x="8" y="13">M{match_number}</text>
          <text class="{team_a_class}" x="9" y="35">{escape(truncate(team_a))}</text>
          <line class="team-divider" x1="0" y1="40" x2="{box_width}" y2="40" />
          <text class="{team_b_class}" x="9" y="52">{escape(truncate(team_b))}</text>
        </g>
        """

    round_labels = [
        ("Round of 32", left_x[0] + box_width / 2),
        ("Round of 16", left_x[1] + box_width / 2),
        ("Quarterfinal", left_x[2] + box_width / 2),
        ("Semifinal", left_x[3] + box_width / 2),
        ("Final", center_x + box_width / 2),
        ("Semifinal", right_x[3] + box_width / 2),
        ("Quarterfinal", right_x[2] + box_width / 2),
        ("Round of 16", right_x[1] + box_width / 2),
        ("Round of 32", right_x[0] + box_width / 2),
    ]
    label_svg = "".join(
        f'<text class="round-label" x="{x:.1f}" y="84">{escape(label)}</text>' for label, x in round_labels
    )
    connector_svg = "\n".join(connector(start, end) for start, end in connector_pairs)
    match_order = [
        74,
        77,
        73,
        75,
        83,
        84,
        81,
        82,
        89,
        90,
        93,
        94,
        97,
        98,
        101,
        104,
        102,
        99,
        100,
        91,
        92,
        95,
        96,
        76,
        78,
        79,
        80,
        86,
        88,
        85,
        87,
    ]
    matches_svg = "\n".join(match_svg(number) for number in match_order if number in matches)

    html = f"""
    <style>
      body {{
        margin: 0;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #111827;
      }}
      .bracket-shell {{
        border: 1px solid #d8dee8;
        border-radius: 10px;
        background: #ffffff;
        padding: 16px 18px 12px;
      }}
      .bracket-title {{
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: flex-start;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 12px;
        margin-bottom: 8px;
      }}
      .bracket-title h3 {{
        margin: 0 0 4px 0;
        font-size: 24px;
        line-height: 1.15;
      }}
      .model-name {{
        color: #64748b;
        font-size: 13px;
        line-height: 1.35;
      }}
      .champion-pill {{
        white-space: nowrap;
        border: 1px solid #15803d;
        color: #14532d;
        background: #dcfce7;
        border-radius: 999px;
        padding: 7px 13px;
        font-weight: 750;
        font-size: 14px;
      }}
      .bracket-svg {{
        width: 100%;
        height: auto;
        display: block;
      }}
      .round-label {{
        fill: #64748b;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        text-anchor: middle;
      }}
      .connectors path {{
        fill: none;
        stroke: #cbd5e1;
        stroke-width: 1.8;
        shape-rendering: geometricPrecision;
      }}
      .match-bg {{
        fill: #f8fafc;
        stroke: #d8dee8;
        stroke-width: 1;
      }}
      .match-head {{
        fill: #ffffff;
      }}
      .match-number {{
        fill: #64748b;
        font-size: 11px;
        font-weight: 800;
      }}
      .team {{
        fill: #334155;
        font-size: 11.5px;
        font-weight: 600;
      }}
      .winner {{
        fill: #14532d;
        font-size: 11.5px;
        font-weight: 850;
      }}
      .team-divider {{
        stroke: #edf2f7;
        stroke-width: 1;
      }}
      @media (max-width: 900px) {{
        .model-name {{
          max-width: 520px;
        }}
      }}
    </style>
    <div class="bracket-shell">
      <div class="bracket-title">
        <div>
          <h3>{escape(model_title)}</h3>
          <div class="model-name">Most likely 2026 bracket</div>
        </div>
        <div class="champion-pill">Champion: {escape(champion)}</div>
      </div>
      <svg class="bracket-svg" viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMin meet" role="img" aria-label="Current temporary World Cup bracket">
        {label_svg}
        <g class="connectors">{connector_svg}</g>
        <g class="matches">{matches_svg}</g>
      </svg>
    </div>
    """
    components.html(html, height=900, scrolling=False)


def render_2022_snapshot() -> None:
    st.header("2022 World Cup Snapshot")
    st.caption(
        "A compact diagnostic view only. The dashboard is centered on comparing model approaches and 2026 outputs."
    )

    wc_rows = [
        ("Regression Model", 1.1939, 0.4844, 0.6974),
        ("Gradient Boosting Model", 1.0064, 0.5469, 0.5989),
        ("Blended Model", 0.9931, 0.5000, 0.5884),
        ("Poisson Goal Model", 1.2111, 0.5000, 0.6636),
        ("Elo Model", 1.1006, 0.5313, 0.6363),
    ]
    df = pd.DataFrame(wc_rows, columns=["Model", "Log Loss", "Accuracy", "Brier"])

    st.write(
        "The 2022 World Cup is useful as a stress test because it includes short-tournament effects, "
        "knockout incentives, favorites underperforming, and draw-heavy situations."
    )

    with st.expander("Show compact 2022 WC metrics"):
        display = df.copy()
        display["Log Loss"] = display["Log Loss"].map(lambda value: f"{value:.4f}")
        display["Accuracy"] = display["Accuracy"].map(pct)
        display["Brier"] = display["Brier"].map(lambda value: f"{value:.4f}")
        st.dataframe(display, width="stretch", hide_index=True)

    st.subheader("Draw Diagnostics")
    draw_file = safe_read_csv("results/wc_2022_draw_likelihood_ranked_with_phase.csv")
    gap_file = safe_read_csv("results/wc_2022_win_gap_ranked_with_phase.csv")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Highest draw likelihood examples**")
        if draw_file.empty:
            st.info("Draw likelihood file not found.")
        else:
            st.dataframe(draw_file.head(8), width="stretch", hide_index=True)
    with col2:
        st.markdown("**Smallest win-gap examples**")
        if gap_file.empty:
            st.info("Win-gap file not found.")
        else:
            st.dataframe(gap_file.head(8), width="stretch", hide_index=True)


def render_data_sources() -> None:
    st.header("Data Sources Used By The Dashboard")
    rows = [
        ("Today's match predictions", "results/todays_match_predictions.csv"),
        ("Live World Cup match cache", "data/live/world_cup_matches.csv"),
        ("Historical plus live results", "data/live/results_with_live_world_cup.csv"),
        ("Model summary", "Hard-coded recap from latest saved backtests"),
        ("Current 2026 bracket", "results/temp_2026_world_cup_current_model_predictions.csv"),
        ("Poisson deterministic bracket", "results/deterministic_poisson_bracket.json"),
        ("Elo sample bracket", "results/sample_bracket.json"),
        ("Elo Monte Carlo probabilities", "results/team_stage_probabilities.csv"),
        ("Poisson Monte Carlo probabilities", "results/poisson_team_stage_probabilities.csv"),
        ("2022 draw diagnostics", "results/wc_2022_draw_likelihood_ranked_with_phase.csv"),
        ("2022 win-gap diagnostics", "results/wc_2022_win_gap_ranked_with_phase.csv"),
    ]
    df = pd.DataFrame(rows, columns=["Section", "File"])
    df["Exists"] = df["File"].map(
        lambda value: "Yes" if value.startswith("Hard-coded") or (ROOT / value).exists() else "No"
    )
    st.dataframe(df, width="stretch", hide_index=True)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 8px;
        }
        .metric-card {
            border: 1px solid #d8dee8;
            border-radius: 8px;
            padding: 16px 18px;
            background: #ffffff;
            min-height: 126px;
        }
        .metric-label {
            color: #667085;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .metric-model {
            color: #111827;
            font-size: 1.1rem;
            font-weight: 700;
            margin-top: 10px;
            line-height: 1.25;
        }
        .metric-value {
            color: #334155;
            font-size: 1.15rem;
            margin-top: 8px;
            font-variant-numeric: tabular-nums;
        }
        .best-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: 1px solid #15803d;
            background: #dcfce7;
            color: #14532d;
            border-radius: 999px;
            padding: 3px 10px;
            font-weight: 750;
            font-variant-numeric: tabular-nums;
            line-height: 1.25;
        }
        .metric-plain {
            display: inline-flex;
            padding: 3px 0;
            font-variant-numeric: tabular-nums;
            line-height: 1.25;
        }
        .model-card {
            border: 1px solid #d8dee8;
            border-radius: 8px;
            background: #ffffff;
            padding: 18px;
            margin: 8px 0 14px 0;
            min-height: 220px;
        }
        .model-card-title {
            font-size: 1.15rem;
            font-weight: 750;
            color: #111827;
            line-height: 1.2;
        }
        .model-card-subtitle {
            color: #475569;
            margin-top: 4px;
            font-size: 0.95rem;
        }
        .model-card-body {
            color: #334155;
            margin-top: 14px;
            min-height: 44px;
        }
        .model-card-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin-top: 16px;
        }
        .model-card-grid div {
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 10px;
            background: #f8fafc;
        }
        .model-card-grid span {
            display: block;
            color: #64748b;
            font-size: 0.78rem;
        }
        .model-card-grid strong {
            display: block;
            color: #111827;
            font-size: 1rem;
            margin-top: 4px;
            font-variant-numeric: tabular-nums;
        }
        .model-card-foot {
            color: #64748b;
            font-size: 0.82rem;
            margin-top: 14px;
        }
        .empty-graphic {
            height: 360px;
            border: 1px dashed #cbd5e1;
            border-radius: 8px;
            background: repeating-linear-gradient(
                135deg,
                #f8fafc,
                #f8fafc 12px,
                #f1f5f9 12px,
                #f1f5f9 24px
            );
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 20px;
        }
        .empty-graphic-title {
            font-weight: 750;
            color: #334155;
        }
        .empty-graphic-line {
            width: 110px;
            height: 6px;
            border-radius: 999px;
            background: #94a3b8;
            margin: 16px 0;
        }
        .empty-graphic-copy {
            color: #64748b;
            max-width: 280px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="World Cup Prediction Dashboard",
        page_icon=None,
        layout="wide",
    )
    apply_styles()

    st.title("World Cup Prediction Dashboard")
    st.write(
        "A local dashboard for comparing the project model families, explaining how they work, "
        "and reviewing the saved 2026 prediction outputs."
    )

    tabs = st.tabs(
        [
            "Today's Matches",
            "Model Comparison",
            "Model Guide",
            "2026 Predictions",
        ]
    )
    with tabs[0]:
        render_todays_matches()
    with tabs[1]:
        render_model_comparison()
    with tabs[2]:
        render_model_descriptions()
    with tabs[3]:
        render_2026_predictions()


if __name__ == "__main__":
    main()
