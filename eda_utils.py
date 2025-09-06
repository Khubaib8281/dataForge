import json, hashlib
from functools import lru_cache
import pandas as pd
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# def data_info(df):
#     info = {
#         "shape": df.shape,    
#         "columns": df.columns.tolist(),
#         "dtypes": df.dtypes.to_dict(),
#         "missing_values": df.isnull().sum().to_dict(),
#     }   
#     return info


def data_info(df):
    if df is None or df.empty:
        return {"error": "Dataset is empty or not loaded properly."}

    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "summary_stats": df.describe(include="all", percentiles=[.25, .5, .75]).transpose()
    }

    # Build professional human-readable summary text for LLMs
    summary_parts = []
    summary_parts.append(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

    # Column types
    summary_parts.append("\nColumn types:")
    for col, dtype in info["dtypes"].items():
        summary_parts.append(f"- {col}: {dtype}")

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        summary_parts.append("\nMissing values detected in:")
        for col, val in missing.items():
            if val > 0:
                summary_parts.append(f"- {col}: {val}")
    else:
        summary_parts.append("\nNo missing values detected.")

    info["summary_text"] = "\n".join(summary_parts)
    return info
    


def clean_data(df):
    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].fillna("Unknown")

    return df
   
def generate_basic_visuals(df):
    plots = []
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols[:2]:
        fig, ax = plt.subplots()
        df[col].hist(ax=ax)
        ax.set_title(f"Distribution of {col}")   
        plots.append(fig)
    return plots

 
def make_light(fig, default_color="#636efa"):
    """
    Convert a Plotly figure to a light theme.
    Ensures bars, lines, and markers are visible, 
    and axes/titles are black for PDF/HTML export.
    default_color is applied if a bar or line loses color.
    """

    fig.update_layout(
    title=dict(text=fig.layout.title.text, font=dict(color="black")),
    font=dict(color="black"),          # axis labels, legend
    paper_bgcolor="white",
    plot_bgcolor="white"
)

# Axis ticks and lines
    fig.update_xaxes(title_font=dict(color="black"), tickfont=dict(color="black"), showline=True, linecolor="black")
    fig.update_yaxes(title_font=dict(color="black"), tickfont=dict(color="black"), showline=True, linecolor="black")

    colors = px.colors.qualitative.Plotly
    for i, trace in enumerate(fig.data):
        if trace.type in ["box", "violin", "histogram"]:  
            trace.marker.color = colors[i % len(colors)]
          
    return fig



def generate_visuals_from_suggestions(df, suggestions, max_points=2000):
    """
    Generates Plotly figures from selected visualization suggestions.
    Downsamples large datasets for scatter/pairplot/violin plots.
    Returns a list of interactive Plotly figures.
    """
    import plotly.express as px
    figures = []

    for vis in suggestions:
        vis_type = vis.get("type")
        fig = None

        # Histogram
        if vis_type == "histogram":
            col = vis.get("x")
            if col in df.columns:
                fig = px.histogram(df, x=col, nbins=10, title=f"Histogram: {col}", template="plotly_white")
                if df[col].max() > 10000:
                    fig.update_xaxes(dtick=10000)

        # Scatter
        elif vis_type == "scatter":
            x_col, y_col = vis.get("x"), vis.get("y")
            if x_col in df.columns and y_col in df.columns:
                data = df[[x_col, y_col]]   
                if len(data) > max_points:
                    data = data.sample(max_points, random_state=42)
                fig = px.scatter(data, x=x_col, y=y_col, title=f"Scatter: {x_col} vs {y_col}", template="plotly_white")

        # Bar
        elif vis_type == "bar":
            x_col, y_col = vis.get("x"), vis.get("y")
            if x_col in df.columns and y_col in df.columns:
                if df[x_col].nunique() > 20:
                    top_cats = df[x_col].value_counts().nlargest(20).index
                    data = df[df[x_col].isin(top_cats)]
                else:
                    data = df
                fig = px.bar(data, x=x_col, y=y_col, title=f"Bar: {x_col} vs {y_col}", template="plotly_dark")

        # Box
        elif vis_type == "box":
            x_col, y_col = vis.get("x"), vis.get("y")
            if x_col in df.columns and y_col in df.columns:
                data = df
                fig = px.box(data, x=x_col, y=y_col, title=f"Box: {x_col} vs {y_col}", template="plotly_dark")

        # Violin
        elif vis_type == "violin":
            x_col, y_col = vis.get("x"), vis.get("y")
            if x_col in df.columns and y_col in df.columns:
                data = df
                if len(df) > max_points:
                    data = df.sample(max_points, random_state=42)
                fig = px.violin(data, x=x_col, y=y_col, box=True, points="all", title=f"Violin: {x_col} vs {y_col}", template="plotly_dark")

        # Heatmap
        elif vis_type == "heatmap":
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Heatmap", template="plotly_white")

        # Pairplot
        elif vis_type == "pairplot":
            cols = [c for c in vis.get("x", []) if c in df.columns]
            if len(cols) > 1:
                data = df[cols]
                if len(data) > max_points:
                    data = data.sample(max_points, random_state=42)
                fig = px.scatter_matrix(data, title="Pairplot", template="plotly_white")

        # Apply light/dark theme fixes
        if fig:
            fig = make_light(fig)
            figures.append(fig)

    return figures
    


# ---------- perf helpers ----------

def _df_signature(df: pd.DataFrame) -> str:
    """Stable hash for the data used in plots. Uses dtypes + first 10 rows only."""
    head = df.head(10).to_dict(orient="list")
    payload = {
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "head": head,
        "nrows": int(df.shape[0]),
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()

def _vis_key(vis: dict) -> str:
    """Stable key per visualization suggestion."""
    return hashlib.md5(json.dumps(vis, sort_keys=True).encode()).hexdigest()

# ---------- styling helpers ----------

def _enforce_readable_layout(fig, dark=False):
    if dark:
        fig.update_layout(template="plotly_dark")
        text_color = "white"
        line_color = "white"
    else:
        fig.update_layout(template="plotly_white")
        text_color = "black"
        line_color = "black"

    fig.update_layout(
        paper_bgcolor="black" if dark else "white",
        plot_bgcolor="black" if dark else "white",
        title=dict(text=(fig.layout.title.text or ""), font=dict(color=text_color, size=18)),
        font=dict(color=text_color),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    fig.update_xaxes(title_font=dict(color=text_color), tickfont=dict(color=text_color),
                     showline=True, linecolor=line_color, gridcolor="gray" if dark else "lightgray")
    fig.update_yaxes(title_font=dict(color=text_color), tickfont=dict(color=text_color),
                     showline=True, linecolor=line_color, gridcolor="gray" if dark else "lightgray")
    return fig

def _strongen_bar_family_colors(fig):
    # Force categorical/fill-like traces to strong palette (prevents washed/white bars)
    palette = px.colors.qualitative.Plotly
    for i, tr in enumerate(fig.data):
        if tr.type in ("bar", "box", "violin", "histogram"):
            if getattr(tr, "marker", None) is None or getattr(tr.marker, "color", None) in (None, "white", "#ffffff"):
                tr.marker = dict(color=palette[i % len(palette)], opacity=1)
    return fig

# ---------- single-figure generator (cached) ----------

def _generate_single_figure(df: pd.DataFrame, vis: dict, max_points=1000):
    """Return a single Plotly fig for a suggestion dict."""
    vtype = vis.get("type")
    fig = None

    if vtype == "histogram":
        col = vis.get("x")
        if col in df.columns:
            data = df[col].dropna()

            # Choose number of bins
            nbins = 20

            # Pre-compute bins and counts in pandas
            counts, bins = pd.cut(data, bins=nbins, retbins=True)
            bin_counts = counts.value_counts().sort_index()

            # Create readable bin labels for x-axis
            bin_labels = [f"{bins[i]:.2f} - {bins[i+1]:.2f}" for i in range(len(bins)-1)]

            # Build bar plot (instead of px.histogram)
            fig = px.bar(
                x=bin_labels,
                y=bin_counts.values,
                labels={"x": col, "y": "Count"},
                title=f"Histogram: {col}",
                template="plotly_white"
            )

            # Rotate x labels for readability
            fig.update_xaxes(tickangle=45)

            # Improve layout
            fig = _enforce_readable_layout(fig, dark=False)
    
    elif vtype == "scatter":
        x, y = vis.get("x"), vis.get("y")
        if x in df.columns and y in df.columns:
            data = df[[x, y]].dropna()
            if len(data) > max_points:
                data = data.sample(max_points, random_state=42)
            # no trendline by default (heavy); you can add a UI toggle if you want
            fig = px.scatter(data, x=x, y=y, title=f"Scatter: {x} vs {y}", template="plotly_white")
            fig = _enforce_readable_layout(fig, dark=False)
   
    elif vtype == "bar":
        x, y = vis.get("x"), vis.get("y")
        if x in df.columns and y in df.columns:
            data = df[[x, y]].dropna()
            # keep top 20 cats
            if data[x].nunique() > 20:
                top = data[x].value_counts().nlargest(20).index
                data = data[data[x].isin(top)]
            fig = px.bar(data, x=x, y=y, title=f"Bar: {x} vs {y}", template="plotly_dark")
            fig = _strongen_bar_family_colors(fig)
            fig = _enforce_readable_layout(fig, dark=True)

    elif vtype == "box":
        x, y = vis.get("x"), vis.get("y")
        if x in df.columns and y in df.columns:
            data = df[[x, y]].dropna()
            if data[x].nunique() > 20:
                top = data[x].value_counts().nlargest(20).index
                data = data[data[x].isin(top)]
            fig = px.box(data, x=x, y=y, title=f"Box: {x} vs {y}", template="plotly_dark")
            fig = _strongen_bar_family_colors(fig)
            fig = _enforce_readable_layout(fig, dark=True)

    elif vtype == "violin":
        x, y = vis.get("x"), vis.get("y")
        if x in df.columns and y in df.columns:
            data = df[[x, y]].dropna()
            if len(data) > max_points:
                data = data.sample(max_points, random_state=42)
            fig = px.violin(data, x=x, y=y, box=True, points="all",
                            title=f"Violin: {x} vs {y}", template="plotly_dark")
            fig = _strongen_bar_family_colors(fig)
            fig = _enforce_readable_layout(fig, dark=True)

    elif vtype == "heatmap":
        nums = df.select_dtypes(include="number")
        if nums.shape[1] > 1:
            # if too many numeric cols, keep top 15 by variance (faster, clearer)
            if nums.shape[1] > 15:
                var = nums.var().sort_values(ascending=False).head(15).index
                nums = nums[var]
            corr = nums.corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                            title="Correlation Heatmap", template="plotly_white")
            fig = _enforce_readable_layout(fig, dark=False)

    elif vtype == "pairplot":
        cols = [c for c in (vis.get("x") or []) if c in df.columns]
        if len(cols) > 1:
            data = df[cols].dropna()
            if len(data) > 1000:
                data = data.sample(1000, random_state=42)
            fig = px.scatter_matrix(data, dimensions=cols, title="Pairplot", template="plotly_white")
            fig = _enforce_readable_layout(fig, dark=False)

    return fig   

# Cache single figures per df signature + vis key (fast re-use)
def generate_fig_cached(df: pd.DataFrame, vis: dict, max_points=2000):
    sig = _df_signature(df)
    key = _vis_key(vis)
    cache_key = f"{sig}:{key}:{max_points}"
    # simple in-memory cache (Streamlit reruns clear funcs, so also store in session_state)
    store = st.session_state.setdefault("_fig_cache", {})
    if cache_key not in store:
        store[cache_key] = _generate_single_figure(df, vis, max_points=max_points)
    return store[cache_key]
            

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def generate_static_matplotlib(df: pd.DataFrame, vis: dict):
    """
    Generate a single static Matplotlib/Seaborn plot from a visualization suggestion.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset
    vis : dict
        Visualization instruction, e.g.
        {"type":"histogram", "x":"price"}
        {"type":"scatter", "x":"mileage", "y":"price"}

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object
    """

    if not isinstance(vis, dict):
        return None   

    vis_type = vis.get("type", "histogram").lower()
    fig, ax = plt.subplots(figsize=(6,4))

    # Histogram
    if vis_type == "histogram":
        col = vis.get("x")
        if col in df.columns:
            sns.histplot(df[col].dropna(), bins=30, kde=True, ax=ax, color="skyblue")
            ax.set_title(f"Histogram of {col}", fontsize=12, color="black")
            ax.set_xlabel(col, color="black")
            ax.set_ylabel("Count", color="black")

            if df[col].max() > 10000:
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))

    # Scatter plot
    elif vis_type == "scatter":
        x_col = vis.get("x")
        y_col = vis.get("y")
        if x_col in df.columns and y_col in df.columns:
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, s=40, color="teal", alpha=0.6)
            ax.set_title(f"Scatter: {x_col} vs {y_col}", fontsize=12, color="black")
            ax.set_xlabel(x_col, color="black")
            ax.set_ylabel(y_col, color="black")

    # Bar plot
    elif vis_type == "bar":
        x_col = vis.get("x")
        y_col = vis.get("y")
        if x_col in df.columns and y_col in df.columns:
            # If too many categories, pick top 20
            if df[x_col].nunique() > 20:
                top_cats = df[x_col].value_counts().nlargest(20).index
                data = df[df[x_col].isin(top_cats)]
            else:
                data = df
    
            sns.barplot(data=data, x=x_col, y=y_col, ax=ax, palette="viridis", hue = x_col, legend=False)
            ax.set_title(f"Bar: {x_col} vs {y_col}", fontsize=12, color="black")
            ax.set_xlabel(x_col, color="black")
            ax.set_ylabel(y_col, color="black")
            ax.tick_params(axis='x', rotation=45)

    # Box plot
    elif vis_type == "box":
        x_col = vis.get("x")
        y_col = vis.get("y")
        if x_col in df.columns and y_col in df.columns:
            if df[x_col].nunique() > 20:
                top_cats = df[x_col].value_counts().nlargest(20).index
                data = df[df[x_col].isin(top_cats)]
            sns.boxplot(data=df, x=x_col, y=y_col, ax=ax, palette="Set2", hue = x_col, legend=False)
            ax.set_title(f"Box: {x_col} vs {y_col}", fontsize=12, color="black")
            ax.set_xlabel(x_col, color="black")
            ax.set_ylabel(y_col, color="black")
            ax.tick_params(axis='x', rotation=45)    

    # Violin plot
    elif vis_type == "violin":
        x_col = vis.get("x")
        y_col = vis.get("y")
        if x_col in df.columns and y_col in df.columns:
            sns.violinplot(data=df, x=x_col, y=y_col, ax=ax, inner="box", palette="Pastel1", hue = x_col, legend=False)
            ax.set_title(f"Violin: {x_col} vs {y_col}", fontsize=12, color="black")
            ax.set_xlabel(x_col, color="black")   
            ax.set_ylabel(y_col, color="black")
            ax.tick_params(axis='x', rotation=45)

    # Heatmap (correlation)   
    elif vis_type == "heatmap":
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap", fontsize=12, color="black")

    # Pairplot
    elif vis_type == "pairplot":
        cols = vis.get("x", [])
        cols = [c for c in cols if c in df.columns]
        if len(cols) > 1:
            # Pairplot creates its own figure, not using ax
            fig = sns.pairplot(df[cols], diag_kind="kde", corner=True)
            fig.fig.suptitle("Pairplot", fontsize=12, color="black")
            return fig.fig  # return underlying matplotlib Figure

    plt.tight_layout()
    return fig
