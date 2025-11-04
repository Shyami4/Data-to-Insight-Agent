# chart_style.py
import plotly.express as px

def apply_common_style(
    fig,
    *,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    legend_title: str = "",
    height: int = 500,
    bargap: float = 0.25,
    show_legend: bool = True,
    dark: bool = True,
):
    """Applies layout + trace-safe styling to both bar and line charts."""
    template = "plotly_dark" if dark else "plotly_white"

    # --- Layout (safe for all traces) ---
    fig.update_layout(
        template=template,
        title={"text": title, "x": 0.01, "xanchor": "left"},
        height=height,
        bargap=bargap,                 # ignored for non-bar plots
        showlegend=show_legend,
        legend=dict(
            title=legend_title,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        font=dict(size=14),
        uniformtext_minsize=12,
        uniformtext_mode="hide",
    )

    # --- Trace-specific styling ---
    for tr in fig.data:
        if tr.type == "bar":
            # labels outside; slight outline; bar width
            tr.update(
                texttemplate="%{text:,}" if getattr(tr, "text", None) is not None else None,
                textposition="outside",
                marker_line_color="rgba(0,0,0,0.25)",
                marker_line_width=1,
                cliponaxis=False,
            )
            # bar 'width' is valid; set via figure-wide update_traces is OK, but set per trace to be safe
            if hasattr(tr, "width"):
                tr.update(width=0.6)
            tr.update(hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>")
        elif tr.type in ("scatter",):  # line charts
            tr.update(
                mode="lines+markers" if getattr(tr, "mode", "") in ("", "lines") else tr.mode,
                line={"width": 2},
                marker={"size": 6},
                hovertemplate="<b>%{x}</b><br>%{y:,}<extra></extra>",
            )
        else:
            # leave other trace types as-is
            pass

    return fig


def apply_common_bar_style(*args, **kwargs):
    """Backward-compatible wrapper; uses apply_common_style."""
    return apply_common_style(*args, **kwargs)


def color_discrete_map_from_categories(categories, palette):
    if not categories:
        return None
    colors = (palette * ((len(categories) // len(palette)) + 1))[: len(categories)]
    return {cat: col for cat, col in zip(categories, colors)}
