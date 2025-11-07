# chart_style.py
from typing import Sequence, Dict, Optional
from plotly.colors import qualitative as q

# Nice default qualitative palette
DEFAULT_PALETTE = q.Set2


def color_discrete_map_from_categories(
    categories: Sequence[str],
    palette: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """
    Return a Plotly color_discrete_map for the given category order.
    Ensures we have enough distinct colors by repeating the palette if needed.
    """
    cats = list(dict.fromkeys([str(c) for c in categories]))  # keep order, unique
    pal = list(palette or DEFAULT_PALETTE)
    if not pal:
        pal = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]  # fallback
    # repeat palette to cover all categories
    color_cycle = (pal * ((len(cats) // len(pal)) + 1))[: len(cats)]
    return {cat: color_cycle[i] for i, cat in enumerate(cats)}


def apply_common_bar_style(
    fig,
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
):
    """
    Apply consistent styling to Plotly figures (esp. bar charts).
    Safe for any figure type; bar-specific tweaks only applied to bar traces.
    """
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1.0,
            bgcolor="rgba(0,0,0,0)"
        ),
    )
    # Bar-only tweaks
    for tr in fig.data:
        if getattr(tr, "type", "") == "bar":
            tr.update(
                texttemplate="%{value:,.0f}",
                textposition="outside",
                cliponaxis=False,   # allow labels outside
            )

    fig.update_xaxes(title=xaxis_title, showgrid=False, zeroline=False)
    fig.update_yaxes(title=yaxis_title, gridcolor="rgba(200,200,200,.35)", zeroline=False)
    return fig