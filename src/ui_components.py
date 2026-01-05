"""
Simple UI Components for Aadhaar Analytics Dashboard
"""

import plotly.graph_objects as go
import plotly.express as px

def apply_plotly_theme(fig, theme_config=None):
    """Apply simple theme to Plotly figures"""
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title_font_size=16,
        showlegend=True,
        font=dict(size=12),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig