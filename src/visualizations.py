"""
Visualization utilities for Aadhaar data analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class AadhaarVisualizer:
    """Create visualizations for Aadhaar data insights"""
    
    def __init__(self, style='seaborn-v0_8', figsize=(12, 8)):
        plt.style.use(style)
        self.figsize = figsize
        sns.set_palette("husl")
    
    def plot_enrollment_trends(self, df: pd.DataFrame, date_col: str, value_col: str) -> go.Figure:
        """Plot enrollment trends over time"""
        fig = px.line(df, x=date_col, y=value_col, 
                     title="Aadhaar Enrollment Trends Over Time",
                     labels={date_col: "Date", value_col: "Enrollments"})
        
        fig.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Number of Enrollments",
            hovermode='x unified'
        )
        
        return fig
    
    def plot_demographic_distribution(self, df: pd.DataFrame, column: str) -> go.Figure:
        """Plot demographic distribution"""
        value_counts = df[column].value_counts()
        
        fig = px.pie(values=value_counts.values, names=value_counts.index,
                    title=f"Distribution of {column.title()}")
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig
    
    def plot_geographic_heatmap(self, df: pd.DataFrame, pincode_col: str, value_col: str) -> go.Figure:
        """Create geographic heatmap by pincode"""
        geo_data = df.groupby(pincode_col)[value_col].sum().reset_index()
        
        fig = px.density_heatmap(geo_data, x=pincode_col, y=value_col,
                               title="Geographic Distribution of Aadhaar Activities")
        
        return fig
    
    def plot_age_group_analysis(self, df: pd.DataFrame, age_col: str, value_col: str) -> go.Figure:
        """Analyze patterns by age group"""
        age_data = df.groupby(age_col)[value_col].agg(['count', 'mean', 'std']).reset_index()
        
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Count by Age Group', 'Average by Age Group',
                                         'Standard Deviation', 'Box Plot'))
        
        # Count
        fig.add_trace(go.Bar(x=age_data[age_col], y=age_data['count'], name='Count'),
                     row=1, col=1)
        
        # Average
        fig.add_trace(go.Bar(x=age_data[age_col], y=age_data['mean'], name='Average'),
                     row=1, col=2)
        
        # Standard deviation
        fig.add_trace(go.Bar(x=age_data[age_col], y=age_data['std'], name='Std Dev'),
                     row=2, col=1)
        
        # Box plot
        for age in df[age_col].unique():
            age_subset = df[df[age_col] == age][value_col]
            fig.add_trace(go.Box(y=age_subset, name=str(age)), row=2, col=2)
        
        fig.update_layout(title_text="Age Group Analysis", showlegend=False)
        
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, columns: list = None) -> go.Figure:
        """Plot correlation matrix heatmap"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        corr_matrix = df[columns].corr()
        
        fig = px.imshow(corr_matrix, 
                       title="Correlation Matrix",
                       color_continuous_scale='RdBu_r',
                       aspect="auto")
        
        fig.update_layout(
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        
        return fig
    
    def plot_time_series_decomposition(self, df: pd.DataFrame, date_col: str, value_col: str) -> go.Figure:
        """Decompose time series into trend, seasonal, and residual components"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Prepare time series data
        ts_data = df.set_index(date_col)[value_col].resample('M').sum()
        
        # Perform decomposition
        decomposition = seasonal_decompose(ts_data, model='additive', period=12)
        
        fig = make_subplots(rows=4, cols=1,
                           subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
                           vertical_spacing=0.08)
        
        # Original
        fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, name='Original'),
                     row=1, col=1)
        
        # Trend
        fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, name='Trend'),
                     row=2, col=1)
        
        # Seasonal
        fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, name='Seasonal'),
                     row=3, col=1)
        
        # Residual
        fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, name='Residual'),
                     row=4, col=1)
        
        fig.update_layout(title_text="Time Series Decomposition", showlegend=False, height=800)
        
        return fig
    
    def plot_anomaly_detection(self, df: pd.DataFrame, columns: list, anomaly_indices: list) -> go.Figure:
        """Visualize detected anomalies"""
        fig = make_subplots(rows=len(columns), cols=1,
                           subplot_titles=[f"Anomalies in {col}" for col in columns])
        
        for i, col in enumerate(columns):
            # Normal points
            normal_mask = ~df.index.isin(anomaly_indices)
            fig.add_trace(go.Scatter(x=df.index[normal_mask], y=df[col][normal_mask],
                                   mode='markers', name=f'Normal {col}',
                                   marker=dict(color='blue', size=4)),
                         row=i+1, col=1)
            
            # Anomalous points
            anomaly_mask = df.index.isin(anomaly_indices)
            fig.add_trace(go.Scatter(x=df.index[anomaly_mask], y=df[col][anomaly_mask],
                                   mode='markers', name=f'Anomaly {col}',
                                   marker=dict(color='red', size=8, symbol='x')),
                         row=i+1, col=1)
        
        fig.update_layout(title_text="Anomaly Detection Results", height=200*len(columns))
        
        return fig
    
    def create_dashboard(self, df: pd.DataFrame, config: dict) -> go.Figure:
        """Create comprehensive dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Enrollment Trends', 'Geographic Distribution',
                          'Age Group Analysis', 'Update Types',
                          'Seasonal Patterns', 'Anomalies'),
            specs=[[{"secondary_y": True}, {"type": "geo"}],
                   [{"colspan": 2}, None],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        # Add various plots based on configuration
        # This would be customized based on specific data structure
        
        fig.update_layout(
            title_text="Aadhaar Data Analytics Dashboard",
            height=1000,
            showlegend=True
        )
        
        return fig
    
    def save_plot(self, fig: go.Figure, filename: str, format: str = 'html'):
        """Save plot to file"""
        if format == 'html':
            fig.write_html(f"visualizations/{filename}.html")
        elif format == 'png':
            fig.write_image(f"visualizations/{filename}.png")
        elif format == 'pdf':
            fig.write_image(f"visualizations/{filename}.pdf")