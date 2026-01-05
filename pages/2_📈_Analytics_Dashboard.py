"""
Analytics Dashboard - Advanced analytics and insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
sys.path.append('src')

from api_client import AadhaarAPIClient
from analysis import AadhaarAnalyzer

st.set_page_config(page_title="Analytics Dashboard", page_icon="📈", layout="wide")

st.title("📈 Aadhaar Analytics Dashboard")
st.markdown("Advanced analytics and insights from Aadhaar data")

# Initialize clients
@st.cache_resource
def get_clients():
    return AadhaarAPIClient(), AadhaarAnalyzer()

client, analyzer = get_clients()

# Sidebar configuration
st.sidebar.header("⚙️ Dashboard Configuration")

# Analysis type selection
analysis_modes = [
    "Geographic Analysis",
    "Temporal Trends", 
    "Age Group Analysis",
    "Comparative Analysis",
    "Correlation Analysis",
    "Anomaly Detection"
]

selected_mode = st.sidebar.selectbox("Select Analysis Mode", analysis_modes)

# Data loading section
st.sidebar.header("📊 Data Selection")

# Load real data from API
@st.cache_data
def load_real_data():
    """Load real data from Aadhaar APIs for analytics"""
    try:
        enrolment_df = client.fetch_data('enrolment', limit=2000)
        demographic_df = client.fetch_data('demographic', limit=2000) 
        biometric_df = client.fetch_data('biometric', limit=2000)
        return enrolment_df, demographic_df, biometric_df
    except:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

if st.sidebar.button("🔄 Load Analytics Data", type="primary"):
    with st.spinner("Loading real Aadhaar data for analytics..."):
        enrolment_df, demographic_df, biometric_df = load_real_data()
        
        st.session_state['analytics_enrolment'] = enrolment_df
        st.session_state['analytics_demographic'] = demographic_df
        st.session_state['analytics_biometric'] = biometric_df
        
        if not enrolment_df.empty or not demographic_df.empty or not biometric_df.empty:
            st.sidebar.success("✅ Data loaded successfully!")
        else:
            st.sidebar.error("❌ Failed to load data")

# Main dashboard content
if ('analytics_enrolment' in st.session_state or 
    'analytics_demographic' in st.session_state or 
    'analytics_biometric' in st.session_state):
    
    # Get loaded data
    enrolment_df = st.session_state.get('analytics_enrolment', pd.DataFrame())
    demographic_df = st.session_state.get('analytics_demographic', pd.DataFrame())
    biometric_df = st.session_state.get('analytics_biometric', pd.DataFrame())
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_enrolments = len(enrolment_df)
        st.metric("📊 Enrolment Records", f"{total_enrolments:,}")
    
    with col2:
        total_demographic = len(demographic_df)
        st.metric("👥 Demographic Updates", f"{total_demographic:,}")
    
    with col3:
        total_biometric = len(biometric_df)
        st.metric("👆 Biometric Updates", f"{total_biometric:,}")
    
    with col4:
        # Calculate update ratio
        if total_enrolments > 0:
            update_ratio = ((total_demographic + total_biometric) / total_enrolments) * 100
            st.metric("🔄 Update Ratio", f"{update_ratio:.1f}%")
        else:
            st.metric("🔄 Update Ratio", "N/A")
    
    # Analysis based on selected mode
    if selected_mode == "Geographic Analysis":
        st.header("🗺️ Geographic Analysis")
        
        # State-wise analysis
        col1, col2 = st.columns(2)
        
        with col1:
            if not enrolment_df.empty and 'state' in enrolment_df.columns:
                st.subheader("📊 Enrolment by State")
                
                state_enrolments = enrolment_df['state'].value_counts().head(10)
                fig_states = px.bar(
                    x=state_enrolments.values,
                    y=state_enrolments.index,
                    orientation='h',
                    title="Top 10 States - Enrolments",
                    labels={'x': 'Number of Records', 'y': 'State'}
                )
                fig_states.update_layout(height=400)
                st.plotly_chart(fig_states, use_container_width=True)
            else:
                st.info("Enrolment data not available")
        
        with col2:
            if not demographic_df.empty and 'state' in demographic_df.columns:
                st.subheader("👥 Demographic Updates by State")
                
                state_updates = demographic_df['state'].value_counts().head(10)
                fig_updates = px.bar(
                    x=state_updates.values,
                    y=state_updates.index,
                    orientation='h',
                    title="Top 10 States - Demographic Updates",
                    labels={'x': 'Number of Records', 'y': 'State'},
                    color=state_updates.values,
                    color_continuous_scale='Viridis'
                )
                fig_updates.update_layout(height=400)
                st.plotly_chart(fig_updates, use_container_width=True)
            else:
                st.info("Demographic data not available")
        
        # District-level analysis
        st.subheader("🏘️ District-Level Insights")
        
        if not enrolment_df.empty and 'district' in enrolment_df.columns:
            # Select state for district analysis
            available_states = enrolment_df['state'].unique()
            selected_state = st.selectbox("Select State for District Analysis", available_states)
            
            state_data = enrolment_df[enrolment_df['state'] == selected_state]
            district_counts = state_data['district'].value_counts().head(15)
            
            fig_districts = px.bar(
                x=district_counts.index,
                y=district_counts.values,
                title=f"District-wise Enrolments in {selected_state}",
                labels={'x': 'District', 'y': 'Number of Records'}
            )
            fig_districts.update_xaxes(tickangle=45)
            st.plotly_chart(fig_districts, use_container_width=True)
    
    elif selected_mode == "Temporal Trends":
        st.header("📅 Temporal Trends Analysis")
        
        # Process date columns
        datasets_with_dates = []
        
        for df, name in [(enrolment_df, 'Enrolment'), (demographic_df, 'Demographic'), (biometric_df, 'Biometric')]:
            if not df.empty and 'date' in df.columns:
                df_copy = df.copy()
                df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
                df_copy = df_copy.dropna(subset=['date'])
                if not df_copy.empty:
                    datasets_with_dates.append((df_copy, name))
        
        if datasets_with_dates:
            # Monthly trends
            st.subheader("📈 Monthly Trends")
            
            fig_trends = go.Figure()
            
            for df, name in datasets_with_dates:
                monthly_counts = df.groupby(df['date'].dt.to_period('M')).size()
                fig_trends.add_trace(go.Scatter(
                    x=monthly_counts.index.astype(str),
                    y=monthly_counts.values,
                    mode='lines+markers',
                    name=name,
                    line=dict(width=3)
                ))
            
            fig_trends.update_layout(
                title="Monthly Activity Trends",
                xaxis_title="Month",
                yaxis_title="Number of Records",
                hovermode='x unified'
            )
            st.plotly_chart(fig_trends, use_container_width=True)
            
            # Seasonal analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌍 Seasonal Patterns")
                
                if datasets_with_dates:
                    df, name = datasets_with_dates[0]  # Use first available dataset
                    df['month'] = df['date'].dt.month
                    monthly_avg = df.groupby('month').size()
                    
                    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    
                    fig_seasonal = px.bar(
                        x=months,
                        y=monthly_avg.values,
                        title=f"Seasonal Pattern - {name}",
                        labels={'x': 'Month', 'y': 'Average Records'}
                    )
                    st.plotly_chart(fig_seasonal, use_container_width=True)
            
            with col2:
                st.subheader("📊 Day of Week Analysis")
                
                if datasets_with_dates:
                    df, name = datasets_with_dates[0]
                    df['day_of_week'] = df['date'].dt.day_name()
                    
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    day_counts = df['day_of_week'].value_counts().reindex(day_order, fill_value=0)
                    
                    fig_dow = px.bar(
                        x=day_counts.index,
                        y=day_counts.values,
                        title=f"Day of Week Pattern - {name}",
                        labels={'x': 'Day of Week', 'y': 'Number of Records'}
                    )
                    st.plotly_chart(fig_dow, use_container_width=True)
        else:
            st.info("No date information available in the loaded datasets")
    
    elif selected_mode == "Age Group Analysis":
        st.header("👥 Age Group Analysis")
        
        # Find age-related columns
        age_columns = {}
        
        for df, name in [(enrolment_df, 'Enrolment'), (demographic_df, 'Demographic'), (biometric_df, 'Biometric')]:
            if not df.empty:
                age_cols = [col for col in df.columns if 'age' in col.lower() and df[col].dtype in ['int64', 'float64']]
                if age_cols:
                    age_columns[name] = (df, age_cols)
        
        if age_columns:
            # Age group distribution
            st.subheader("📊 Age Group Distribution")
            
            fig_age = make_subplots(
                rows=1, cols=len(age_columns),
                subplot_titles=list(age_columns.keys()),
                specs=[[{"type": "bar"}] * len(age_columns)]
            )
            
            colors = ['#FF6B35', '#004E89', '#1A936F']
            
            for i, (name, (df, age_cols)) in enumerate(age_columns.items()):
                age_totals = df[age_cols].sum()
                
                fig_age.add_trace(
                    go.Bar(
                        x=age_cols,
                        y=age_totals.values,
                        name=name,
                        marker_color=colors[i % len(colors)],
                        showlegend=False
                    ),
                    row=1, col=i+1
                )
            
            fig_age.update_layout(title_text="Age Group Distribution Across Datasets", height=400)
            st.plotly_chart(fig_age, use_container_width=True)
            
            # Age group comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔍 Age Group Insights")
                
                # Calculate insights
                insights = []
                
                for name, (df, age_cols) in age_columns.items():
                    age_totals = df[age_cols].sum()
                    max_age_group = age_totals.idxmax()
                    max_value = age_totals.max()
                    
                    insights.append(f"**{name}**: Highest activity in {max_age_group} ({max_value:,} records)")
                
                for insight in insights:
                    st.markdown(insight)
            
            with col2:
                st.subheader("📈 Age Group Trends")
                
                # Create pie chart for first dataset
                if age_columns:
                    first_dataset = list(age_columns.values())[0]
                    df, age_cols = first_dataset
                    age_totals = df[age_cols].sum()
                    
                    fig_pie = px.pie(
                        values=age_totals.values,
                        names=age_cols,
                        title="Age Group Distribution"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No age group data available in the loaded datasets")
    
    elif selected_mode == "Comparative Analysis":
        st.header("⚖️ Comparative Analysis")
        
        # Compare datasets
        st.subheader("📊 Dataset Comparison")
        
        comparison_data = {
            'Dataset': ['Enrolment', 'Demographic Updates', 'Biometric Updates'],
            'Records': [len(enrolment_df), len(demographic_df), len(biometric_df)],
            'States': [
                enrolment_df['state'].nunique() if not enrolment_df.empty and 'state' in enrolment_df.columns else 0,
                demographic_df['state'].nunique() if not demographic_df.empty and 'state' in demographic_df.columns else 0,
                biometric_df['state'].nunique() if not biometric_df.empty and 'state' in biometric_df.columns else 0
            ],
            'Districts': [
                enrolment_df['district'].nunique() if not enrolment_df.empty and 'district' in enrolment_df.columns else 0,
                demographic_df['district'].nunique() if not demographic_df.empty and 'district' in demographic_df.columns else 0,
                biometric_df['district'].nunique() if not biometric_df.empty and 'district' in biometric_df.columns else 0
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(comparison_df, use_container_width=True)
        
        with col2:
            fig_comparison = px.bar(
                comparison_df,
                x='Dataset',
                y='Records',
                title="Records by Dataset Type",
                color='Records',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # State-wise comparison
        st.subheader("🗺️ State-wise Comparison")
        
        # Find common states
        common_states = set()
        state_data = {}
        
        for df, name in [(enrolment_df, 'Enrolment'), (demographic_df, 'Demographic'), (biometric_df, 'Biometric')]:
            if not df.empty and 'state' in df.columns:
                states = set(df['state'].unique())
                if not common_states:
                    common_states = states
                else:
                    common_states = common_states.intersection(states)
                
                state_counts = df['state'].value_counts()
                state_data[name] = state_counts
        
        if common_states and len(common_states) > 0:
            # Select states for comparison
            selected_states = st.multiselect(
                "Select states for comparison",
                list(common_states),
                default=list(common_states)[:5]
            )
            
            if selected_states:
                # Create comparison chart
                comparison_chart_data = []
                
                for state in selected_states:
                    for dataset_name, state_counts in state_data.items():
                        if state in state_counts.index:
                            comparison_chart_data.append({
                                'State': state,
                                'Dataset': dataset_name,
                                'Count': state_counts[state]
                            })
                
                if comparison_chart_data:
                    comp_df = pd.DataFrame(comparison_chart_data)
                    
                    fig_state_comp = px.bar(
                        comp_df,
                        x='State',
                        y='Count',
                        color='Dataset',
                        title="State-wise Dataset Comparison",
                        barmode='group'
                    )
                    st.plotly_chart(fig_state_comp, use_container_width=True)
        else:
            st.info("No common states found across datasets for comparison")
    
    elif selected_mode == "Correlation Analysis":
        st.header("🔗 Correlation Analysis")
        
        # Select dataset for correlation analysis
        available_datasets = []
        if not enrolment_df.empty:
            available_datasets.append(('Enrolment', enrolment_df))
        if not demographic_df.empty:
            available_datasets.append(('Demographic', demographic_df))
        if not biometric_df.empty:
            available_datasets.append(('Biometric', biometric_df))
        
        if available_datasets:
            dataset_names = [name for name, _ in available_datasets]
            selected_dataset_name = st.selectbox("Select Dataset for Correlation Analysis", dataset_names)
            
            # Get selected dataset
            selected_df = next(df for name, df in available_datasets if name == selected_dataset_name)
            
            # Get numeric columns
            numeric_columns = selected_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) > 1:
                # Calculate correlation matrix
                corr_matrix = selected_df[numeric_columns].corr()
                
                # Correlation heatmap
                fig_corr = px.imshow(
                    corr_matrix,
                    title=f"Correlation Matrix - {selected_dataset_name}",
                    color_continuous_scale='RdBu_r',
                    aspect='auto'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Strong correlations
                st.subheader("🔍 Strong Correlations")
                
                # Find strong correlations (> 0.7 or < -0.7)
                strong_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            strong_corr.append({
                                'Variable 1': corr_matrix.columns[i],
                                'Variable 2': corr_matrix.columns[j],
                                'Correlation': corr_val,
                                'Strength': 'Strong Positive' if corr_val > 0 else 'Strong Negative'
                            })
                
                if strong_corr:
                    strong_corr_df = pd.DataFrame(strong_corr)
                    st.dataframe(strong_corr_df, use_container_width=True)
                else:
                    st.info("No strong correlations (>0.7 or <-0.7) found")
            else:
                st.info("Not enough numeric columns for correlation analysis")
        else:
            st.info("No datasets available for correlation analysis")
    
    elif selected_mode == "Anomaly Detection":
        st.header("🚨 Anomaly Detection")
        
        # Select dataset for anomaly detection
        available_datasets = []
        if not enrolment_df.empty:
            available_datasets.append(('Enrolment', enrolment_df))
        if not demographic_df.empty:
            available_datasets.append(('Demographic', demographic_df))
        if not biometric_df.empty:
            available_datasets.append(('Biometric', biometric_df))
        
        if available_datasets:
            dataset_names = [name for name, _ in available_datasets]
            selected_dataset_name = st.selectbox("Select Dataset for Anomaly Detection", dataset_names)
            
            # Get selected dataset
            selected_df = next(df for name, df in available_datasets if name == selected_dataset_name)
            
            # Perform anomaly detection
            numeric_columns = selected_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_columns:
                anomalies = analyzer.detect_anomalies(selected_df, numeric_columns)
                
                # Anomaly metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    z_anomalies = len(anomalies.get('z_score_anomalies', []))
                    st.metric("Z-Score Anomalies", z_anomalies)
                
                with col2:
                    iqr_anomalies = len(anomalies.get('iqr_anomalies', []))
                    st.metric("IQR Anomalies", iqr_anomalies)
                
                with col3:
                    anomaly_rate = anomalies.get('anomaly_percentage', 0)
                    st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
                
                # Visualize anomalies
                if numeric_columns:
                    selected_col = st.selectbox("Select Column for Anomaly Visualization", numeric_columns)
                    
                    fig_anomaly = go.Figure()
                    
                    # Normal points
                    normal_indices = [i for i in selected_df.index if i not in anomalies.get('z_score_anomalies', [])]
                    if normal_indices:
                        fig_anomaly.add_trace(go.Scatter(
                            x=normal_indices,
                            y=selected_df.loc[normal_indices, selected_col],
                            mode='markers',
                            name='Normal',
                            marker=dict(color='blue', size=4, opacity=0.6)
                        ))
                    
                    # Anomalous points
                    anomaly_indices = anomalies.get('z_score_anomalies', [])
                    if anomaly_indices:
                        fig_anomaly.add_trace(go.Scatter(
                            x=anomaly_indices,
                            y=selected_df.loc[anomaly_indices, selected_col],
                            mode='markers',
                            name='Anomalies',
                            marker=dict(color='red', size=8, symbol='x')
                        ))
                    
                    fig_anomaly.update_layout(
                        title=f"Anomaly Detection: {selected_col}",
                        xaxis_title="Index",
                        yaxis_title=selected_col
                    )
                    st.plotly_chart(fig_anomaly, use_container_width=True)
                
                # Anomaly details
                if anomaly_indices:
                    st.subheader("🔍 Anomaly Details")
                    anomaly_data = selected_df.loc[anomaly_indices]
                    st.dataframe(anomaly_data, use_container_width=True)
            else:
                st.info("No numeric columns available for anomaly detection")
        else:
            st.info("No datasets available for anomaly detection")

else:
    st.info("👆 Please load analytics data from the sidebar to begin analysis.")
    
    st.header("📊 Analytics Dashboard")
    
    st.info("👆 **Load Real Data**: Click the 'Load Analytics Data' button in the sidebar to fetch real Aadhaar data for comprehensive analytics.")
    
    st.markdown("""
    ### 🔍 Available Analytics Features:
    
    **Once you load real data, you'll be able to:**
    - 📈 **Trend Analysis**: Visualize enrollment and update patterns over time
    - 🗺️ **Geographic Analysis**: Compare performance across states and districts  
    - 👥 **Demographic Insights**: Analyze age group distributions and patterns
    - 🔗 **Cross-Dataset Comparisons**: Compare enrollment vs update ratios
    - 📊 **Statistical Analysis**: Correlation analysis and pattern detection
    - 🎯 **Anomaly Detection**: Identify unusual patterns in the data
    
    ### 📋 Analysis Process:
    1. **Load Data**: Use the sidebar button to fetch real Aadhaar datasets
    2. **Explore**: Browse through different analysis sections
    3. **Visualize**: Generate interactive charts and graphs
    4. **Insights**: Extract meaningful patterns and trends
    5. **Export**: Save your findings for hackathon submission
    
    **Note**: All analysis uses real government data from data.gov.in APIs - no synthetic data.
    """)