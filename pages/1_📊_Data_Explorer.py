"""
Data Explorer Page - Enhanced with modern UI/UX and dark/light mode
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append('src')

from api_client import AadhaarAPIClient
from ui_components import UIThemeManager, ModernComponents, create_theme_toggle, apply_plotly_theme

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

# Initialize theme and UI components
theme_manager = create_theme_toggle()
theme_manager.apply_theme()
theme_config = theme_manager.get_theme_config()
ui_components = ModernComponents(theme_manager)

# Header
st.markdown("""
<div class="main-title">📊 Aadhaar Data Explorer</div>
<div class="sub-title">Explore and filter Aadhaar datasets in detail</div>
""", unsafe_allow_html=True)

# Initialize client
@st.cache_resource
def get_client():
    return AadhaarAPIClient()

client = get_client()

# Sidebar filters
st.sidebar.markdown("### 🔍 Data Filters")

# Dataset selection
dataset_type = st.sidebar.selectbox(
    "📊 Select Dataset",
    ["enrolment", "demographic", "biometric"],
    format_func=lambda x: f"{icons.get_icon('data', 16)} {x.title()} Data"
)

# Advanced filters section
ui_components.create_section_header("Advanced Filters", "Customize your data selection", "settings")

# State filter with enhanced styling
states = [
    "All States", "Maharashtra", "Uttar Pradesh", "Tamil Nadu", "West Bengal", 
    "Karnataka", "Gujarat", "Rajasthan", "Andhra Pradesh", "Bihar",
    "Madhya Pradesh", "Telangana", "Kerala", "Odisha", "Punjab"
]
selected_state = st.sidebar.selectbox(
    f"{icons.get_icon('location', 16)} State", 
    states,
    help="Filter data by specific state"
)

# Sample size with enhanced slider
sample_size = st.sidebar.slider(
    f"{icons.get_icon('chart', 16)} Sample Size", 
    100, 10000, 2000, 100,
    help="Number of records to analyze"
)

# Date range section
ui_components.create_section_header("Date Range", "Filter by time period", "calendar")
start_date = st.sidebar.date_input(
    f"{icons.get_icon('calendar', 16)} Start Date", 
    value=pd.to_datetime("2023-01-01")
)
end_date = st.sidebar.date_input(
    f"{icons.get_icon('calendar', 16)} End Date", 
    value=pd.to_datetime("2024-12-31")
)

# Enhanced load data button
if st.sidebar.button(f"{icons.get_icon('refresh', 16)} Refresh Data", type="primary"):
    with st.spinner(f"🔄 Loading {dataset_type} data..."):
        state_filter = None if selected_state == "All States" else selected_state
        
        try:
            df = client.fetch_data(
                dataset_type=dataset_type,
                limit=sample_size,
                state_filter=state_filter
            )
            
            if not df.empty:
                st.session_state[f'{dataset_type}_explorer_data'] = df
                ui_components.create_info_box(
                    "Success!", 
                    f"Loaded {len(df):,} records successfully",
                    "success",
                    "check"
                )
            else:
                ui_components.create_info_box(
                    "No Data Found", 
                    "No data found with current filters. Try adjusting your selection.",
                    "warning",
                    "warning"
                )
                
        except Exception as e:
            ui_components.create_info_box(
                "Error Loading Data", 
                f"Error loading data: {str(e)}",
                "error",
                "error"
            )

# Main content with enhanced UI
if f'{dataset_type}_explorer_data' in st.session_state:
    df = st.session_state[f'{dataset_type}_explorer_data']
    
    # Enhanced data overview
    ui_components.create_section_header("Data Overview", "Comprehensive dataset summary", "dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ui_components.create_metric_card("Total Records", f"{len(df):,}", icon="database")
    with col2:
        ui_components.create_metric_card("Columns", str(len(df.columns)), icon="table")
    with col3:
        if 'state' in df.columns:
            ui_components.create_metric_card("States", str(df['state'].nunique()), icon="location")
        else:
            ui_components.create_metric_card("States", "N/A", icon="location")
    with col4:
        if 'district' in df.columns:
            ui_components.create_metric_card("Districts", str(df['district'].nunique()), icon="map")
        else:
            ui_components.create_metric_card("Districts", "N/A", icon="map")
    
    # Enhanced data table with search
    ui_components.create_section_header("Data Table", "Interactive data exploration", "table")
    
    # Search functionality with enhanced UI
    search_term = st.text_input(
        f"{icons.get_icon('search', 16)} Search in data", 
        placeholder="Enter search term...",
        help="Search across all text columns"
    )
    
    if search_term:
        # Search across all string columns
        string_columns = df.select_dtypes(include=['object']).columns
        mask = df[string_columns].astype(str).apply(
            lambda x: x.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        filtered_df = df[mask]
        
        ui_components.create_info_box(
            "Search Results", 
            f"Found {len(filtered_df)} records matching '{search_term}'",
            "info",
            "search"
        )
    else:
        filtered_df = df
    
    # Enhanced pagination
    page_size = st.selectbox(
        f"{icons.get_icon('table', 16)} Records per page", 
        [10, 25, 50, 100], 
        index=1
    )
    
    total_pages = len(filtered_df) // page_size + (1 if len(filtered_df) % page_size > 0 else 0)
    
    if total_pages > 1:
        page = st.selectbox(f"{icons.get_icon('chart', 16)} Page", range(1, total_pages + 1))
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        display_df = filtered_df.iloc[start_idx:end_idx]
        
        # Progress indicator for pagination
        ui_components.create_progress_indicator(page, total_pages, f"Page {page} of {total_pages}")
    else:
        display_df = filtered_df
    
    st.dataframe(display_df, use_container_width=True)
    
    # Enhanced column analysis
    ui_components.create_section_header("Column Analysis", "Detailed statistical analysis", "analytics")
    
    # Select column for detailed analysis
    selected_column = st.selectbox(
        f"{icons.get_icon('chart', 16)} Select column for analysis", 
        df.columns
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced basic statistics
        ui_components.create_section_header(f"Statistics: {selected_column}", "Descriptive statistics", "chart")
        
        if df[selected_column].dtype in ['int64', 'float64']:
            # Numeric column analysis
            stats = df[selected_column].describe()
            
            # Display stats in enhanced cards
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                ui_components.create_metric_card("Count", f"{stats['count']:.0f}", icon="chart")
                ui_components.create_metric_card("Mean", f"{stats['mean']:.2f}", icon="chart")
                ui_components.create_metric_card("Std", f"{stats['std']:.2f}", icon="chart")
            with stats_col2:
                ui_components.create_metric_card("Min", f"{stats['min']:.2f}", icon="chart")
                ui_components.create_metric_card("Max", f"{stats['max']:.2f}", icon="chart")
                ui_components.create_metric_card("Median", f"{stats['50%']:.2f}", icon="chart")
            
            # Enhanced histogram
            fig_hist = px.histogram(df, x=selected_column, 
                                  title=f"Distribution of {selected_column}",
                                  color_discrete_sequence=['#FF6B35'])
            fig_hist = apply_plotly_theme(fig_hist, theme_config)
            st.plotly_chart(fig_hist, use_container_width=True)
            
        else:
            # Categorical column analysis
            value_counts = df[selected_column].value_counts()
            
            # Display categorical stats
            cat_col1, cat_col2 = st.columns(2)
            with cat_col1:
                ui_components.create_metric_card("Unique Values", str(df[selected_column].nunique()), icon="table")
                ui_components.create_metric_card("Most Common", str(value_counts.index[0]), icon="trophy")
            with cat_col2:
                ui_components.create_metric_card("Total Count", str(len(df[selected_column])), icon="chart")
                ui_components.create_metric_card("Mode Frequency", str(value_counts.iloc[0]), icon="chart")
            
            # Top values display
            st.write("**Top 10 values:**")
            top_values_df = pd.DataFrame({
                'Value': value_counts.head(10).index,
                'Count': value_counts.head(10).values,
                'Percentage': (value_counts.head(10).values / len(df) * 100).round(2)
            })
            st.dataframe(top_values_df, use_container_width=True)
    
    with col2:
        # Enhanced visualization
        ui_components.create_section_header(f"Visualization: {selected_column}", "Interactive charts", "chart")
        
        if df[selected_column].dtype in ['int64', 'float64']:
            # Enhanced box plot
            fig_box = px.box(df, y=selected_column, 
                           title=f"Box Plot: {selected_column}",
                           color_discrete_sequence=['#FF6B35'])
            fig_box = apply_plotly_theme(fig_box, theme_config)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            # Enhanced bar chart for top values
            value_counts = df[selected_column].value_counts().head(10)
            fig_bar = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"Top 10 Values: {selected_column}",
                           color=value_counts.values,
                           color_continuous_scale='Viridis')
            fig_bar.update_xaxes(tickangle=45)
            fig_bar = apply_plotly_theme(fig_bar, theme_config)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Enhanced data quality assessment
    ui_components.create_section_header("Data Quality Assessment", "Comprehensive quality analysis", "check")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ui_components.create_section_header("Missing Values", "Data completeness analysis", "warning")
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True)
            
            # Enhanced missing data visualization
            fig_missing = px.bar(missing_df, x='Column', y='Missing %',
                               title="Missing Data by Column",
                               color='Missing %',
                               color_continuous_scale='Reds')
            fig_missing = apply_plotly_theme(fig_missing, theme_config)
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            ui_components.create_info_box(
                "Excellent Data Quality!", 
                "No missing values found in the dataset.",
                "success",
                "check"
            )
    
    with col2:
        ui_components.create_section_header("Data Types", "Column type distribution", "table")
        dtype_df = pd.DataFrame({
            'Column': df.dtypes.index,
            'Data Type': df.dtypes.values.astype(str)
        })
        st.dataframe(dtype_df, use_container_width=True)
        
        # Enhanced data type distribution
        type_counts = dtype_df['Data Type'].value_counts()
        fig_types = px.pie(values=type_counts.values, names=type_counts.index,
                         title="Data Type Distribution",
                         color_discrete_sequence=px.colors.qualitative.Set3)
        fig_types = apply_plotly_theme(fig_types, theme_config)
        st.plotly_chart(fig_types, use_container_width=True)
    
    # Enhanced export functionality
    ui_components.create_section_header("Export Data", "Download your analysis results", "download")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(f"{icons.get_icon('download', 16)} Download CSV", type="primary"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV file",
                data=csv,
                file_name=f"aadhaar_{dataset_type}_data.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button(f"{icons.get_icon('download', 16)} Download JSON", type="secondary"):
            json_data = filtered_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📊 Download JSON file",
                data=json_data,
                file_name=f"aadhaar_{dataset_type}_data.json",
                mime="application/json"
            )
    
    with col3:
        if st.button(f"{icons.get_icon('report', 16)} Generate Report"):
            # Generate enhanced summary report
            report = f"""
# Aadhaar {dataset_type.title()} Data Report

## Summary
- **Total Records**: {len(filtered_df):,}
- **Columns**: {len(filtered_df.columns)}
- **Date Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Filter Applied**: {selected_state if selected_state != "All States" else "None"}

## Data Quality
- **Missing Values**: {filtered_df.isnull().sum().sum()}
- **Completeness**: {((1 - filtered_df.isnull().sum().sum() / (len(filtered_df) * len(filtered_df.columns))) * 100):.2f}%

## Key Statistics
{filtered_df.describe().to_string() if len(filtered_df.select_dtypes(include=['number']).columns) > 0 else 'No numeric columns available'}

## Column Information
{dtype_df.to_string(index=False)}

---
Generated by Aadhaar Analytics Dashboard
"""
            
            st.download_button(
                label="📈 Download Report",
                data=report,
                file_name=f"aadhaar_{dataset_type}_report.md",
                mime="text/markdown"
            )

else:
    # Enhanced welcome screen
    ui_components.create_info_box(
        "Welcome to Data Explorer!", 
        "Please select filters and click 'Refresh Data' to load the dataset.",
        "info",
        "info"
    )
    
    # Enhanced sample data structure
    ui_components.create_section_header("Expected Data Structure", "Preview of dataset schemas", "table")
    
    sample_structures = {
        "enrolment": {
            "date": "2024-01-01",
            "state": "Maharashtra", 
            "district": "Mumbai",
            "pincode": "400001",
            "age_0_5": 1250,
            "age_5_17": 3400,
            "age_18_40": 8900,
            "age_40_60": 4500,
            "age_60_plus": 2100
        },
        "demographic": {
            "date": "2024-01-01",
            "state": "Maharashtra",
            "district": "Mumbai", 
            "pincode": "400001",
            "demo_age_5_17": 450,
            "demo_age_18_40": 1200,
            "demo_age_40_60": 800,
            "demo_age_60_plus": 300
        },
        "biometric": {
            "date": "2024-01-01",
            "state": "Maharashtra",
            "district": "Mumbai",
            "pincode": "400001", 
            "bio_age_5_17": 120,
            "bio_age_18_40": 340,
            "bio_age_40_60": 280,
            "bio_age_60_plus": 95
        }
    }
    
    ui_components.create_section_header(f"Sample {dataset_type.title()} Data Structure", "Expected column format", "data")
    sample_df = pd.DataFrame([sample_structures[dataset_type]])
    st.dataframe(sample_df, use_container_width=True)