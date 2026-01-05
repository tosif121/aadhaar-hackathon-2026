"""
Data Explorer Page - Simplified for real data analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append('src')

from api_client import AadhaarAPIClient

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

# Header
st.title("📊 Aadhaar Data Explorer")
st.markdown("Explore and filter Aadhaar datasets in detail")

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
    format_func=lambda x: f"{x.title()} Data"
)

# Advanced filters section
st.sidebar.markdown("#### ⚙️ Advanced Filters")

# State filter
states = [
    "All States", "Maharashtra", "Uttar Pradesh", "Tamil Nadu", "West Bengal", 
    "Karnataka", "Gujarat", "Rajasthan", "Andhra Pradesh", "Bihar",
    "Madhya Pradesh", "Telangana", "Kerala", "Odisha", "Punjab"
]
selected_state = st.sidebar.selectbox(
    "🗺️ State", 
    states,
    help="Filter data by specific state"
)

# Records to analyze with enhanced slider
records_to_analyze = st.sidebar.slider(
    "📊 Records to Analyze", 
    100, 10000, 2000, 100,
    help="Number of records to analyze"
)

# Date range section
st.sidebar.markdown("#### 📅 Date Range")
start_date = st.sidebar.date_input(
    "📅 Start Date", 
    value=pd.to_datetime("2023-01-01")
)
end_date = st.sidebar.date_input(
    "📅 End Date", 
    value=pd.to_datetime("2024-12-31")
)

# Load data button
if st.sidebar.button("📥 Load Data", type="primary"):
    with st.spinner(f"🔄 Loading {dataset_type} data..."):
        state_filter = None if selected_state == "All States" else selected_state
        
        try:
            df = client.fetch_data(
                dataset_type=dataset_type,
                limit=records_to_analyze,
                state_filter=state_filter
            )
            
            if not df.empty:
                st.session_state[f'{dataset_type}_explorer_data'] = df
                st.sidebar.success(f"✅ Loaded {len(df):,} records successfully")
            else:
                st.sidebar.warning("⚠️ No data found with current filters. Try adjusting your selection.")
                
        except Exception as e:
            st.sidebar.error(f"❌ Error loading data: {str(e)}")

# Main content
if f'{dataset_type}_explorer_data' in st.session_state:
    df = st.session_state[f'{dataset_type}_explorer_data']
    
    # Data overview
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Columns", str(len(df.columns)))
    with col3:
        if 'state' in df.columns:
            st.metric("States", str(df['state'].nunique()))
        else:
            st.metric("States", "N/A")
    with col4:
        if 'district' in df.columns:
            st.metric("Districts", str(df['district'].nunique()))
        else:
            st.metric("Districts", "N/A")
    
    # Data table with search
    st.header("📋 Data Table")
    
    # Search functionality
    search_term = st.text_input(
        "🔍 Search in data", 
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
        
        st.info(f"🔍 Found {len(filtered_df)} records matching '{search_term}'")
    else:
        filtered_df = df
    
    # Pagination
    page_size = st.selectbox(
        "📄 Records per page", 
        [10, 25, 50, 100], 
        index=1
    )
    
    total_pages = len(filtered_df) // page_size + (1 if len(filtered_df) % page_size > 0 else 0)
    
    if total_pages > 1:
        page = st.selectbox("📖 Page", range(1, total_pages + 1))
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        display_df = filtered_df.iloc[start_idx:end_idx]
        
        st.info(f"Page {page} of {total_pages}")
    else:
        display_df = filtered_df
    
    st.dataframe(display_df, use_container_width=True)
    
    # Column analysis
    st.header("📈 Column Analysis")
    
    # Select column for detailed analysis
    selected_column = st.selectbox(
        "📊 Select column for analysis", 
        df.columns
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic statistics
        st.subheader(f"📊 Statistics: {selected_column}")
        
        if df[selected_column].dtype in ['int64', 'float64']:
            # Numeric column analysis
            stats = df[selected_column].describe()
            
            # Display stats in metrics
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.metric("Count", f"{stats['count']:.0f}")
                st.metric("Mean", f"{stats['mean']:.2f}")
                st.metric("Std", f"{stats['std']:.2f}")
            with stats_col2:
                st.metric("Min", f"{stats['min']:.2f}")
                st.metric("Max", f"{stats['max']:.2f}")
                st.metric("Median", f"{stats['50%']:.2f}")
            
            # Histogram
            fig_hist = px.histogram(df, x=selected_column, 
                                  title=f"Distribution of {selected_column}",
                                  color_discrete_sequence=['#FF6B35'])
            st.plotly_chart(fig_hist, use_container_width=True)
            
        else:
            # Categorical column analysis
            value_counts = df[selected_column].value_counts()
            
            # Display categorical stats
            cat_col1, cat_col2 = st.columns(2)
            with cat_col1:
                st.metric("Unique Values", str(df[selected_column].nunique()))
                st.metric("Most Common", str(value_counts.index[0]))
            with cat_col2:
                st.metric("Total Count", str(len(df[selected_column])))
                st.metric("Mode Frequency", str(value_counts.iloc[0]))
            
            # Top values display
            st.write("**Top 10 values:**")
            top_values_df = pd.DataFrame({
                'Value': value_counts.head(10).index,
                'Count': value_counts.head(10).values,
                'Percentage': (value_counts.head(10).values / len(df) * 100).round(2)
            })
            st.dataframe(top_values_df, use_container_width=True)
    
    with col2:
        # Visualization
        st.subheader(f"📊 Visualization: {selected_column}")
        
        if df[selected_column].dtype in ['int64', 'float64']:
            # Box plot
            fig_box = px.box(df, y=selected_column, 
                           title=f"Box Plot: {selected_column}",
                           color_discrete_sequence=['#FF6B35'])
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            # Bar chart for top values
            value_counts = df[selected_column].value_counts().head(10)
            fig_bar = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"Top 10 Values: {selected_column}",
                           color=value_counts.values,
                           color_continuous_scale='Viridis')
            fig_bar.update_xaxes(tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Data quality assessment
    st.header("✅ Data Quality Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚠️ Missing Values")
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
            
            # Missing data visualization
            fig_missing = px.bar(missing_df, x='Column', y='Missing %',
                               title="Missing Data by Column",
                               color='Missing %',
                               color_continuous_scale='Reds')
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✅ Excellent Data Quality! No missing values found in the dataset.")
    
    with col2:
        st.subheader("📋 Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.dtypes.index,
            'Data Type': df.dtypes.values.astype(str)
        })
        st.dataframe(dtype_df, use_container_width=True)
        
        # Data type distribution
        type_counts = dtype_df['Data Type'].value_counts()
        fig_types = px.pie(values=type_counts.values, names=type_counts.index,
                         title="Data Type Distribution",
                         color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_types, use_container_width=True)
    
    # Export functionality
    st.header("💾 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Download CSV", type="primary"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV file",
                data=csv,
                file_name=f"aadhaar_{dataset_type}_data.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Download JSON", type="secondary"):
            json_data = filtered_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📊 Download JSON file",
                data=json_data,
                file_name=f"aadhaar_{dataset_type}_data.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("📈 Generate Report"):
            # Generate summary report
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
    # Welcome screen
    st.info("👆 Please select filters and click 'Load Data' to load the dataset.")
    
    st.markdown("""
    ### 🔍 Data Explorer Features:
    
    **Real-time Data Analysis:**
    - 📊 **Interactive Filtering**: Filter by state, date range, and record count
    - 🔍 **Advanced Search**: Search across all data columns
    - 📈 **Statistical Analysis**: Comprehensive statistics for all columns
    - 📊 **Visualizations**: Interactive charts and graphs
    - ✅ **Data Quality**: Missing value analysis and data type overview
    - 💾 **Export Options**: Download data in CSV, JSON, or generate reports
    
    ### 📋 How to Use:
    1. **Select Dataset**: Choose from Enrolment, Demographic, or Biometric data
    2. **Apply Filters**: Set state, date range, and number of records
    3. **Load Data**: Click 'Load Data' to fetch real government data
    4. **Explore**: Use search, pagination, and column analysis tools
    5. **Export**: Download your filtered data or generate reports
    
    **Note**: All data comes directly from government APIs - completely real and up-to-date.
    """)