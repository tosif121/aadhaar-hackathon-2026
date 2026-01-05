#!/usr/bin/env python3
"""
Quick analysis script for Aadhaar data insights
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from api_client import AadhaarAPIClient
from analysis import AadhaarAnalyzer
from visualizations import AadhaarVisualizer
import argparse

def quick_insights(df, dataset_name):
    """Generate quick insights from a dataset"""
    
    print(f"\n{'='*50}")
    print(f"QUICK INSIGHTS: {dataset_name.upper()}")
    print(f"{'='*50}")
    
    if df.empty:
        print("No data available for analysis")
        return
    
    # Basic statistics
    print(f"📊 Dataset Size: {len(df):,} records")
    print(f"📅 Date Range: {df['date'].min() if 'date' in df.columns else 'N/A'} to {df['date'].max() if 'date' in df.columns else 'N/A'}")
    
    # Geographic coverage
    if 'state' in df.columns:
        print(f"🗺️  States Covered: {df['state'].nunique()}")
        print(f"🏘️  Districts Covered: {df['district'].nunique() if 'district' in df.columns else 'N/A'}")
        
        # Top states
        top_states = df['state'].value_counts().head(5)
        print(f"\n🔝 Top 5 States by Records:")
        for i, (state, count) in enumerate(top_states.items(), 1):
            print(f"   {i}. {state}: {count:,} records")
    
    # Age group analysis (if available)
    age_columns = [col for col in df.columns if 'age' in col.lower()]
    if age_columns:
        print(f"\n👥 Age Groups Available: {len(age_columns)}")
        print(f"   Columns: {age_columns}")
        
        # Calculate total across age groups for numeric columns
        numeric_age_cols = [col for col in age_columns if df[col].dtype in ['int64', 'float64']]
        if numeric_age_cols:
            age_totals = df[numeric_age_cols].sum()
            print(f"\n📈 Age Group Totals:")
            for col, total in age_totals.items():
                print(f"   {col}: {total:,}")
    
    # Temporal patterns (if date column exists)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        print(f"\n📅 Temporal Distribution:")
        yearly_counts = df['year'].value_counts().sort_index()
        for year, count in yearly_counts.items():
            print(f"   {year}: {count:,} records")
        
        # Monthly patterns
        monthly_avg = df.groupby('month').size().mean()
        peak_month = df['month'].value_counts().index[0]
        print(f"\n📊 Peak Month: {peak_month} (Month {peak_month})")
        print(f"📊 Average Records per Month: {monthly_avg:.0f}")
    
    # Data quality insights
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        print(f"\n⚠️  Data Quality Issues:")
        for col, missing in missing_data.items():
            if missing > 0:
                pct = (missing / len(df)) * 100
                print(f"   {col}: {missing:,} missing ({pct:.1f}%)")
    else:
        print(f"\n✅ Data Quality: No missing values detected")

def create_quick_visualizations(enrolment_df, demographic_df, biometric_df):
    """Create quick visualization dashboard"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Aadhaar Data Quick Analysis Dashboard', fontsize=16)
    
    # Row 1: State distributions
    datasets = [
        (enrolment_df, 'Enrolment', axes[0, 0]),
        (demographic_df, 'Demographic Updates', axes[0, 1]),
        (biometric_df, 'Biometric Updates', axes[0, 2])
    ]
    
    for df, title, ax in datasets:
        if not df.empty and 'state' in df.columns:
            top_states = df['state'].value_counts().head(8)
            ax.barh(range(len(top_states)), top_states.values)
            ax.set_yticks(range(len(top_states)))
            ax.set_yticklabels(top_states.index)
            ax.set_title(f'Top States - {title}')
            ax.set_xlabel('Number of Records')
        else:
            ax.text(0.5, 0.5, f'{title}\nNo Data Available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
    
    # Row 2: Temporal and age group analysis
    # Temporal trends
    if not enrolment_df.empty and 'date' in enrolment_df.columns:
        enrolment_df['date'] = pd.to_datetime(enrolment_df['date'], errors='coerce')
        monthly_trends = enrolment_df.groupby(enrolment_df['date'].dt.to_period('M')).size()
        axes[1, 0].plot(monthly_trends.index.astype(str), monthly_trends.values)
        axes[1, 0].set_title('Enrolment Trends Over Time')
        axes[1, 0].tick_params(axis='x', rotation=45)
    else:
        axes[1, 0].text(0.5, 0.5, 'Temporal Trends\nNo Date Data', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # Age group comparison (if available)
    age_data = []
    for df, name in [(enrolment_df, 'Enrolment'), (demographic_df, 'Demographic'), (biometric_df, 'Biometric')]:
        if not df.empty:
            age_cols = [col for col in df.columns if 'age' in col.lower() and df[col].dtype in ['int64', 'float64']]
            if age_cols:
                total = df[age_cols].sum().sum()
                age_data.append((name, total))
    
    if age_data:
        names, totals = zip(*age_data)
        axes[1, 1].bar(names, totals)
        axes[1, 1].set_title('Total Activity by Dataset')
        axes[1, 1].set_ylabel('Total Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'Age Group Analysis\nNo Numeric Data', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    # Data availability summary
    data_summary = {
        'Enrolment': len(enrolment_df),
        'Demographic': len(demographic_df),
        'Biometric': len(biometric_df)
    }
    
    axes[1, 2].pie(data_summary.values(), labels=data_summary.keys(), autopct='%1.1f%%')
    axes[1, 2].set_title('Data Distribution by Dataset')
    
    plt.tight_layout()
    plt.savefig('visualizations/quick_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Quick analysis of Aadhaar data')
    parser.add_argument('--records-to-analyze', type=int, default=1000, help='Number of records for quick analysis')
    parser.add_argument('--state', type=str, help='Focus on specific state')
    
    args = parser.parse_args()
    
    # Initialize client
    client = AadhaarAPIClient()
    
    print("🚀 Starting Quick Aadhaar Data Analysis...")
    print(f"📊 Records to Analyze: {args.records_to_analyze:,} records per dataset")
    if args.state:
        print(f"🗺️  Focus State: {args.state}")
    
    # Fetch real data
    print("\n📥 Fetching data...")
    enrolment_df = client.fetch_data('enrolment', limit=args.records_to_analyze, state_filter=args.state)
    demographic_df = client.fetch_data('demographic', limit=args.records_to_analyze, state_filter=args.state)
    biometric_df = client.fetch_data('biometric', limit=args.records_to_analyze, state_filter=args.state)
    
    # Generate insights
    quick_insights(enrolment_df, "Enrolment")
    quick_insights(demographic_df, "Demographic Updates")
    quick_insights(biometric_df, "Biometric Updates")
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    create_quick_visualizations(enrolment_df, demographic_df, biometric_df)
    
    # Combined insights
    print(f"\n{'='*50}")
    print("🎯 COMBINED INSIGHTS")
    print(f"{'='*50}")
    
    total_records = len(enrolment_df) + len(demographic_df) + len(biometric_df)
    print(f"📊 Total Records Analyzed: {total_records:,}")
    
    # Find common states
    states_enrolment = set(enrolment_df['state'].unique() if 'state' in enrolment_df.columns else [])
    states_demographic = set(demographic_df['state'].unique() if 'state' in demographic_df.columns else [])
    states_biometric = set(biometric_df['state'].unique() if 'state' in biometric_df.columns else [])
    
    common_states = states_enrolment & states_demographic & states_biometric
    if common_states:
        print(f"🗺️  States with All Data Types: {len(common_states)}")
        print(f"   Examples: {list(common_states)[:5]}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS FOR HACKATHON:")
    print("1. Focus on states with complete data across all datasets")
    print("2. Analyze temporal trends to identify seasonal patterns")
    print("3. Compare enrollment vs update ratios for insights")
    print("4. Use age group data for demographic analysis")
    print("5. Identify geographic gaps for policy recommendations")
    
    print(f"\n✅ Quick analysis completed! Check visualizations/ folder for charts.")

if __name__ == "__main__":
    main()