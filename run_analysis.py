#!/usr/bin/env python3
"""
Main script to run complete Aadhaar data analysis for hackathon
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🚀 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run complete Aadhaar data analysis')
    parser.add_argument('--quick', action='store_true', help='Run quick analysis only')
    parser.add_argument('--state', type=str, help='Focus on specific state')
    parser.add_argument('--max-records', type=int, default=5000, help='Maximum records to fetch')
    
    args = parser.parse_args()
    
    print("🎯 Aadhaar Data-Driven Innovation Hackathon 2026")
    print("=" * 60)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.state:
        print(f"🗺️  Focus State: {args.state}")
    
    # Step 1: Install dependencies
    print("\n📦 Installing dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("⚠️  Some packages might already be installed. Continuing...")
    
    # Step 2: Quick analysis
    quick_cmd = f"python scripts/quick_analysis.py --records-to-analyze 1000"
    if args.state:
        quick_cmd += f" --state '{args.state}'"
    
    if not run_command(quick_cmd, "Running quick analysis"):
        print("❌ Quick analysis failed. Check your API connection.")
        return
    
    if args.quick:
        print("\n✅ Quick analysis completed!")
        return
    
    # Step 3: Fetch comprehensive data
    fetch_cmd = f"python scripts/fetch_all_data.py --max-records {args.max_records}"
    if args.state:
        fetch_cmd += f" --state '{args.state}'"
    
    if not run_command(fetch_cmd, "Fetching comprehensive datasets"):
        print("⚠️  Data fetching had issues. Continuing with available data...")
    
    # Step 4: Run Jupyter notebooks
    print("\n📓 Running analysis notebooks...")
    
    notebooks = [
        "notebooks/00_data_exploration.ipynb",
        "notebooks/01_exploratory_data_analysis.ipynb", 
        "notebooks/02_predictive_modeling.ipynb"
    ]
    
    for notebook in notebooks:
        if os.path.exists(notebook):
            cmd = f"jupyter nbconvert --to notebook --execute {notebook}"
            run_command(cmd, f"Executing {notebook}")
        else:
            print(f"⚠️  Notebook {notebook} not found")
    
    # Step 5: Generate final report
    print("\n📄 Generating final insights...")
    
    # Create a summary report
    summary_report = f"""
# Aadhaar Data Analysis Summary Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Analysis Configuration
- Focus State: {args.state or 'All States'}
- Max Records: {args.max_records:,}
- Analysis Type: {'Quick' if args.quick else 'Comprehensive'}

## Files Generated
- Data files: data/
- Visualizations: visualizations/
- Analysis notebooks: notebooks/
- Models: models/

## Next Steps for Hackathon Submission
1. Review generated visualizations in visualizations/ folder
2. Check analysis results in executed notebooks
3. Use insights to complete submission/hackathon_submission_template.md
4. Create final PDF from the completed template

## Key Analysis Areas Covered
- Univariate analysis of enrollment and update patterns
- Bivariate analysis of geographic and demographic relationships
- Trivariate analysis of complex interactions
- Predictive modeling for system optimization
- Anomaly detection for data quality insights

## Recommended Focus Areas for Submission
1. Geographic penetration gaps analysis
2. Age group adoption patterns
3. Seasonal trends in enrollments and updates
4. System load prediction and optimization
5. Policy recommendations based on data insights
"""
    
    with open("reports/analysis_summary.md", "w") as f:
        f.write(summary_report)
    
    print("\n🎉 Analysis Pipeline Completed!")
    print("=" * 60)
    print("📁 Check these folders for results:")
    print("   - data/ : Downloaded datasets")
    print("   - visualizations/ : Charts and graphs")
    print("   - notebooks/ : Executed analysis notebooks")
    print("   - reports/ : Summary reports")
    print("   - submission/ : Hackathon submission template")
    
    print("\n💡 Next Steps:")
    print("1. Review the generated visualizations and insights")
    print("2. Complete the submission template in submission/")
    print("3. Convert your submission to PDF format")
    print("4. Submit to the hackathon portal")
    
    print(f"\n✅ Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()