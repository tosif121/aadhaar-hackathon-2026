#!/usr/bin/env python3
"""
Script to fetch all Aadhaar datasets for comprehensive analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from api_client import AadhaarAPIClient
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Fetch Aadhaar datasets')
    parser.add_argument('--dataset', choices=['enrolment', 'demographic', 'biometric', 'all'], 
                       default='all', help='Dataset to fetch')
    parser.add_argument('--state', type=str, help='Filter by state name')
    parser.add_argument('--max-records', type=int, help='Maximum records to fetch')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for API calls')
    
    args = parser.parse_args()
    
    # Initialize API client
    client = AadhaarAPIClient()
    
    datasets_to_fetch = ['enrolment', 'demographic', 'biometric'] if args.dataset == 'all' else [args.dataset]
    
    for dataset_type in datasets_to_fetch:
        logger.info(f"Starting to fetch {dataset_type} data...")
        
        try:
            # Fetch data
            df = client.fetch_all_data(
                dataset_type=dataset_type,
                batch_size=args.batch_size,
                max_records=args.max_records,
                state_filter=args.state
            )
            
            if not df.empty:
                # Generate filename
                filename = f"aadhaar_{dataset_type}_data"
                if args.state:
                    filename += f"_{args.state.replace(' ', '_').lower()}"
                
                # Save data
                client.save_data(df, filename, 'csv')
                client.save_data(df, filename, 'json')
                
                logger.info(f"Successfully fetched and saved {len(df)} records for {dataset_type}")
                
                # Print summary
                print(f"\n=== {dataset_type.upper()} DATA SUMMARY ===")
                print(f"Records: {len(df)}")
                print(f"Columns: {list(df.columns)}")
                print(f"Date range: {df['date'].min() if 'date' in df.columns else 'N/A'} to {df['date'].max() if 'date' in df.columns else 'N/A'}")
                print(f"States: {df['state'].nunique() if 'state' in df.columns else 'N/A'}")
                print(f"Districts: {df['district'].nunique() if 'district' in df.columns else 'N/A'}")
                
            else:
                logger.warning(f"No data fetched for {dataset_type}")
                
        except Exception as e:
            logger.error(f"Error fetching {dataset_type} data: {e}")
    
    logger.info("Data fetching completed!")

if __name__ == "__main__":
    main()