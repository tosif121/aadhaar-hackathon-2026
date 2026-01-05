"""
API client for fetching Aadhaar datasets from data.gov.in
"""

import requests
import pandas as pd
import numpy as np
import json
from typing import Optional, Dict, List
import time
import logging

class AadhaarAPIClient:
    """Client for accessing Aadhaar datasets via data.gov.in API"""
    
    def __init__(self, api_key: str = "579b464db66ec23bdd0000019c6c0867b1854ffd43489eb616c6282f"):
        self.api_key = api_key
        self.base_url = "https://api.data.gov.in/resource"
        self.logger = logging.getLogger(__name__)
        
        # Dataset resource IDs
        self.datasets = {
            'enrolment': 'ecd49b12-3084-4521-8f7e-ca8bf72069ba',
            'demographic': '19eac040-0b94-49fa-b239-4f2fd8677d53',
            'biometric': '65454dab-1517-40a3-ac1d-47d4dfe6891c'
        }
    
    def fetch_data(self, dataset_type: str, format: str = 'json', 
                   limit: int = 1000, offset: int = 0,
                   state_filter: Optional[str] = None,
                   district_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data from specified Aadhaar dataset
        
        Args:
            dataset_type: 'enrolment', 'demographic', or 'biometric'
            format: 'json', 'csv', or 'xml'
            limit: Maximum number of records to return
            offset: Number of records to skip
            state_filter: Filter by state name
            district_filter: Filter by district name
        """
        
        if dataset_type not in self.datasets:
            raise ValueError(f"Invalid dataset type. Choose from: {list(self.datasets.keys())}")
        
        resource_id = self.datasets[dataset_type]
        url = f"{self.base_url}/{resource_id}"
        
        params = {
            'api-key': self.api_key,
            'format': format,
            'limit': limit,
            'offset': offset
        }
        
        # Add filters if provided
        if state_filter:
            params['filters[state]'] = state_filter
        if district_filter:
            params['filters[district]'] = district_filter
        
        try:
            self.logger.info(f"Fetching {dataset_type} data with limit={limit}, offset={offset}")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            if format == 'json':
                data = response.json()
                if 'records' in data:
                    return pd.DataFrame(data['records'])
                else:
                    self.logger.warning("No 'records' field in response")
                    return pd.DataFrame()
            
            elif format == 'csv':
                return pd.read_csv(pd.StringIO(response.text))
            
            else:
                self.logger.error(f"Format {format} not supported for DataFrame conversion")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error processing response: {e}")
            return pd.DataFrame()
    
    def fetch_all_data(self, dataset_type: str, batch_size: int = 1000,
                       max_records: Optional[int] = None,
                       state_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch all available data for a dataset with pagination
        
        Args:
            dataset_type: 'enrolment', 'demographic', or 'biometric'
            batch_size: Number of records per API call
            max_records: Maximum total records to fetch (None for all)
            state_filter: Filter by state name
        """
        
        all_data = []
        offset = 0
        total_fetched = 0
        
        while True:
            # Determine current batch size
            current_limit = batch_size
            if max_records and (total_fetched + batch_size > max_records):
                current_limit = max_records - total_fetched
            
            # Fetch batch
            batch_df = self.fetch_data(
                dataset_type=dataset_type,
                limit=current_limit,
                offset=offset,
                state_filter=state_filter
            )
            
            # Check if we got data
            if batch_df.empty:
                self.logger.info(f"No more data available. Total fetched: {total_fetched}")
                break
            
            all_data.append(batch_df)
            total_fetched += len(batch_df)
            offset += batch_size
            
            self.logger.info(f"Fetched {len(batch_df)} records. Total: {total_fetched}")
            
            # Check stopping conditions
            if max_records and total_fetched >= max_records:
                break
            
            if len(batch_df) < batch_size:
                # Likely reached end of data
                break
            
            # Rate limiting - be nice to the API
            time.sleep(0.5)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"Successfully fetched {len(combined_df)} total records")
            return combined_df
        else:
            return pd.DataFrame()
    
    def get_available_filters(self, dataset_type: str, sample_size: int = 100) -> Dict:
        """Get available filter options from actual data"""
        
        sample_df = self.fetch_data(dataset_type, limit=sample_size)
        
        filters = {
            'columns': list(sample_df.columns) if not sample_df.empty else [],
            'states': [],
            'districts': [],
            'date_range': None,
            'numeric_columns': [],
            'categorical_columns': []
        }
        
        if not sample_df.empty:
            # Get unique states
            if 'state' in sample_df.columns:
                filters['states'] = sorted(sample_df['state'].dropna().unique().tolist())
            
            # Get unique districts
            if 'district' in sample_df.columns:
                filters['districts'] = sorted(sample_df['district'].dropna().unique().tolist())
            
            # Get date range
            date_columns = [col for col in sample_df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                for date_col in date_columns:
                    try:
                        sample_df[date_col] = pd.to_datetime(sample_df[date_col], errors='coerce')
                        if not sample_df[date_col].isna().all():
                            filters['date_range'] = {
                                'column': date_col,
                                'min_date': sample_df[date_col].min(),
                                'max_date': sample_df[date_col].max()
                            }
                            break
                    except:
                        continue
            
            # Categorize columns
            filters['numeric_columns'] = sample_df.select_dtypes(include=[np.number]).columns.tolist()
            filters['categorical_columns'] = sample_df.select_dtypes(include=['object']).columns.tolist()
        
        return filters
    
    def get_data_summary(self, dataset_type: str) -> Dict:
        """Get summary information about a dataset"""
        
        sample_df = self.fetch_data(dataset_type, limit=100)
        
        return {
            'dataset_type': dataset_type,
            'sample_size': len(sample_df),
            'columns': list(sample_df.columns) if not sample_df.empty else [],
            'available_states': self.get_available_states(dataset_type)[:10],  # First 10 states
            'data_types': sample_df.dtypes.to_dict() if not sample_df.empty else {}
        }
    
    def save_data(self, df: pd.DataFrame, filename: str, format: str = 'csv'):
        """Save DataFrame to file"""
        
        filepath = f"data/{filename}"
        
        if format == 'csv':
            df.to_csv(f"{filepath}.csv", index=False)
        elif format == 'json':
            df.to_json(f"{filepath}.json", orient='records', indent=2)
        elif format == 'parquet':
            df.to_parquet(f"{filepath}.parquet", index=False)
        
        self.logger.info(f"Data saved to {filepath}.{format}")

# Convenience functions for quick data access
def get_enrolment_data(limit: int = 1000, state: Optional[str] = None) -> pd.DataFrame:
    """Quick function to get enrolment data"""
    client = AadhaarAPIClient()
    return client.fetch_data('enrolment', limit=limit, state_filter=state)

def get_demographic_data(limit: int = 1000, state: Optional[str] = None) -> pd.DataFrame:
    """Quick function to get demographic update data"""
    client = AadhaarAPIClient()
    return client.fetch_data('demographic', limit=limit, state_filter=state)

def get_biometric_data(limit: int = 1000, state: Optional[str] = None) -> pd.DataFrame:
    """Quick function to get biometric update data"""
    client = AadhaarAPIClient()
    return client.fetch_data('biometric', limit=limit, state_filter=state)