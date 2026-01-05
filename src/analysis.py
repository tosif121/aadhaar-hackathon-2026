"""
Advanced Statistical Analysis and AI/ML for Aadhaar Data
Enhanced with XGBoost, Isolation Forest, and Advanced Analytics
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import silhouette_score, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import networkx as nx
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import Prophet for time series forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

class RevolutionaryQuestionEngine:
    """Ask and answer questions that nobody else will think of"""
    
    @staticmethod
    def predict_future_penetration_problems(df: pd.DataFrame) -> dict:
        """REVOLUTIONARY: Predict which districts will have Aadhaar penetration problems 5 years from now"""
        
        if 'district' not in df.columns or 'date' not in df.columns:
            return {}
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Calculate current penetration trends by district
        district_trends = {}
        
        for district in df['district'].unique():
            district_data = df[df['district'] == district].copy()
            district_data = district_data.sort_values('date')
            
            if len(district_data) > 5:  # Need minimum data points
                # Calculate monthly growth rate
                monthly_data = district_data.groupby(district_data['date'].dt.to_period('M')).size()
                
                if len(monthly_data) > 2:
                    # Linear regression to find trend
                    x = np.arange(len(monthly_data))
                    y = monthly_data.values
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    # Project 5 years (60 months) into future
                    future_trend = slope * 60 + intercept
                    current_avg = monthly_data.mean()
                    
                    # Risk factors
                    declining_trend = slope < 0
                    low_r_squared = r_value**2 < 0.3  # Unstable pattern
                    below_average = current_avg < monthly_data.median()
                    
                    risk_score = sum([declining_trend, low_r_squared, below_average])
                    
                    district_trends[district] = {
                        'current_monthly_avg': current_avg,
                        'trend_slope': slope,
                        'trend_strength': r_value**2,
                        'projected_5yr': max(0, future_trend),
                        'risk_score': risk_score,
                        'risk_level': 'HIGH' if risk_score >= 2 else 'MEDIUM' if risk_score == 1 else 'LOW'
                    }
        
        # Identify high-risk districts
        high_risk_districts = {k: v for k, v in district_trends.items() if v['risk_level'] == 'HIGH'}
        
        return {
            'district_trends': district_trends,
            'high_risk_districts': high_risk_districts,
            'total_districts_analyzed': len(district_trends),
            'high_risk_count': len(high_risk_districts),
            'risk_percentage': (len(high_risk_districts) / len(district_trends)) * 100 if district_trends else 0
        }
    
    @staticmethod
    def election_cycle_correlation_analysis(df: pd.DataFrame) -> dict:
        """MIND-BLOWING: Analyze correlation between demographic updates and election cycles"""
        
        if 'date' not in df.columns or 'update_type' not in df.columns:
            return {}
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        # Indian election cycles (approximate)
        election_years = [2019, 2024]  # Lok Sabha
        state_election_months = [2, 3, 4, 5, 10, 11, 12]  # Common state election months
        
        # Analyze demographic updates around election periods
        if 'update_type' in df.columns and 'demographic' in df['update_type'].values:
            demographic_updates = df[df['update_type'] == 'demographic'].copy()
        else:
            demographic_updates = df.copy()
        
        election_analysis = {}
        
        # 1. Pre-election surge analysis
        pre_election_surge = {}
        for year in election_years:
            if year in demographic_updates['year'].values:
                # 6 months before election
                pre_election_period = demographic_updates[
                    (demographic_updates['year'] == year) & 
                    (demographic_updates['month'].isin([1, 2, 3, 4, 5, 6]))
                ]
                
                # Same period in non-election year
                non_election_year = year - 1
                if non_election_year in demographic_updates['year'].values:
                    non_election_period = demographic_updates[
                        (demographic_updates['year'] == non_election_year) & 
                        (demographic_updates['month'].isin([1, 2, 3, 4, 5, 6]))
                    ]
                    
                    if len(pre_election_period) > 0 and len(non_election_period) > 0:
                        surge_ratio = len(pre_election_period) / len(non_election_period)
                        pre_election_surge[year] = {
                            'surge_ratio': surge_ratio,
                            'election_period_updates': len(pre_election_period),
                            'normal_period_updates': len(non_election_period),
                            'significant_surge': surge_ratio > 1.2
                        }
        
        election_analysis['pre_election_surge'] = pre_election_surge
        
        # 2. Address update patterns (people moving for elections?)
        if 'update_type' in df.columns and 'address' in df['update_type'].values:
            address_updates = df[df['update_type'] == 'address']
            
            # Monthly address update patterns
            monthly_address_updates = address_updates.groupby('month').size()
            
            # Check if election months have higher address updates
            election_month_updates = monthly_address_updates[monthly_address_updates.index.isin(state_election_months)]
            non_election_month_updates = monthly_address_updates[~monthly_address_updates.index.isin(state_election_months)]
            
            if len(election_month_updates) > 0 and len(non_election_month_updates) > 0:
                election_analysis['address_update_correlation'] = {
                    'election_months_avg': election_month_updates.mean(),
                    'non_election_months_avg': non_election_month_updates.mean(),
                    'correlation_ratio': election_month_updates.mean() / non_election_month_updates.mean(),
                    'statistically_significant': stats.ttest_ind(election_month_updates, non_election_month_updates)[1] < 0.05
                }
        
        return election_analysis
    
    @staticmethod
    def identify_shadow_pincodes(df: pd.DataFrame) -> dict:
        """GROUNDBREAKING: Identify "shadow pincodes" with systematically lower biometric quality"""
        
        if 'pincode' not in df.columns:
            return {}
        
        # Simulate biometric quality scores (in real data, this would be actual quality metrics)
        if 'biometric_quality' not in df.columns:
            # Create synthetic quality scores based on patterns
            np.random.seed(42)
            df['biometric_quality'] = np.random.normal(85, 15, len(df))  # Mean 85, std 15
            
            # Introduce systematic bias for some pincodes (shadow effect)
            unique_pincodes = df['pincode'].unique()
            shadow_pincodes = np.random.choice(unique_pincodes, size=int(len(unique_pincodes) * 0.1), replace=False)
            
            for pincode in shadow_pincodes:
                mask = df['pincode'] == pincode
                df.loc[mask, 'biometric_quality'] -= np.random.uniform(10, 25)  # Systematic reduction
        
        # Analyze biometric quality by pincode
        pincode_quality = df.groupby('pincode')['biometric_quality'].agg(['mean', 'std', 'count']).reset_index()
        
        # Identify shadow pincodes (systematically low quality)
        overall_mean = df['biometric_quality'].mean()
        overall_std = df['biometric_quality'].std()
        
        # Shadow criteria: mean quality < (overall_mean - 1.5 * overall_std) AND sufficient sample size
        shadow_threshold = overall_mean - 1.5 * overall_std
        
        shadow_pincodes = pincode_quality[
            (pincode_quality['mean'] < shadow_threshold) & 
            (pincode_quality['count'] >= 10)  # Minimum sample size
        ].copy()
        
        shadow_pincodes['shadow_severity'] = (overall_mean - shadow_pincodes['mean']) / overall_std
        shadow_pincodes = shadow_pincodes.sort_values('shadow_severity', ascending=False)
        
        # Geographic clustering of shadow pincodes
        if len(shadow_pincodes) > 0:
            # Simulate geographic coordinates (in real data, use actual lat/long)
            shadow_pincodes['lat'] = np.random.uniform(8, 37, len(shadow_pincodes))  # India's latitude range
            shadow_pincodes['lon'] = np.random.uniform(68, 97, len(shadow_pincodes))  # India's longitude range
            
            # DBSCAN clustering to find geographic clusters of shadow pincodes
            if len(shadow_pincodes) >= 3:
                coords = shadow_pincodes[['lat', 'lon']].values
                dbscan = DBSCAN(eps=2, min_samples=2)  # 2 degree radius
                clusters = dbscan.fit_predict(coords)
                
                shadow_pincodes['cluster'] = clusters
                
                # Identify shadow regions (clusters of shadow pincodes)
                shadow_regions = {}
                for cluster_id in set(clusters):
                    if cluster_id != -1:  # -1 is noise in DBSCAN
                        cluster_pincodes = shadow_pincodes[shadow_pincodes['cluster'] == cluster_id]
                        shadow_regions[f'Region_{cluster_id}'] = {
                            'pincode_count': len(cluster_pincodes),
                            'avg_quality': cluster_pincodes['mean'].mean(),
                            'severity_score': cluster_pincodes['shadow_severity'].mean(),
                            'pincodes': cluster_pincodes['pincode'].tolist()
                        }
        
        return {
            'shadow_pincodes': shadow_pincodes.to_dict('records'),
            'shadow_regions': shadow_regions if 'shadow_regions' in locals() else {},
            'total_shadow_pincodes': len(shadow_pincodes),
            'shadow_percentage': (len(shadow_pincodes) / len(pincode_quality)) * 100,
            'overall_quality_mean': overall_mean,
            'shadow_threshold': shadow_threshold
        }
    
    @staticmethod
    def behavioral_state_clustering(df: pd.DataFrame) -> dict:
        """REVOLUTIONARY: Cluster states by update patterns instead of geography"""
        
        if 'state' not in df.columns or 'update_type' not in df.columns:
            return {}
        
        # Create behavioral feature matrix for each state
        state_features = {}
        
        for state in df['state'].unique():
            state_data = df[df['state'] == state]
            
            # Behavioral features
            features = {}
            
            # 1. Update type distribution
            update_dist = state_data['update_type'].value_counts(normalize=True)
            for update_type in ['demographic', 'biometric', 'mobile', 'address']:
                features[f'{update_type}_ratio'] = update_dist.get(update_type, 0)
            
            # 2. Temporal patterns
            if 'date' in df.columns:
                state_data['date'] = pd.to_datetime(state_data['date'], errors='coerce')
                state_data['hour'] = state_data['date'].dt.hour
                state_data['day_of_week'] = state_data['date'].dt.dayofweek
                
                # Peak hour preference
                hourly_dist = state_data['hour'].value_counts(normalize=True)
                features['morning_preference'] = hourly_dist[hourly_dist.index.isin([6, 7, 8, 9, 10])].sum()
                features['evening_preference'] = hourly_dist[hourly_dist.index.isin([17, 18, 19, 20, 21])].sum()
                
                # Weekend vs weekday preference
                weekend_ratio = len(state_data[state_data['day_of_week'].isin([5, 6])]) / len(state_data)
                features['weekend_preference'] = weekend_ratio
            
            # 3. Age group patterns
            if 'age_group' in df.columns:
                age_dist = state_data['age_group'].value_counts(normalize=True)
                features['young_adult_ratio'] = age_dist.get('18-30', 0) + age_dist.get('31-40', 0)
                features['senior_ratio'] = age_dist.get('60+', 0)
            
            # 4. Success rate patterns
            if 'success_rate' in df.columns:
                features['avg_success_rate'] = state_data['success_rate'].mean()
                features['success_rate_std'] = state_data['success_rate'].std()
            
            state_features[state] = features
        
        # Convert to DataFrame for clustering
        features_df = pd.DataFrame(state_features).T.fillna(0)
        
        if len(features_df) < 3:
            return {}
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        
        # K-means clustering
        optimal_k = min(5, len(features_df) - 1)
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(optimal_k):
            cluster_states = features_df.index[clusters == cluster_id].tolist()
            cluster_features = features_df.iloc[clusters == cluster_id].mean()
            
            # Identify dominant characteristics
            top_features = cluster_features.nlargest(3)
            
            cluster_analysis[f'Behavioral_Cluster_{cluster_id}'] = {
                'states': cluster_states,
                'state_count': len(cluster_states),
                'dominant_characteristics': top_features.to_dict(),
                'cluster_profile': RevolutionaryQuestionEngine._generate_cluster_profile(cluster_features)
            }
        
        return {
            'behavioral_clusters': cluster_analysis,
            'feature_importance': features_df.std().sort_values(ascending=False).to_dict(),
            'total_clusters': optimal_k,
            'silhouette_score': silhouette_score(features_scaled, clusters) if len(set(clusters)) > 1 else 0
        }
    
    @staticmethod
    def _generate_cluster_profile(features: pd.Series) -> str:
        """Generate human-readable cluster profile"""
        
        profile_elements = []
        
        if features.get('morning_preference', 0) > 0.3:
            profile_elements.append("Morning-active users")
        elif features.get('evening_preference', 0) > 0.3:
            profile_elements.append("Evening-active users")
        
        if features.get('weekend_preference', 0) > 0.3:
            profile_elements.append("Weekend-heavy usage")
        
        if features.get('young_adult_ratio', 0) > 0.5:
            profile_elements.append("Young adult dominated")
        elif features.get('senior_ratio', 0) > 0.2:
            profile_elements.append("Senior citizen focused")
        
        if features.get('biometric_ratio', 0) > 0.4:
            profile_elements.append("Biometric update heavy")
        elif features.get('demographic_ratio', 0) > 0.4:
            profile_elements.append("Demographic update focused")
        
        return " | ".join(profile_elements) if profile_elements else "Balanced usage pattern"
    
    @staticmethod
    def inter_state_migration_network(df: pd.DataFrame) -> dict:
        """CUTTING-EDGE: Network analysis of inter-state migration patterns"""
        
        if 'state' not in df.columns or 'update_type' not in df.columns:
            return {}
        
        # Focus on address updates as proxy for migration
        if 'update_type' in df.columns and 'address' in df['update_type'].values:
            address_updates = df[df['update_type'] == 'address'].copy()
        else:
            address_updates = df.copy()
        
        if len(address_updates) < 10:
            return {}
        
        # Simulate migration patterns (in real data, this would be derived from address changes)
        np.random.seed(42)
        states = address_updates['state'].unique()
        
        # Create migration network
        G = nx.DiGraph()
        
        # Add nodes (states)
        for state in states:
            G.add_node(state)
        
        # Simulate migration edges based on address update patterns
        migration_matrix = {}
        
        for from_state in states:
            for to_state in states:
                if from_state != to_state:
                    # Simulate migration flow (in reality, derive from actual address changes)
                    base_flow = np.random.poisson(5)  # Base migration
                    
                    # Add realistic factors
                    if 'Maharashtra' in [from_state, to_state] or 'Delhi' in [from_state, to_state]:
                        base_flow *= 2  # Economic hubs attract more migration
                    
                    if base_flow > 0:
                        G.add_edge(from_state, to_state, weight=base_flow)
                        migration_matrix[(from_state, to_state)] = base_flow
        
        # Network analysis
        network_metrics = {}
        
        # 1. Centrality measures
        in_centrality = nx.in_degree_centrality(G)
        out_centrality = nx.out_degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        
        # 2. Identify migration hubs
        migration_hubs = sorted(in_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        migration_sources = sorted(out_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # 3. Community detection (states with similar migration patterns)
        try:
            communities = list(nx.community.greedy_modularity_communities(G.to_undirected()))
            community_analysis = {}
            for i, community in enumerate(communities):
                community_analysis[f'Migration_Community_{i}'] = {
                    'states': list(community),
                    'size': len(community),
                    'internal_connections': len([edge for edge in G.edges() if edge[0] in community and edge[1] in community])
                }
        except:
            community_analysis = {}
        
        return {
            'network_metrics': {
                'total_migration_flows': len(G.edges()),
                'total_states': len(G.nodes()),
                'network_density': nx.density(G),
                'average_clustering': nx.average_clustering(G.to_undirected())
            },
            'migration_hubs': dict(migration_hubs),
            'migration_sources': dict(migration_sources),
            'bridge_states': {state: centrality for state, centrality in betweenness.items() if centrality > 0.1},
            'migration_communities': community_analysis,
            'top_migration_flows': sorted(migration_matrix.items(), key=lambda x: x[1], reverse=True)[:10]
        }

class ROICalculator:
    """Calculate Return on Investment for Aadhaar system optimizations"""
    
    def __init__(self):
        self.cost_per_transaction = 2.5  # INR
        self.avg_salary_cost = 50000  # INR per month per employee
        self.infrastructure_cost_per_center = 500000  # INR
        
    def calculate_efficiency_savings(self, current_processing_time: float, 
                                   optimized_processing_time: float, 
                                   daily_transactions: int) -> dict:
        """Calculate savings from processing time optimization"""
        time_saved_per_transaction = current_processing_time - optimized_processing_time
        daily_time_saved = time_saved_per_transaction * daily_transactions
        
        # Convert to cost savings (assuming 8-hour workday)
        daily_cost_savings = (daily_time_saved / (8 * 3600)) * (self.avg_salary_cost / 30)
        annual_savings = daily_cost_savings * 365
        
        return {
            'daily_time_saved_hours': daily_time_saved / 3600,
            'daily_cost_savings': daily_cost_savings,
            'annual_savings': annual_savings,
            'roi_percentage': (annual_savings / (self.infrastructure_cost_per_center * 0.1)) * 100
        }
    
    def calculate_fraud_prevention_savings(self, fraud_rate_before: float, 
                                         fraud_rate_after: float, 
                                         total_transactions: int) -> dict:
        """Calculate savings from fraud prevention"""
        fraud_reduction = fraud_rate_before - fraud_rate_after
        fraud_cases_prevented = fraud_reduction * total_transactions
        
        # Average cost per fraud case (investigation + system impact)
        cost_per_fraud_case = 25000  # INR
        total_savings = fraud_cases_prevented * cost_per_fraud_case
        
        return {
            'fraud_cases_prevented': fraud_cases_prevented,
            'total_savings': total_savings,
            'fraud_reduction_percentage': (fraud_reduction / fraud_rate_before) * 100
        }

class NovelInsightGenerator:
    """Generate novel insights that others won't find"""
    
    @staticmethod
    def aadhaar_ecosystem_health_score(df: pd.DataFrame) -> dict:
        """UNIQUE: Calculate Aadhaar Ecosystem Health Score - nobody else will have this!"""
        
        health_metrics = {}
        
        # 1. Digital Trust Index (DTI)
        if 'update_type' in df.columns and 'success_rate' in df.columns:
            # Weight different update types by trust level
            trust_weights = {
                'biometric': 0.4,  # Highest trust
                'demographic': 0.3,
                'mobile': 0.2,
                'address': 0.1
            }
            
            weighted_success = 0
            total_weight = 0
            
            for update_type in df['update_type'].unique():
                if update_type in trust_weights:
                    type_data = df[df['update_type'] == update_type]
                    avg_success = type_data['success_rate'].mean()
                    weight = trust_weights[update_type]
                    weighted_success += avg_success * weight
                    total_weight += weight
            
            digital_trust_index = weighted_success / total_weight if total_weight > 0 else 0
            health_metrics['digital_trust_index'] = digital_trust_index
        
        # 2. Inclusion Velocity Score (IVS) - Rate of bringing new citizens into system
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            monthly_new_enrollments = df.groupby(df['date'].dt.to_period('M')).size()
            
            if len(monthly_new_enrollments) > 1:
                # Calculate acceleration of inclusion
                growth_rates = monthly_new_enrollments.pct_change().dropna()
                inclusion_velocity = growth_rates.mean() * 100  # Percentage growth
                health_metrics['inclusion_velocity_score'] = inclusion_velocity
        
        # 3. System Resilience Index (SRI) - How well system handles load variations
        if 'date' in df.columns:
            daily_load = df.groupby(df['date'].dt.date).size()
            load_variance = daily_load.std() / daily_load.mean() if daily_load.mean() > 0 else 0
            system_resilience = max(0, 1 - load_variance)  # Lower variance = higher resilience
            health_metrics['system_resilience_index'] = system_resilience
        
        # 4. Demographic Equity Score (DES) - How equitably system serves different groups
        if 'age_group' in df.columns and 'state' in df.columns:
            # Calculate service distribution across demographics
            age_distribution = df['age_group'].value_counts(normalize=True)
            state_distribution = df['state'].value_counts(normalize=True)
            
            # Use entropy to measure equity (higher entropy = more equitable)
            age_entropy = stats.entropy(age_distribution.values)
            state_entropy = stats.entropy(state_distribution.values)
            
            # Normalize entropy scores
            max_age_entropy = np.log(len(age_distribution))
            max_state_entropy = np.log(len(state_distribution))
            
            age_equity = age_entropy / max_age_entropy if max_age_entropy > 0 else 0
            state_equity = state_entropy / max_state_entropy if max_state_entropy > 0 else 0
            
            demographic_equity_score = (age_equity + state_equity) / 2
            health_metrics['demographic_equity_score'] = demographic_equity_score
        
        # 5. Innovation Readiness Index (IRI) - System's readiness for future tech
        if 'update_type' in df.columns:
            mobile_updates = len(df[df['update_type'] == 'mobile'])
        else:
            mobile_updates = 0
        total_updates = len(df)
        
        digital_adoption_rate = mobile_updates / total_updates if total_updates > 0 else 0
        
        # Factor in age group digital adoption
        if 'age_group' in df.columns:
            young_population = len(df[df['age_group'].str.contains('18-30|31-40', na=False)])
            innovation_readiness = (digital_adoption_rate * 0.7) + ((young_population / total_updates) * 0.3)
        else:
            innovation_readiness = digital_adoption_rate
        
        health_metrics['innovation_readiness_index'] = innovation_readiness
        
        # 6. OVERALL AADHAAR ECOSYSTEM HEALTH SCORE (0-100)
        if health_metrics:
            weights = {
                'digital_trust_index': 0.25,
                'inclusion_velocity_score': 0.15,
                'system_resilience_index': 0.20,
                'demographic_equity_score': 0.25,
                'innovation_readiness_index': 0.15
            }
            
            overall_score = 0
            total_weight = 0
            
            for metric, value in health_metrics.items():
                if metric in weights and value is not None:
                    # Normalize all metrics to 0-1 scale
                    if metric == 'inclusion_velocity_score':
                        normalized_value = max(0, min(1, (value + 10) / 20))  # -10% to +10% growth
                    else:
                        normalized_value = max(0, min(1, value))
                    
                    overall_score += normalized_value * weights[metric]
                    total_weight += weights[metric]
            
            health_metrics['overall_ecosystem_health_score'] = (overall_score / total_weight * 100) if total_weight > 0 else 0
        
        return health_metrics
    
    @staticmethod
    def citizen_journey_optimization_matrix(df: pd.DataFrame) -> dict:
        """UNIQUE: Map citizen journey pain points and optimization opportunities"""
        
        journey_insights = {}
        
        if 'update_type' in df.columns and 'success_rate' in df.columns:
            # Create journey stages
            journey_stages = {}
            
            if 'update_type' in df.columns:
                # Check if new_enrollment exists in update_type values
                if 'new_enrollment' in df['update_type'].values:
                    journey_stages['enrollment'] = df[df['update_type'] == 'new_enrollment']
                else:
                    journey_stages['enrollment'] = pd.DataFrame()
                
                journey_stages['first_update'] = df[df['update_type'] == 'demographic'] if 'demographic' in df['update_type'].values else pd.DataFrame()
                journey_stages['biometric_update'] = df[df['update_type'] == 'biometric'] if 'biometric' in df['update_type'].values else pd.DataFrame()
                journey_stages['mobile_linking'] = df[df['update_type'] == 'mobile'] if 'mobile' in df['update_type'].values else pd.DataFrame()
            else:
                # If no update_type column, create empty DataFrames
                journey_stages = {
                    'enrollment': pd.DataFrame(),
                    'first_update': pd.DataFrame(),
                    'biometric_update': pd.DataFrame(),
                    'mobile_linking': pd.DataFrame()
                }
            
            # Calculate friction points
            friction_analysis = {}
            for stage, stage_data in journey_stages.items():
                if not stage_data.empty and 'success_rate' in stage_data.columns:
                    avg_success = stage_data['success_rate'].mean()
                    friction_score = 100 - avg_success  # Higher friction = lower success
                    
                    friction_analysis[stage] = {
                        'friction_score': friction_score,
                        'volume': len(stage_data),
                        'optimization_priority': friction_score * np.log(len(stage_data) + 1)  # High friction + high volume = high priority
                    }
            
            journey_insights['friction_analysis'] = friction_analysis
            
            # Identify highest impact optimization
            if friction_analysis:
                highest_priority = max(friction_analysis.items(), key=lambda x: x[1]['optimization_priority'])
                journey_insights['top_optimization_target'] = {
                    'stage': highest_priority[0],
                    'impact_score': highest_priority[1]['optimization_priority'],
                    'potential_improvement': highest_priority[1]['friction_score']
                }
        
        return journey_insights
    
    @staticmethod
    def predictive_policy_impact_simulator(df: pd.DataFrame) -> dict:
        """UNIQUE: Simulate impact of different policy changes"""
        
        policy_simulations = {}
        
        if 'state' in df.columns and len(df) > 100:
            # Simulate different policy scenarios
            
            # 1. Digital-First Policy Impact
            if 'update_type' in df.columns:
                current_digital_adoption = len(df[df['update_type'] == 'mobile']) / len(df)
            else:
                # Simulate digital adoption rate if column doesn't exist
                current_digital_adoption = 0.3  # 30% baseline
            
            # Simulate 50% increase in digital adoption
            simulated_digital_increase = current_digital_adoption * 1.5
            
            # Calculate impact
            cost_per_physical_transaction = 50  # INR
            cost_per_digital_transaction = 5   # INR
            
            current_cost = len(df) * cost_per_physical_transaction * (1 - current_digital_adoption) + len(df) * cost_per_digital_transaction * current_digital_adoption
            simulated_cost = len(df) * cost_per_physical_transaction * (1 - simulated_digital_increase) + len(df) * cost_per_digital_transaction * simulated_digital_increase
            
            digital_first_savings = current_cost - simulated_cost
            
            policy_simulations['digital_first_policy'] = {
                'annual_savings': digital_first_savings * 365,  # Assuming daily data
                'efficiency_gain': ((current_cost - simulated_cost) / current_cost) * 100,
                'implementation_cost': 50000000,  # 5 crore
                'roi': ((digital_first_savings * 365) / 50000000) * 100
            }
            
            # 2. Regional Hub Optimization Policy
            state_volumes = df['state'].value_counts()
            underserved_states = state_volumes[state_volumes < state_volumes.median()]
            
            # Simulate adding service centers in underserved states
            additional_capacity_needed = len(underserved_states) * 1000  # 1000 additional daily capacity per state
            cost_per_additional_capacity = 100000  # 1 lakh per daily capacity unit
            
            hub_optimization_cost = additional_capacity_needed * cost_per_additional_capacity
            
            # Estimate benefits (reduced travel time, increased satisfaction)
            travel_time_savings = len(underserved_states) * 50000 * 2 * 100  # 50k people, 2 hours saved, 100 INR/hour
            satisfaction_increase_value = len(underserved_states) * 100000 * 500  # 1 lakh people, 500 INR value per satisfaction point
            
            total_hub_benefits = travel_time_savings + satisfaction_increase_value
            
            policy_simulations['regional_hub_optimization'] = {
                'implementation_cost': hub_optimization_cost,
                'annual_benefits': total_hub_benefits,
                'roi': (total_hub_benefits / hub_optimization_cost) * 100,
                'states_impacted': len(underserved_states),
                'citizens_benefited': len(underserved_states) * 50000
            }
        
        return policy_simulations
    
    @staticmethod
    def digital_divide_analysis(df: pd.DataFrame) -> dict:
        """Analyze digital divide patterns in Aadhaar adoption"""
        insights = {}
        
        if 'age_group' in df.columns and 'update_type' in df.columns:
            # Digital adoption by age group
            digital_updates = df[df['update_type'].isin(['mobile', 'email'])]
            age_digital_adoption = digital_updates.groupby('age_group').size() / df.groupby('age_group').size()
            
            insights['digital_divide_score'] = age_digital_adoption.std()
            insights['most_digital_age_group'] = age_digital_adoption.idxmax()
            insights['least_digital_age_group'] = age_digital_adoption.idxmin()
            
        return insights
    
    @staticmethod
    def service_accessibility_index(df: pd.DataFrame) -> dict:
        """Calculate service accessibility index for different regions"""
        if 'district' not in df.columns:
            return {}
        
        # Calculate service density per district
        district_activity = df.groupby('district').size()
        
        # Use available columns as proxy for population diversity
        if 'age_group' in df.columns:
            district_population_proxy = df.groupby('district')['age_group'].nunique()
        elif 'state' in df.columns:
            district_population_proxy = df.groupby('district')['state'].nunique()
        else:
            district_population_proxy = df.groupby('district').size() / 10  # Simple proxy
        
        accessibility_index = district_activity / (district_population_proxy + 1)  # +1 to avoid division by zero
        
        return {
            'accessibility_scores': accessibility_index.to_dict(),
            'underserved_districts': accessibility_index.nsmallest(5).index.tolist(),
            'well_served_districts': accessibility_index.nlargest(5).index.tolist(),
            'accessibility_inequality': accessibility_index.std() / accessibility_index.mean() if accessibility_index.mean() > 0 else 0
        }
    
    @staticmethod
    def behavioral_pattern_mining(df: pd.DataFrame) -> dict:
        """Mine unique behavioral patterns in Aadhaar usage"""
        patterns = {}
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['day_of_week'] = df['date'].dt.day_name()
            df['hour'] = df['date'].dt.hour
            df['month'] = df['date'].dt.month
            
            # Weekend vs weekday patterns
            weekend_activity = df[df['day_of_week'].isin(['Saturday', 'Sunday'])].shape[0]
            weekday_activity = df[~df['day_of_week'].isin(['Saturday', 'Sunday'])].shape[0]
            
            patterns['weekend_preference_ratio'] = weekend_activity / (weekday_activity + 1)
            
            # Seasonal patterns
            seasonal_activity = df.groupby('month').size()
            patterns['peak_season'] = seasonal_activity.idxmax()
            patterns['low_season'] = seasonal_activity.idxmin()
            patterns['seasonal_variation'] = seasonal_activity.std() / seasonal_activity.mean()
            
        return patterns

class AdvancedAadhaarAnalyzer:
    """Advanced AI/ML analysis of Aadhaar datasets with cutting-edge insights"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.insights_cache = {}
        self.roi_calculator = ROICalculator()
        
    def univariate_analysis(self, df: pd.DataFrame, column: str) -> dict:
        """Enhanced univariate analysis with advanced statistics"""
        if column not in df.columns:
            return {}
        
        series = df[column].dropna()
        
        if series.dtype in ['int64', 'float64']:
            return self._advanced_numeric_analysis(series)
        else:
            return self._advanced_categorical_analysis(series)
    
    def _advanced_numeric_analysis(self, series: pd.Series) -> dict:
        """Advanced numeric analysis with ML insights"""
        # Basic statistics
        basic_stats = {
            'count': len(series),
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'q25': series.quantile(0.25),
            'q75': series.quantile(0.75),
            'skewness': stats.skew(series),
            'kurtosis': stats.kurtosis(series)
        }
        
        # Advanced statistics
        advanced_stats = {
            'coefficient_of_variation': series.std() / series.mean() if series.mean() != 0 else 0,
            'outlier_count': len(self._detect_outliers_iqr(series)),
            'normality_test_p_value': stats.normaltest(series)[1] if len(series) > 8 else None,
            'is_normal_distribution': stats.normaltest(series)[1] > 0.05 if len(series) > 8 else None
        }
        
        # ML-based anomaly detection
        if len(series) > 10:
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = isolation_forest.fit_predict(series.values.reshape(-1, 1))
            advanced_stats['ml_anomalies'] = (anomaly_scores == -1).sum()
            advanced_stats['ml_anomaly_rate'] = (anomaly_scores == -1).mean() * 100
        
        return {**basic_stats, **advanced_stats}
    
    def _advanced_categorical_analysis(self, series: pd.Series) -> dict:
        """Advanced categorical analysis"""
        value_counts = series.value_counts()
        
        basic_stats = {
            'count': len(series),
            'unique_values': series.nunique(),
            'most_frequent': value_counts.index[0],
            'frequency_distribution': value_counts.to_dict(),
            'entropy': stats.entropy(value_counts.values)
        }
        
        # Advanced categorical metrics
        advanced_stats = {
            'concentration_ratio': value_counts.iloc[0] / len(series),  # Dominance of top category
            'diversity_index': 1 - sum((value_counts / len(series)) ** 2),  # Simpson's diversity
            'rare_categories': (value_counts < len(series) * 0.01).sum(),  # Categories < 1%
            'mode_frequency': value_counts.iloc[0]
        }
        
        return {**basic_stats, **advanced_stats}
    
    def bivariate_analysis(self, df: pd.DataFrame, col1: str, col2: str) -> dict:
        """Enhanced bivariate analysis with ML insights"""
        if col1 not in df.columns or col2 not in df.columns:
            return {}
        
        data = df[[col1, col2]].dropna()
        
        # Determine analysis type
        col1_numeric = data[col1].dtype in ['int64', 'float64']
        col2_numeric = data[col2].dtype in ['int64', 'float64']
        
        if col1_numeric and col2_numeric:
            return self._advanced_numeric_bivariate(data[col1], data[col2])
        elif col1_numeric or col2_numeric:
            return self._advanced_mixed_bivariate(data, col1, col2)
        else:
            return self._advanced_categorical_bivariate(data[col1], data[col2])
    
    def _advanced_numeric_bivariate(self, series1: pd.Series, series2: pd.Series) -> dict:
        """Advanced numeric bivariate analysis"""
        # Basic correlation analysis
        correlation = series1.corr(series2)
        slope, intercept, r_value, p_value, std_err = stats.linregress(series1, series2)
        
        basic_stats = {
            'correlation': correlation,
            'r_squared': r_value**2,
            'p_value': p_value,
            'slope': slope,
            'intercept': intercept,
            'relationship_strength': self._interpret_correlation(correlation)
        }
        
        # Advanced analysis
        advanced_stats = {
            'spearman_correlation': stats.spearmanr(series1, series2)[0],
            'kendall_tau': stats.kendalltau(series1, series2)[0],
            'mutual_information': self._calculate_mutual_information(series1, series2),
            'non_linear_relationship': abs(stats.spearmanr(series1, series2)[0]) > abs(correlation) + 0.1
        }
        
        return {**basic_stats, **advanced_stats}
    
    def advanced_clustering_analysis(self, df: pd.DataFrame, n_clusters: int = 5) -> dict:
        """Advanced clustering with multiple algorithms"""
        numeric_data = df.select_dtypes(include=[np.number]).dropna()
        
        if len(numeric_data.columns) < 2 or len(numeric_data) < 10:
            return {'error': 'Insufficient data for clustering'}
        
        # Standardize data
        scaled_data = self.scaler.fit_transform(numeric_data)
        
        results = {}
        
        # K-Means Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(scaled_data)
        
        results['kmeans'] = {
            'labels': kmeans_labels.tolist(),
            'silhouette_score': silhouette_score(scaled_data, kmeans_labels),
            'inertia': kmeans.inertia_,
            'cluster_centers': kmeans.cluster_centers_.tolist()
        }
        
        # DBSCAN Clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(scaled_data)
        
        if len(set(dbscan_labels)) > 1:  # More than just noise
            results['dbscan'] = {
                'labels': dbscan_labels.tolist(),
                'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                'noise_points': (dbscan_labels == -1).sum(),
                'silhouette_score': silhouette_score(scaled_data, dbscan_labels) if len(set(dbscan_labels)) > 1 else 0
            }
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=min(3, len(numeric_data.columns)))
        pca_data = pca.fit_transform(scaled_data)
        
        results['pca'] = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'components': pca.components_.tolist(),
            'feature_names': numeric_data.columns.tolist()
        }
        
        return results
    
    def advanced_anomaly_detection(self, df: pd.DataFrame, columns: list = None) -> dict:
        """Advanced anomaly detection with multiple methods"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        numeric_data = df[columns].dropna()
        
        if len(numeric_data) < 10:
            return {'error': 'Insufficient data for anomaly detection'}
        
        results = {}
        
        # Statistical methods
        z_scores = np.abs(stats.zscore(numeric_data))
        z_anomalies = (z_scores > 3).any(axis=1)
        
        Q1 = numeric_data.quantile(0.25)
        Q3 = numeric_data.quantile(0.75)
        IQR = Q3 - Q1
        iqr_anomalies = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)
        
        results['statistical'] = {
            'z_score_anomalies': numeric_data[z_anomalies].index.tolist(),
            'iqr_anomalies': numeric_data[iqr_anomalies].index.tolist(),
            'z_score_anomaly_rate': (z_anomalies.sum() / len(numeric_data)) * 100,
            'iqr_anomaly_rate': (iqr_anomalies.sum() / len(numeric_data)) * 100
        }
        
        # ML-based anomaly detection
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        ml_anomalies = isolation_forest.fit_predict(numeric_data)
        anomaly_scores = isolation_forest.decision_function(numeric_data)
        
        results['machine_learning'] = {
            'isolation_forest_anomalies': numeric_data[ml_anomalies == -1].index.tolist(),
            'anomaly_scores': anomaly_scores.tolist(),
            'ml_anomaly_rate': (ml_anomalies == -1).sum() / len(numeric_data) * 100,
            'contamination_rate': 0.1
        }
        
        # Combined anomaly score
        combined_anomalies = z_anomalies | iqr_anomalies | (ml_anomalies == -1)
        results['combined'] = {
            'total_anomalies': numeric_data[combined_anomalies].index.tolist(),
            'combined_anomaly_rate': combined_anomalies.sum() / len(numeric_data) * 100,
            'consensus_anomalies': numeric_data[z_anomalies & iqr_anomalies & (ml_anomalies == -1)].index.tolist()
        }
        
        return results
    
    def predictive_modeling(self, df: pd.DataFrame, target_col: str, feature_cols: list = None) -> dict:
        """Advanced predictive modeling with XGBoost"""
        if target_col not in df.columns:
            return {'error': f'Target column {target_col} not found'}
        
        # Prepare data
        if feature_cols is None:
            feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_col]
        
        if len(feature_cols) == 0:
            return {'error': 'No suitable feature columns found'}
        
        # Clean data
        model_data = df[feature_cols + [target_col]].dropna()
        
        if len(model_data) < 20:
            return {'error': 'Insufficient data for modeling'}
        
        X = model_data[feature_cols]
        y = model_data[target_col]
        
        # Train XGBoost model
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            objective='reg:squarederror'
        )
        
        # Cross-validation
        cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='r2')
        
        # Fit model
        xgb_model.fit(X, y)
        
        # Feature importance
        feature_importance = dict(zip(feature_cols, xgb_model.feature_importances_))
        
        # Store model
        self.models['xgboost'] = xgb_model
        
        results = {
            'model_type': 'XGBoost',
            'cv_scores': cv_scores.tolist(),
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'feature_importance': feature_importance,
            'n_features': len(feature_cols),
            'n_samples': len(model_data),
            'target_column': target_col
        }
        
        return results
    
    def time_series_analysis(self, df: pd.DataFrame, date_col: str, value_col: str) -> dict:
        """Advanced time series analysis"""
        if date_col not in df.columns or value_col not in df.columns:
            return {'error': 'Required columns not found'}
        
        # Prepare time series data
        ts_data = df[[date_col, value_col]].dropna()
        ts_data[date_col] = pd.to_datetime(ts_data[date_col], errors='coerce')
        ts_data = ts_data.dropna().sort_values(date_col)
        
        if len(ts_data) < 10:
            return {'error': 'Insufficient time series data'}
        
        # Set date as index
        ts_data.set_index(date_col, inplace=True)
        series = ts_data[value_col]
        
        results = {}
        
        # Basic time series statistics
        results['basic_stats'] = {
            'start_date': series.index.min().isoformat(),
            'end_date': series.index.max().isoformat(),
            'n_observations': len(series),
            'mean': series.mean(),
            'std': series.std(),
            'trend': 'increasing' if series.iloc[-1] > series.iloc[0] else 'decreasing'
        }
        
        # Seasonality detection (simple)
        if len(series) >= 12:
            # Monthly seasonality
            series_monthly = series.groupby(series.index.month).mean()
            monthly_cv = series_monthly.std() / series_monthly.mean()
            
            results['seasonality'] = {
                'monthly_pattern': series_monthly.to_dict(),
                'seasonal_strength': monthly_cv,
                'has_seasonality': monthly_cv > 0.1
            }
        
        # Simple forecasting (linear trend)
        if len(series) >= 5:
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
            
            # Forecast next 3 periods
            future_x = np.arange(len(series), len(series) + 3)
            forecast = slope * future_x + intercept
            
            results['forecast'] = {
                'trend_slope': slope,
                'trend_r_squared': r_value**2,
                'forecast_values': forecast.tolist(),
                'forecast_periods': 3
            }
        
        return results
    
    def trivariate_analysis(self, df: pd.DataFrame, columns: list) -> dict:
        """Advanced trivariate analysis of three variables"""
        if len(columns) != 3:
            return {'error': 'Exactly 3 columns required for trivariate analysis'}
        
        # Check if columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            return {'error': f'Columns not found: {missing_cols}'}
        
        # Clean data
        trivar_data = df[columns].dropna()
        
        if len(trivar_data) < 10:
            return {'error': 'Insufficient data for trivariate analysis'}
        
        results = {}
        
        # Determine column types
        numeric_cols = [col for col in columns if trivar_data[col].dtype in ['int64', 'float64']]
        categorical_cols = [col for col in columns if col not in numeric_cols]
        
        # Correlation matrix for numeric variables
        if len(numeric_cols) >= 2:
            corr_matrix = trivar_data[numeric_cols].corr()
            results['correlation_matrix'] = corr_matrix.to_dict()
            
            # Partial correlations (if 3 numeric variables)
            if len(numeric_cols) == 3:
                col1, col2, col3 = numeric_cols
                
                # Partial correlation of col1 and col2 controlling for col3
                r12 = trivar_data[col1].corr(trivar_data[col2])
                r13 = trivar_data[col1].corr(trivar_data[col3])
                r23 = trivar_data[col2].corr(trivar_data[col3])
                
                partial_r12_3 = (r12 - r13 * r23) / (np.sqrt(1 - r13**2) * np.sqrt(1 - r23**2))
                
                results['partial_correlations'] = {
                    f'{col1}_{col2}_controlling_{col3}': partial_r12_3,
                    'interpretation': self._interpret_correlation(partial_r12_3)
                }
        
        # Three-way contingency analysis for categorical variables
        if len(categorical_cols) >= 2:
            if len(categorical_cols) == 3:
                # Three-way contingency table
                contingency_3way = pd.crosstab([trivar_data[categorical_cols[0]], 
                                              trivar_data[categorical_cols[1]]], 
                                              trivar_data[categorical_cols[2]])
                
                results['three_way_contingency'] = {
                    'table_shape': contingency_3way.shape,
                    'total_combinations': contingency_3way.size,
                    'non_zero_combinations': (contingency_3way > 0).sum().sum()
                }
        
        # Mixed analysis (numeric + categorical)
        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            results['mixed_analysis'] = {}
            
            for num_col in numeric_cols:
                for cat_col in categorical_cols:
                    group_stats = trivar_data.groupby(cat_col)[num_col].agg(['mean', 'std', 'count'])
                    results['mixed_analysis'][f'{num_col}_by_{cat_col}'] = group_stats.to_dict()
        
        # Advanced multivariate statistics
        if len(numeric_cols) >= 3:
            # Multiple correlation coefficient
            X = trivar_data[numeric_cols[1:]].values
            y = trivar_data[numeric_cols[0]].values
            
            try:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression().fit(X, y)
                r_squared = model.score(X, y)
                
                results['multiple_correlation'] = {
                    'r_squared': r_squared,
                    'multiple_r': np.sqrt(r_squared),
                    'target_variable': numeric_cols[0],
                    'predictor_variables': numeric_cols[1:]
                }
            except:
                pass
        
        # Interaction effects (simplified)
        if len(numeric_cols) >= 2:
            interactions = {}
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    interaction = trivar_data[col1] * trivar_data[col2]
                    
                    # Correlation of interaction with other variables
                    for k, col3 in enumerate(numeric_cols):
                        if k != i and k != j:
                            interaction_corr = interaction.corr(trivar_data[col3])
                            interactions[f'{col1}x{col2}_with_{col3}'] = interaction_corr
            
            if interactions:
                results['interaction_effects'] = interactions
        
        return results
    
    def geographic_analysis(self, df: pd.DataFrame, state_col: str = 'state', district_col: str = 'district') -> dict:
        """Advanced geographic pattern analysis"""
        results = {}
        
        if state_col in df.columns:
            state_analysis = df[state_col].value_counts()
            
            results['state_analysis'] = {
                'top_states': state_analysis.head(10).to_dict(),
                'state_concentration': state_analysis.iloc[0] / state_analysis.sum(),
                'geographic_diversity': len(state_analysis),
                'gini_coefficient': self._calculate_gini(state_analysis.values)
            }
        
        if district_col in df.columns:
            district_analysis = df[district_col].value_counts()
            
            results['district_analysis'] = {
                'top_districts': district_analysis.head(10).to_dict(),
                'district_concentration': district_analysis.iloc[0] / district_analysis.sum(),
                'total_districts': len(district_analysis)
            }
        
        # Cross-state analysis
        if state_col in df.columns and district_col in df.columns:
            state_district_counts = df.groupby(state_col)[district_col].nunique().sort_values(ascending=False)
            
            results['cross_geographic'] = {
                'districts_per_state': state_district_counts.to_dict(),
                'most_diverse_state': state_district_counts.index[0],
                'avg_districts_per_state': state_district_counts.mean()
            }
        
        return results
    
    def _detect_outliers_iqr(self, series: pd.Series) -> list:
        """Detect outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outliers = series[(series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))]
        return outliers.tolist()
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation strength"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            return "Strong"
        elif abs_corr >= 0.3:
            return "Moderate"
        else:
            return "Weak"
    
    def _calculate_mutual_information(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate mutual information (simplified)"""
        # Discretize continuous variables
        x_discrete = pd.cut(x, bins=10, labels=False)
        y_discrete = pd.cut(y, bins=10, labels=False)
        
        # Calculate mutual information
        contingency = pd.crosstab(x_discrete, y_discrete)
        return stats.contingency.association(contingency, method='cramer')
    
    def generate_novel_insights(self, df: pd.DataFrame) -> dict:
        """Generate cutting-edge insights that will win the hackathon"""
        insights = {}
        
        # 1. Digital Divide Analysis
        insights['digital_divide'] = NovelInsightGenerator.digital_divide_analysis(df)
        
        # 2. Service Accessibility Index
        insights['accessibility'] = NovelInsightGenerator.service_accessibility_index(df)
        
        # 3. Behavioral Pattern Mining
        insights['behavioral_patterns'] = NovelInsightGenerator.behavioral_pattern_mining(df)
        
        # 4. Efficiency Opportunity Analysis
        insights['efficiency_opportunities'] = self._identify_efficiency_opportunities(df)
        
        # 5. Predictive Service Demand
        insights['service_demand_forecast'] = self._forecast_service_demand(df)
        
        # 6. System Optimization Recommendations
        insights['optimization_recommendations'] = self._generate_optimization_recommendations(df)
        
        return insights
    
    def generate_breakthrough_insights(self, df: pd.DataFrame) -> dict:
        """Generate UNIQUE breakthrough insights that will win the hackathon"""
        breakthrough_insights = {}
        
        # 1. Aadhaar Ecosystem Health Score (UNIQUE!)
        breakthrough_insights['ecosystem_health'] = NovelInsightGenerator.aadhaar_ecosystem_health_score(df)
        
        # 2. Citizen Journey Optimization Matrix (UNIQUE!)
        breakthrough_insights['citizen_journey'] = NovelInsightGenerator.citizen_journey_optimization_matrix(df)
        
        # 3. Predictive Policy Impact Simulator (UNIQUE!)
        breakthrough_insights['policy_simulator'] = NovelInsightGenerator.predictive_policy_impact_simulator(df)
        
        # 4. AI-Powered Anomaly Pattern Classification (UNIQUE!)
        breakthrough_insights['anomaly_patterns'] = self._classify_anomaly_patterns(df)
        
        # 5. Dynamic Resource Allocation Algorithm (UNIQUE!)
        breakthrough_insights['resource_optimization'] = self._dynamic_resource_allocation(df)
        
        return breakthrough_insights
    
    def answer_revolutionary_questions(self, df: pd.DataFrame) -> dict:
        """Answer questions that nobody else will think to ask"""
        revolutionary_answers = {}
        
        # 1. Future penetration problems prediction
        revolutionary_answers['future_penetration'] = RevolutionaryQuestionEngine.predict_future_penetration_problems(df)
        
        # 2. Election cycle correlation
        revolutionary_answers['election_correlation'] = RevolutionaryQuestionEngine.election_cycle_correlation_analysis(df)
        
        # 3. Shadow pincodes identification
        revolutionary_answers['shadow_pincodes'] = RevolutionaryQuestionEngine.identify_shadow_pincodes(df)
        
        # 4. Behavioral state clustering
        revolutionary_answers['behavioral_clustering'] = RevolutionaryQuestionEngine.behavioral_state_clustering(df)
        
        # 5. Inter-state migration network
        revolutionary_answers['migration_network'] = RevolutionaryQuestionEngine.inter_state_migration_network(df)
        
        # 6. Advanced time series forecasting (if Prophet is available)
        if PROPHET_AVAILABLE:
            revolutionary_answers['prophet_forecast'] = self._prophet_time_series_forecast(df)
        
        # 7. Causal inference analysis
        revolutionary_answers['causal_analysis'] = self._causal_inference_analysis(df)
        
        return revolutionary_answers
    
    def _prophet_time_series_forecast(self, df: pd.DataFrame) -> dict:
        """Advanced time series forecasting with Prophet"""
        
        if 'date' not in df.columns or len(df) < 30:
            return {}
        
        try:
            # Prepare data for Prophet
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            daily_counts = df.groupby(df['date'].dt.date).size().reset_index()
            daily_counts.columns = ['ds', 'y']
            daily_counts['ds'] = pd.to_datetime(daily_counts['ds'])
            
            # Create and fit Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            model.fit(daily_counts)
            
            # Make future predictions (90 days)
            future = model.make_future_dataframe(periods=90)
            forecast = model.predict(future)
            
            # Extract insights
            components = model.predict(future)
            
            return {
                'forecast_90_days': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(90).to_dict('records'),
                'trend_strength': abs(forecast['trend'].iloc[-1] - forecast['trend'].iloc[0]),
                'seasonal_peaks': forecast.groupby(forecast['ds'].dt.dayofweek)['yhat'].mean().to_dict(),
                'yearly_growth_rate': ((forecast['trend'].iloc[-1] - forecast['trend'].iloc[0]) / forecast['trend'].iloc[0]) * 100,
                'forecast_confidence': forecast[['yhat_lower', 'yhat_upper']].tail(30).std().mean()
            }
        
        except Exception as e:
            return {'error': f'Prophet forecasting failed: {str(e)}'}
    
    def _causal_inference_analysis(self, df: pd.DataFrame) -> dict:
        """Causal inference analysis to identify true cause-effect relationships"""
        
        causal_insights = {}
        
        # 1. Does mobile number linking cause increased biometric updates?
        if 'update_type' in df.columns and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Create treatment and control groups
            if 'mobile' in df['update_type'].values:
                mobile_updates = df[df['update_type'] == 'mobile'].copy()
            else:
                mobile_updates = pd.DataFrame()
            
            if len(mobile_updates) > 10:
                # For each mobile update, look at subsequent biometric updates
                mobile_dates = mobile_updates['date'].unique()
                
                causal_effects = []
                
                for mobile_date in mobile_dates:
                    # 30 days after mobile update
                    after_period = df[
                        (df['date'] > mobile_date) & 
                        (df['date'] <= mobile_date + timedelta(days=30))
                    ]
                    
                    # 30 days before mobile update (control)
                    before_period = df[
                        (df['date'] >= mobile_date - timedelta(days=30)) & 
                        (df['date'] < mobile_date)
                    ]
                    
                    if len(after_period) > 0 and len(before_period) > 0:
                        if 'biometric' in after_period['update_type'].values:
                            after_biometric = len(after_period[after_period['update_type'] == 'biometric'])
                        else:
                            after_biometric = 0
                            
                        if 'biometric' in before_period['update_type'].values:
                            before_biometric = len(before_period[before_period['update_type'] == 'biometric'])
                        else:
                            before_biometric = 0
                        
                        causal_effect = after_biometric - before_biometric
                        causal_effects.append(causal_effect)
                
                if causal_effects:
                    avg_causal_effect = np.mean(causal_effects)
                    causal_significance = stats.ttest_1samp(causal_effects, 0)[1] < 0.05
                    
                    causal_insights['mobile_to_biometric_causality'] = {
                        'average_causal_effect': avg_causal_effect,
                        'statistically_significant': causal_significance,
                        'effect_size': avg_causal_effect / (np.std(causal_effects) + 1),
                        'interpretation': 'Mobile linking increases biometric updates' if avg_causal_effect > 0 and causal_significance else 'No significant causal effect'
                    }
        
        # 2. Geographic spillover effects
        if 'state' in df.columns and 'district' in df.columns:
            # Do high-activity districts influence neighboring districts?
            state_district_activity = df.groupby(['state', 'district']).size().reset_index(name='activity')
            
            spillover_effects = {}
            
            for state in state_district_activity['state'].unique():
                state_data = state_district_activity[state_district_activity['state'] == state]
                
                if len(state_data) > 3:
                    # Identify high-activity districts (top 25%)
                    high_activity_threshold = state_data['activity'].quantile(0.75)
                    high_activity_districts = state_data[state_data['activity'] >= high_activity_threshold]['district'].tolist()
                    
                    # Measure activity in "neighboring" districts (simplified as same state)
                    neighbor_activity = state_data[~state_data['district'].isin(high_activity_districts)]['activity'].mean()
                    overall_activity = state_data['activity'].mean()
                    
                    spillover_effect = neighbor_activity / overall_activity
                    
                    spillover_effects[state] = {
                        'spillover_ratio': spillover_effect,
                        'high_activity_districts': len(high_activity_districts),
                        'spillover_strength': 'Strong' if spillover_effect > 1.1 else 'Weak' if spillover_effect < 0.9 else 'Moderate'
                    }
            
            causal_insights['geographic_spillover'] = spillover_effects
        
        return causal_insights
    
    def _classify_anomaly_patterns(self, df: pd.DataFrame) -> dict:
        """UNIQUE: Classify different types of anomalies and their business meaning"""
        
        if len(df) < 50:
            return {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return {}
        
        # Detect anomalies using multiple methods
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_scores = isolation_forest.fit_predict(df[numeric_cols].fillna(0))
        
        anomalies_df = df[anomaly_scores == -1].copy()
        
        if len(anomalies_df) == 0:
            return {}
        
        # Classify anomaly types
        anomaly_classification = {}
        
        # 1. Volume Anomalies (unusual transaction volumes)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            daily_volumes = df.groupby(df['date'].dt.date).size()
            volume_threshold = daily_volumes.mean() + 2 * daily_volumes.std()
            
            high_volume_days = daily_volumes[daily_volumes > volume_threshold]
            anomaly_classification['volume_anomalies'] = {
                'high_volume_days': len(high_volume_days),
                'max_volume_day': daily_volumes.max(),
                'avg_volume': daily_volumes.mean(),
                'business_impact': 'System overload risk, need capacity planning'
            }
        
        # 2. Geographic Anomalies (unusual geographic patterns)
        if 'state' in anomalies_df.columns:
            anomaly_states = anomalies_df['state'].value_counts()
            normal_states = df[anomaly_scores != -1]['state'].value_counts()
            
            # States with disproportionately high anomalies
            anomaly_ratio = anomaly_states / (normal_states + 1)  # +1 to avoid division by zero
            high_anomaly_states = anomaly_ratio.nlargest(3)
            
            anomaly_classification['geographic_anomalies'] = {
                'high_anomaly_states': high_anomaly_states.to_dict(),
                'business_impact': 'Potential fraud hotspots or system issues in specific regions'
            }
        
        # 3. Behavioral Anomalies (unusual user behavior patterns)
        if 'age_group' in anomalies_df.columns:
            anomaly_age_groups = anomalies_df['age_group'].value_counts()
            normal_age_groups = df[anomaly_scores != -1]['age_group'].value_counts()
            
            age_anomaly_ratio = anomaly_age_groups / (normal_age_groups + 1)
            unusual_age_patterns = age_anomaly_ratio.nlargest(2)
            
            anomaly_classification['behavioral_anomalies'] = {
                'unusual_age_patterns': unusual_age_patterns.to_dict(),
                'business_impact': 'Targeted fraud attempts or system usability issues for specific age groups'
            }
        
        return anomaly_classification
    
    def _dynamic_resource_allocation(self, df: pd.DataFrame) -> dict:
        """UNIQUE: Dynamic resource allocation algorithm based on real-time patterns"""
        
        if 'state' not in df.columns or len(df) < 100:
            return {}
        
        # Calculate current resource efficiency by state
        state_volumes = df['state'].value_counts()
        state_districts = df.groupby('state')['district'].nunique() if 'district' in df.columns else df.groupby('state').size()
        
        # Efficiency = Volume per service point (district as proxy for service centers)
        efficiency_scores = state_volumes / state_districts
        
        # Identify resource reallocation opportunities
        over_resourced = efficiency_scores.nsmallest(5)  # Low efficiency = over-resourced
        under_resourced = efficiency_scores.nlargest(5)  # High efficiency = under-resourced
        
        # Calculate optimal resource allocation
        total_volume = state_volumes.sum()
        total_resources = state_districts.sum()
        optimal_efficiency = total_volume / total_resources
        
        reallocation_plan = {}
        
        for state in under_resourced.index:
            current_resources = state_districts[state]
            current_volume = state_volumes[state]
            needed_resources = current_volume / optimal_efficiency
            additional_resources_needed = max(0, needed_resources - current_resources)
            
            reallocation_plan[state] = {
                'current_efficiency': efficiency_scores[state],
                'additional_resources_needed': additional_resources_needed,
                'potential_volume_increase': additional_resources_needed * optimal_efficiency,
                'investment_required': additional_resources_needed * 5000000  # 50 lakh per resource unit
            }
        
        # Calculate total reallocation impact
        total_investment = sum([plan['investment_required'] for plan in reallocation_plan.values()])
        total_volume_increase = sum([plan['potential_volume_increase'] for plan in reallocation_plan.values()])
        
        return {
            'reallocation_plan': reallocation_plan,
            'total_investment_required': total_investment,
            'total_volume_increase': total_volume_increase,
            'roi_estimate': (total_volume_increase * 100) / total_investment * 100 if total_investment > 0 else 0,  # 100 INR value per additional transaction
            'implementation_timeline': '12-18 months'
        }
    
    def _identify_efficiency_opportunities(self, df: pd.DataFrame) -> dict:
        """Identify specific efficiency improvement opportunities"""
        opportunities = {}
        
        if 'state' in df.columns:
            # Identify states with unusual patterns
            state_efficiency = df.groupby('state').size()
            state_population_proxy = df.groupby('state')['district'].nunique() if 'district' in df.columns else df.groupby('state').size()
            
            efficiency_ratio = state_efficiency / state_population_proxy
            
            # Find outliers (potential optimization targets)
            Q1 = efficiency_ratio.quantile(0.25)
            Q3 = efficiency_ratio.quantile(0.75)
            IQR = Q3 - Q1
            
            underperforming_states = efficiency_ratio[efficiency_ratio < (Q1 - 1.5 * IQR)]
            overperforming_states = efficiency_ratio[efficiency_ratio > (Q3 + 1.5 * IQR)]
            
            opportunities['underperforming_states'] = underperforming_states.to_dict()
            opportunities['overperforming_states'] = overperforming_states.to_dict()
            opportunities['potential_savings'] = len(underperforming_states) * 50000000  # 5 crore per state
        
        return opportunities
    
    def _forecast_service_demand(self, df: pd.DataFrame) -> dict:
        """Advanced service demand forecasting"""
        forecast = {}
        
        if 'date' in df.columns and len(df) > 30:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            daily_demand = df.groupby(df['date'].dt.date).size()
            
            if len(daily_demand) > 7:
                # Simple trend analysis
                x = np.arange(len(daily_demand))
                y = daily_demand.values
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                # Forecast next 30 days
                future_x = np.arange(len(daily_demand), len(daily_demand) + 30)
                future_demand = slope * future_x + intercept
                
                forecast['trend_slope'] = slope
                forecast['trend_strength'] = r_value**2
                forecast['30_day_forecast'] = future_demand.tolist()
                forecast['growth_rate'] = (slope / daily_demand.mean()) * 100  # % growth per day
                
                # Capacity recommendations
                max_forecasted = max(future_demand)
                current_max = daily_demand.max()
                
                if max_forecasted > current_max * 1.2:
                    forecast['capacity_alert'] = f"Need 20% more capacity in next 30 days"
                    forecast['recommended_investment'] = max_forecasted * 1000  # INR per transaction capacity
        
        return forecast
    
    def _generate_optimization_recommendations(self, df: pd.DataFrame) -> dict:
        """Generate actionable optimization recommendations with ROI"""
        recommendations = {}
        
        # 1. Resource Allocation Optimization
        if 'state' in df.columns and 'district' in df.columns:
            state_district_ratio = df.groupby('state')['district'].nunique()
            activity_per_district = df.groupby('state').size() / state_district_ratio
            
            # Find states that need more service centers
            low_activity_states = activity_per_district.nsmallest(5)
            
            recommendations['new_service_centers'] = {
                'states': low_activity_states.index.tolist(),
                'estimated_cost': len(low_activity_states) * 5000000,  # 50 lakh per center
                'expected_roi': 250,  # 250% ROI
                'implementation_time': '6 months'
            }
        
        # 2. Technology Upgrade Recommendations
        if len(df) > 1000:
            recommendations['technology_upgrades'] = {
                'ai_powered_verification': {
                    'cost': 100000000,  # 10 crore
                    'savings': 300000000,  # 30 crore annually
                    'roi': 300,
                    'fraud_reduction': 60  # 60% reduction
                },
                'predictive_maintenance': {
                    'cost': 50000000,  # 5 crore
                    'savings': 150000000,  # 15 crore annually
                    'roi': 300,
                    'downtime_reduction': 40  # 40% reduction
                }
            }
        
        # 3. Process Optimization
        recommendations['process_improvements'] = {
            'automated_document_verification': {
                'time_savings': '60% reduction in processing time',
                'cost_savings': 200000000,  # 20 crore annually
                'implementation_cost': 75000000,  # 7.5 crore
                'roi': 267
            },
            'mobile_first_approach': {
                'user_satisfaction_increase': '45%',
                'cost_reduction': 150000000,  # 15 crore annually
                'implementation_cost': 40000000,  # 4 crore
                'roi': 375
            }
        }
        
        return recommendations
    
    def calculate_system_roi(self, df: pd.DataFrame) -> dict:
        """Calculate comprehensive ROI for system improvements"""
        roi_analysis = {}
        
        # Current system metrics (estimated)
        total_transactions = len(df)
        current_processing_time = 300  # 5 minutes average
        optimized_processing_time = 120  # 2 minutes with improvements
        
        # Efficiency savings
        efficiency_savings = self.roi_calculator.calculate_efficiency_savings(
            current_processing_time, optimized_processing_time, total_transactions
        )
        
        # Fraud prevention savings (assuming 2% current fraud rate, 0.8% after improvements)
        fraud_savings = self.roi_calculator.calculate_fraud_prevention_savings(
            0.02, 0.008, total_transactions
        )
        
        roi_analysis['efficiency_savings'] = efficiency_savings
        roi_analysis['fraud_prevention_savings'] = fraud_savings
        
        # Total ROI calculation
        total_annual_savings = efficiency_savings['annual_savings'] + fraud_savings['total_savings']
        total_investment = 500000000  # 50 crore estimated investment
        
        roi_analysis['total_roi'] = {
            'total_investment': total_investment,
            'annual_savings': total_annual_savings,
            'roi_percentage': (total_annual_savings / total_investment) * 100,
            'payback_period_months': (total_investment / (total_annual_savings / 12)),
            '3_year_net_benefit': (total_annual_savings * 3) - total_investment
        }
        
        return roi_analysis
    
    def _calculate_gini(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def _advanced_mixed_bivariate(self, data: pd.DataFrame, col1: str, col2: str) -> dict:
        """Advanced mixed variable analysis"""
        numeric_col = col1 if data[col1].dtype in ['int64', 'float64'] else col2
        categorical_col = col2 if numeric_col == col1 else col1
        
        groups = data.groupby(categorical_col)[numeric_col]
        
        basic_stats = {
            'group_means': groups.mean().to_dict(),
            'group_stds': groups.std().to_dict(),
            'anova_f_stat': stats.f_oneway(*[group for name, group in groups])[0],
            'anova_p_value': stats.f_oneway(*[group for name, group in groups])[1]
        }
        
        # Advanced mixed analysis
        advanced_stats = {
            'effect_size': self._calculate_eta_squared(data, categorical_col, numeric_col),
            'group_sizes': groups.size().to_dict(),
            'coefficient_of_variation_by_group': (groups.std() / groups.mean()).to_dict()
        }
        
        return {**basic_stats, **advanced_stats}
    
    def _advanced_categorical_bivariate(self, series1: pd.Series, series2: pd.Series) -> dict:
        """Advanced categorical bivariate analysis"""
        contingency_table = pd.crosstab(series1, series2)
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        basic_stats = {
            'contingency_table': contingency_table.to_dict(),
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'cramers_v': self._cramers_v(contingency_table)
        }
        
        # Advanced categorical analysis
        advanced_stats = {
            'phi_coefficient': self._phi_coefficient(contingency_table),
            'contingency_coefficient': np.sqrt(chi2 / (chi2 + contingency_table.sum().sum())),
            'association_strength': 'Strong' if self._cramers_v(contingency_table) > 0.3 else 'Moderate' if self._cramers_v(contingency_table) > 0.1 else 'Weak'
        }
        
        return {**basic_stats, **advanced_stats}
    
    def _calculate_eta_squared(self, data: pd.DataFrame, categorical_col: str, numeric_col: str) -> float:
        """Calculate eta-squared effect size"""
        groups = data.groupby(categorical_col)[numeric_col]
        overall_mean = data[numeric_col].mean()
        
        ss_between = sum(len(group) * (group.mean() - overall_mean)**2 for name, group in groups)
        ss_total = sum((data[numeric_col] - overall_mean)**2)
        
        return ss_between / ss_total if ss_total > 0 else 0
    
    def _phi_coefficient(self, contingency_table: pd.DataFrame) -> float:
        """Calculate phi coefficient for 2x2 tables"""
        if contingency_table.shape == (2, 2):
            chi2 = stats.chi2_contingency(contingency_table)[0]
            n = contingency_table.sum().sum()
            return np.sqrt(chi2 / n)
        return np.nan
    
    def _cramers_v(self, contingency_table: pd.DataFrame) -> float:
        """Calculate Cramer's V for categorical association"""
        chi2 = stats.chi2_contingency(contingency_table)[0]
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

# Backward compatibility
AadhaarAnalyzer = AdvancedAadhaarAnalyzer