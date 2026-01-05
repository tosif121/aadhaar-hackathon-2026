"""
Machine Learning Predictions Page - Predictive modeling and forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
import os
sys.path.append('src')

from api_client import AadhaarAPIClient

st.set_page_config(page_title="ML Predictions", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Predictions")
st.markdown("Predictive modeling and forecasting for Aadhaar data")

# Initialize client
@st.cache_resource
def get_client():
    return AadhaarAPIClient()

client = get_client()

# Sidebar configuration
st.sidebar.header("🎛️ ML Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select ML Model",
    ["Random Forest", "Gradient Boosting", "Linear Regression", "Time Series Forecast"]
)

# Prediction type
prediction_type = st.sidebar.selectbox(
    "Prediction Type",
    ["Enrollment Forecasting", "Update Prediction", "System Load Prediction", "Anomaly Prediction"]
)

# Data configuration
st.sidebar.header("📊 Data Configuration")
sample_size = st.sidebar.slider("Training Data Size", 100, 5000, 100, 100)

# Load data for ML
if st.sidebar.button("🚀 Load ML Data", type="primary"):
    with st.spinner("Loading data for machine learning..."):
        try:
            # Load datasets
            enrolment_df = client.fetch_data('enrolment', limit=sample_size)
            demographic_df = client.fetch_data('demographic', limit=sample_size)
            biometric_df = client.fetch_data('biometric', limit=sample_size)
            
            st.session_state['ml_enrolment'] = enrolment_df
            st.session_state['ml_demographic'] = demographic_df
            st.session_state['ml_biometric'] = biometric_df
            
            if not enrolment_df.empty or not demographic_df.empty or not biometric_df.empty:
                st.sidebar.success("✅ ML data loaded!")
            else:
                st.sidebar.error("❌ No data loaded")
                
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")

# Helper functions for ML
def prepare_features(df):
    """Prepare features for machine learning"""
    if df.empty:
        return pd.DataFrame(), pd.Series()
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return pd.DataFrame(), pd.Series()
    
    # Use first column as target, rest as features
    target_col = numeric_cols[0]
    feature_cols = numeric_cols[1:]
    
    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0)
    
    return X, y

def create_time_features(df, date_col='date'):
    """Create time-based features"""
    if date_col not in df.columns:
        return df
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Extract time features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    return df

def train_model(X, y, model_type):
    """Train machine learning model"""
    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:  # Linear Regression
        model = LinearRegression()
        # Scale features for linear regression
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'y_test': y_test,
        'y_pred': y_pred
    }

# Main content
if ('ml_enrolment' in st.session_state or 
    'ml_demographic' in st.session_state or 
    'ml_biometric' in st.session_state):
    
    # Get loaded data
    enrolment_df = st.session_state.get('ml_enrolment', pd.DataFrame())
    demographic_df = st.session_state.get('ml_demographic', pd.DataFrame())
    biometric_df = st.session_state.get('ml_biometric', pd.DataFrame())
    
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Model Type", model_type)
    with col2:
        st.metric("📊 Prediction Type", prediction_type.split()[0])
    with col3:
        total_records = len(enrolment_df) + len(demographic_df) + len(biometric_df)
        st.metric("📈 Training Records", f"{total_records:,}")
    with col4:
        st.metric("🔮 Status", "Ready" if total_records > 0 else "No Data")
    
    # Model training and prediction
    if prediction_type == "Enrollment Forecasting":
        st.header("📈 Enrollment Forecasting")
        
        if not enrolment_df.empty:
            # Prepare data
            df_with_time = create_time_features(enrolment_df)
            X, y = prepare_features(df_with_time)
            
            if not X.empty and len(y) > 0:
                # Train model
                with st.spinner("Training enrollment forecasting model..."):
                    model, metrics = train_model(X, y, model_type)
                
                # Display metrics
                st.subheader("📊 Model Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RMSE", f"{metrics['rmse']:.2f}")
                with col2:
                    st.metric("MAE", f"{metrics['mae']:.2f}")
                with col3:
                    st.metric("R² Score", f"{metrics['r2']:.3f}")
                with col4:
                    accuracy = max(0, metrics['r2'] * 100)
                    st.metric("Accuracy", f"{accuracy:.1f}%")
                
                # Prediction vs Actual plot
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(
                        x=metrics['y_test'],
                        y=metrics['y_pred'],
                        mode='markers',
                        name='Predictions',
                        marker=dict(color='blue', opacity=0.6)
                    ))
                    
                    # Add perfect prediction line
                    min_val = min(metrics['y_test'].min(), metrics['y_pred'].min())
                    max_val = max(metrics['y_test'].max(), metrics['y_pred'].max())
                    fig_pred.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_pred.update_layout(
                        title="Predicted vs Actual Values",
                        xaxis_title="Actual Values",
                        yaxis_title="Predicted Values"
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                
                with col2:
                    # Residuals plot
                    residuals = metrics['y_test'] - metrics['y_pred']
                    fig_residuals = px.histogram(
                        x=residuals,
                        title="Prediction Residuals Distribution",
                        labels={'x': 'Residuals', 'y': 'Frequency'}
                    )
                    st.plotly_chart(fig_residuals, use_container_width=True)
                
                # Feature importance (for tree-based models)
                if model_type in ["Random Forest", "Gradient Boosting"]:
                    st.subheader("🎯 Feature Importance")
                    
                    feature_importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig_importance = px.bar(
                        feature_importance.head(10),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top 10 Most Important Features"
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                # Future predictions
                st.subheader("🔮 Future Enrollment Forecast")
                
                # Generate future predictions (sample)
                future_months = st.slider("Forecast Months", 1, 12, 6)
                
                # Create sample future data
                last_values = X.iloc[-1:].copy()
                future_predictions = []
                
                for i in range(future_months):
                    pred = model.predict(last_values)[0]
                    future_predictions.append(pred)
                
                # Plot forecast
                historical_data = y.tail(20).values
                forecast_data = future_predictions
                
                fig_forecast = go.Figure()
                
                # Historical data
                fig_forecast.add_trace(go.Scatter(
                    x=list(range(-len(historical_data), 0)),
                    y=historical_data,
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Forecast
                fig_forecast.add_trace(go.Scatter(
                    x=list(range(0, len(forecast_data))),
                    y=forecast_data,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
                
                fig_forecast.update_layout(
                    title=f"Enrollment Forecast - Next {future_months} Months",
                    xaxis_title="Time Period",
                    yaxis_title="Predicted Enrollments"
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
                
            else:
                st.error("❌ Insufficient data for model training")
        else:
            st.info("📊 No enrollment data available for forecasting")
    
    elif prediction_type == "Update Prediction":
        st.header("🔄 Update Prediction")
        
        # Combine demographic and biometric data
        update_data = []
        
        if not demographic_df.empty:
            demo_data = demographic_df.copy()
            demo_data['update_type'] = 'demographic'
            update_data.append(demo_data)
        
        if not biometric_df.empty:
            bio_data = biometric_df.copy()
            bio_data['update_type'] = 'biometric'
            update_data.append(bio_data)
        
        if update_data:
            combined_df = pd.concat(update_data, ignore_index=True)
            
            # Prepare features
            df_with_time = create_time_features(combined_df)
            
            # Encode categorical variables
            le = LabelEncoder()
            if 'update_type' in df_with_time.columns:
                df_with_time['update_type_encoded'] = le.fit_transform(df_with_time['update_type'])
            
            X, y = prepare_features(df_with_time)
            
            if not X.empty and len(y) > 0:
                # Train model
                with st.spinner("Training update prediction model..."):
                    model, metrics = train_model(X, y, model_type)
                
                # Display results
                st.subheader("📊 Update Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Accuracy", f"{max(0, metrics['r2'] * 100):.1f}%")
                with col2:
                    st.metric("RMSE", f"{metrics['rmse']:.2f}")
                with col3:
                    st.metric("MAE", f"{metrics['mae']:.2f}")
                
                # Prediction analysis
                fig_update_pred = px.scatter(
                    x=metrics['y_test'],
                    y=metrics['y_pred'],
                    title="Update Prediction Accuracy",
                    labels={'x': 'Actual Updates', 'y': 'Predicted Updates'}
                )
                st.plotly_chart(fig_update_pred, use_container_width=True)
            else:
                st.error("❌ Insufficient data for update prediction")
        else:
            st.info("📊 No update data available for prediction")
    
    elif prediction_type == "System Load Prediction":
        st.header("⚡ System Load Prediction")
        
        # Simulate system load prediction
        st.info("🚧 System load prediction model in development")
        
        # Create sample system load data
        hours = list(range(24))
        base_load = [30, 25, 20, 18, 20, 25, 40, 60, 80, 85, 90, 95, 100, 95, 90, 85, 80, 70, 60, 50, 45, 40, 35, 32]
        
        # Add some randomness
        predicted_load = [load + np.random.randint(-10, 10) for load in base_load]
        
        fig_load = go.Figure()
        
        fig_load.add_trace(go.Scatter(
            x=hours,
            y=base_load,
            mode='lines+markers',
            name='Historical Load',
            line=dict(color='blue')
        ))
        
        fig_load.add_trace(go.Scatter(
            x=hours,
            y=predicted_load,
            mode='lines+markers',
            name='Predicted Load',
            line=dict(color='red', dash='dash')
        ))
        
        fig_load.update_layout(
            title="24-Hour System Load Prediction",
            xaxis_title="Hour of Day",
            yaxis_title="System Load (%)"
        )
        st.plotly_chart(fig_load, use_container_width=True)
        
        # Load recommendations
        st.subheader("💡 Load Management Recommendations")
        
        peak_hours = [i for i, load in enumerate(predicted_load) if load > 80]
        low_hours = [i for i, load in enumerate(predicted_load) if load < 40]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔴 Peak Load Hours:**")
            for hour in peak_hours:
                st.markdown(f"- {hour:02d}:00 - {hour+1:02d}:00 ({predicted_load[hour]}% load)")
        
        with col2:
            st.markdown("**🟢 Low Load Hours (Maintenance Window):**")
            for hour in low_hours:
                st.markdown(f"- {hour:02d}:00 - {hour+1:02d}:00 ({predicted_load[hour]}% load)")
    
    elif prediction_type == "Anomaly Prediction":
        st.header("🚨 Anomaly Prediction")
        
        st.info("🚧 Anomaly prediction model in development")
        
        # Simulate anomaly prediction
        if not enrolment_df.empty:
            # Create sample anomaly scores
            anomaly_scores = np.random.beta(2, 5, len(enrolment_df))
            anomaly_threshold = 0.8
            
            # Identify anomalies
            anomalies = anomaly_scores > anomaly_threshold
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(enrolment_df))
            with col2:
                st.metric("Detected Anomalies", anomalies.sum())
            with col3:
                st.metric("Anomaly Rate", f"{(anomalies.sum() / len(enrolment_df) * 100):.2f}%")
            
            # Anomaly score distribution
            fig_anomaly = go.Figure()
            
            fig_anomaly.add_trace(go.Histogram(
                x=anomaly_scores,
                name='Anomaly Scores',
                opacity=0.7
            ))
            
            fig_anomaly.add_vline(
                x=anomaly_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text="Anomaly Threshold"
            )
            
            fig_anomaly.update_layout(
                title="Anomaly Score Distribution",
                xaxis_title="Anomaly Score",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig_anomaly, use_container_width=True)
        else:
            st.info("📊 No data available for anomaly prediction")

else:
    st.info("👆 Please load ML data from the sidebar to begin predictive modeling.")
    
    # Show sample ML preview
    st.header("🤖 Sample ML Models Preview")
    
    # Sample model performance comparison
    sample_models = {
        'Model': ['Random Forest', 'Gradient Boosting', 'Linear Regression', 'Neural Network'],
        'Accuracy': [0.85, 0.82, 0.75, 0.88],
        'RMSE': [1250, 1380, 1650, 1180],
        'Training Time': [45, 60, 15, 120]
    }
    
    sample_df = pd.DataFrame(sample_models)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_accuracy = px.bar(
            sample_df,
            x='Model',
            y='Accuracy',
            title="Sample: Model Accuracy Comparison",
            color='Accuracy',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_accuracy, use_container_width=True)
    
    with col2:
        fig_rmse = px.bar(
            sample_df,
            x='Model',
            y='RMSE',
            title="Sample: Model RMSE Comparison",
            color='RMSE',
            color_continuous_scale='Reds_r'
        )
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    # Sample forecast
    st.subheader("📈 Sample Enrollment Forecast")
    
    dates = pd.date_range('2024-01-01', periods=12, freq='M')
    historical = [100000, 105000, 98000, 110000, 115000, 108000]
    forecast = [112000, 118000, 115000, 122000, 125000, 120000]
    
    fig_sample_forecast = go.Figure()
    
    fig_sample_forecast.add_trace(go.Scatter(
        x=dates[:6],
        y=historical,
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue')
    ))
    
    fig_sample_forecast.add_trace(go.Scatter(
        x=dates[6:],
        y=forecast,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    fig_sample_forecast.update_layout(
        title="Sample: 6-Month Enrollment Forecast",
        xaxis_title="Month",
        yaxis_title="Predicted Enrollments"
    )
    st.plotly_chart(fig_sample_forecast, use_container_width=True)