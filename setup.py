"""
Setup script for Aadhaar Data Analysis Project
"""

from setuptools import setup, find_packages

setup(
    name="aadhaar-data-analysis",
    version="1.0.0",
    description="Data-driven innovation analysis for Aadhaar Hackathon 2026",
    author="Your Team Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.13.0",
        "xgboost>=1.7.0",
        "lightgbm>=4.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "bokeh>=3.2.0",
        "statsmodels>=0.14.0",
        "pingouin>=0.5.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.25.0",
        "notebook>=7.0.0",
        "openpyxl>=3.1.0",
        "xlrd>=2.0.0",
        "requests>=2.31.0",
        "geopandas>=0.13.0",
        "folium>=0.14.0",
        "prophet>=1.1.0",
        "pmdarima>=2.0.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)