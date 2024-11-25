from setuptools import setup, find_packages

setup(
    name="trading_strategy",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Core Data Processing
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        
        # Financial Data
        "yfinance>=0.1.63",
        "pandas-datareader>=0.10.0",
        
        # Machine Learning & Statistics
        "scikit-learn>=0.24.0",
        "arch>=5.0.0",
        "hmmlearn>=0.3.3",
        
        # Visualization
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        
        # Progress and Utils
        "tqdm>=4.62.0",
        "ipython>=7.0.0",
        "jupyter>=1.0.0",
    ],
    extras_require={
        'dev': [
            # Testing
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'pytest-mock>=3.6.1',
            
            # Code Quality
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.910',
            'isort>=5.10.0',
            
            # Development
            'ipykernel>=6.0.0',
            'pre-commit>=2.17.0',
        ]
    },
    author="Lucas",
    author_email="your.email@example.com",
    description="A sophisticated trading strategy system with Monte Carlo simulation",
    keywords="trading, monte-carlo, financial-analysis, risk-management",
    python_requires=">=3.8",
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)