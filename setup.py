from setuptools import find_packages, setup

setup(
    name="trading_strategy",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Core Data Processing
        "numpy>=1.21.0,<2.0.0",
        "pandas>=1.3.0,<2.0.0",
        "scipy>=1.7.0,<2.0.0",
        # Financial Data
        "yfinance>=0.1.63",
        "pandas-datareader>=0.10.0",
        "alpaca-trade-api>=3.0.0",
        # Machine Learning & Statistics
        "scikit-learn>=0.24.0",
        "arch>=5.0.0",
        "hmmlearn>=0.3.3",
        "statsmodels>=0.13.0",
        # Visualization
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "mplfinance>=0.12.9b7",
        "bokeh>=3.0.0",
        # Progress and Utils
        "tqdm>=4.62.0",
        "ipython>=7.0.0",
        "jupyter>=1.0.0",
        "notebook>=6.4.0",
        "jupyterlab>=4.0.0",
        # Async Support
        "aiohttp>=3.8.0",
        "asyncio>=3.4.3",
        # Type Checking and Validation
        "typing-extensions>=4.0.0",
        "pydantic>=2.0.0",
        # Environment and Utils
        "python-dotenv>=0.19.0",
        "pygments>=2.10.0",
    ],
    extras_require={
        "dev": [
            # Testing
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-mock>=3.6.1",
            "pytest-asyncio>=0.18.0",
            "pytest-benchmark>=4.0.0",
            "pytest-xdist>=3.0.0",
            # Code Quality
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
            "isort>=5.10.0",
            "autopep8>=1.6.0",
            "pre-commit>=2.17.0",
            # Documentation
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbconvert>=6.4.0",
            "jupyter-book>=0.15.0",
            "mkdocs>=1.5.0",
            # Project Packaging
            "setuptools>=60.0.0",
            "wheel>=0.37.0",
        ]
    },
    author="Lucas Kemper & Antonio Schoeffel",
    author_email="contact@lucaskemper.com",
    description="A sophisticated trading strategy system with Monte Carlo simulation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="trading, monte-carlo, financial-analysis, risk-management",
    url="https://github.com/lucaskemper/project-datascience-hec",
    python_requires=">=3.8",
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    license="MIT",
)
