from pathlib import Path

from setuptools import setup, find_packages

setup(
    name="flasheda",
    version="0.4.0",
    description=(
        "Fixed-time EDA for any dataset size, plus validated cleaning, feature "
        "and training code for pandas, polars, DuckDB and PySpark."
    ),
    long_description=Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Japanjot Singh",
    author_email="sjapanjots@gmail.com",
    url="https://github.com/sjapanjots/Flash-EDA",
    project_urls={
        "Source": "https://github.com/sjapanjots/Flash-EDA",
        "Changelog": "https://github.com/sjapanjots/Flash-EDA/blob/main/CHANGELOG.md",
        "Issues": "https://github.com/sjapanjots/Flash-EDA/issues",
    },
    license="MIT",
    license_files=["MIT.License"],
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.22",
        "scipy>=1.7.0",
        "rich>=12.0.0",
        "jinja2>=3.0.0",
        "matplotlib>=3.4.0",
        "fpdf2>=2.7.0",
    ],
    extras_require={
        "fast": ["duckdb>=0.10", "polars>=0.20", "pyarrow>=12"],
        "spark": ["pyspark>=3.4"],
        "ml": ["scikit-learn>=1.2", "pyarrow>=12"],
        "all": ["duckdb>=0.10", "polars>=0.20", "pyarrow>=12", "scikit-learn>=1.2"],
        "dev": ["pytest>=7", "duckdb>=0.10", "polars>=0.20", "pyarrow>=12", "scikit-learn>=1.2",
                "build", "twine"],
    },
    keywords=["eda", "data-cleaning", "pandas", "polars", "duckdb", "pyspark", "code-generation", "feature-engineering", "automl"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
