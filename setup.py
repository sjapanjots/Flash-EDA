from setuptools import setup, find_packages

setup(
    name="flasheda",
    version="0.2.0",
    description="Constant-time EDA for any dataset size — one line, fixed time.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Japanjot Singh",
    author_email="sjapanjots@gmail.com",
    url="https://github.com/sjapanjots/Flash-EDA",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "rich>=12.0.0",
        "jinja2>=3.0.0",
        "matplotlib>=3.4.0",
        "fpdf2>=2.7.0",
    ],
    extras_require={
        "fast": ["duckdb>=0.10", "polars>=0.20", "pyarrow>=12"],
        "spark": ["pyspark>=3.4"],
        "dev": ["pytest>=7", "duckdb>=0.10", "polars>=0.20", "pyarrow>=12"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)