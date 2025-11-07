from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lcgen",
    version="0.1.0",
    author="LC-Gen Team",
    description="Light Curve Generation and Reconstruction using Multi-Modal Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pvl19/lc-gen",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "tapify @ git+https://github.com/aaryapatil/tapify.git",
        "scipy>=1.10.0",
        "nfft>=0.1",  # Required by tapify
        "statsmodels>=0.14.0",  # Required by tapify
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
        ],
    },
)
