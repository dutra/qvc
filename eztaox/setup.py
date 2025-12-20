from setuptools import setup, find_packages

setup(
    name="eztaox",
    version="0.1.0",
    description="A package for multiband quasar light curve modeling and analysis.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/eztaox",  # Replace with your repository URL
    packages=find_packages(),
    install_requires=[
        "numpy",
        "jax",
        "jaxlib",
        "numpyro",
        "tinygp",
        "optax",
        "equinox",
        "astropy",
        "pandas",
        "matplotlib",
        "scipy",
        "tqdm",
        "h5py",
        "blackjax",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "eztaox=eztaox.main:main",  # Replace `eztaox.main:main` with the actual entry point
        ],
    },
)