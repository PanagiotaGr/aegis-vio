from setuptools import find_packages, setup

setup(
    name="aegis-vio",
    version="0.1.0",
    description="Uncertainty-aware visual-inertial navigation for autonomous robots",
    author="Panagiota Grosdouli",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "opencv-python",
        "matplotlib",
        "scipy",
        "pyyaml",
        "tqdm",
    ],
)
