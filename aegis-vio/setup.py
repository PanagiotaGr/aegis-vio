from setuptools import setup, find_packages
setup(
    name="aegis_vio",
    version="1.0.0",
    description="Uncertainty-Aware Visual-Inertial Navigation for Autonomous Drones",
    author="Panagiota Gr",
    author_email="research@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "opencv-python>=4.7.0",
        "opencv-contrib-python>=4.7.0",
        "torch>=2.0.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "transforms3d>=0.4.1",
    ],
    extras_require={
        "dev": ["pytest>=7.3.0", "pytest-cov>=4.0.0"],
        "ros2": ["rclpy"],
    },
    entry_points={
        "console_scripts": [
            "aegis-vio=scripts.run_euroc:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
