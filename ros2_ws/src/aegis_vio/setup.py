from setuptools import setup

package_name = "aegis_vio"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Panagiota Grosdouli",
    maintainer_email="p.g2a15@gmail.com",
    description="ROS2 nodes for AegisVIO uncertainty-aware visual-inertial navigation.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "vio_node = aegis_vio.vio_node:main",
            "navigation_node = aegis_vio.navigation_node:main",
            "uncertainty_node = aegis_vio.uncertainty_node:main",
        ],
    },
)
