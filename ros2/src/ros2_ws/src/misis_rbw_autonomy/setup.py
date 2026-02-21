from glob import glob
import os

from setuptools import find_packages, setup


package_name = "misis_rbw_autonomy"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="misis_arbuzi",
    maintainer_email="ropraname@users.noreply.github.com",
    description="Autonomous strategy for MISIS Robotics Week mobile ROS2 robot hackathon",
    license="Apache-2.0",
    extras_require={
        "test": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "autonomy_node = misis_rbw_autonomy.autonomy_node:main",
        ],
    },
)
