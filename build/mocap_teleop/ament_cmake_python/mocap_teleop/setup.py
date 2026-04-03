from setuptools import find_packages
from setuptools import setup

setup(
    name='mocap_teleop',
    version='0.0.1',
    packages=find_packages(
        include=('mocap_teleop', 'mocap_teleop.*')),
)
