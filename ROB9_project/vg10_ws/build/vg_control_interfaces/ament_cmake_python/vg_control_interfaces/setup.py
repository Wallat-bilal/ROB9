from setuptools import find_packages
from setuptools import setup

setup(
    name='vg_control_interfaces',
    version='0.0.0',
    packages=find_packages(
        include=('vg_control_interfaces', 'vg_control_interfaces.*')),
)
