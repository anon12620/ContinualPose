from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='continual_pose',
    version='0.0.1',
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
)
