from setuptools import find_packages, setup
from glob import glob
import os 

package_name = 'robot_gazebo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'models'), glob('models/*.sdf')),
        (os.path.join("lib", package_name), ["scripts/teleop"]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='iago gomes',
    maintainer_email='iagogomes@usp.br',
    description='Robot Simulation Package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
        ],
    },
)
