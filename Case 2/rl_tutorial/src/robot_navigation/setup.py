from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robot_navigation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'params'), glob('params/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='iago gomes',
    maintainer_email='iagogomes@usp.br',
    description='Robot Navigation Package with RL',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            "train_sac    = robot_navigation.scripts.train_sac:main",
            "eval_sac     = robot_navigation.scripts.eval_sac:main",
            "plot_metrics = robot_navigation.scripts.plot_metrics:main",
        ],
    },
)
