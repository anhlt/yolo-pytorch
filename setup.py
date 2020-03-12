from setuptools import setup
import setuptools

setup(
    name='yolo',
    version='0.2',
    packages=setuptools.find_packages(),
    license='MIT',
    package_data={
      'yolo': ['*.json', '*.pyi'],
    },
    include_package_data=True,
    long_description=open('README.MD').read(),
)