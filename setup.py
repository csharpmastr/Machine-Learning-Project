import re
from setuptools import find_packages, setup
from typing import List


hypen_e_dot = '-e .'

# function to get all package requirements
def get_requirements(file_path:str)->List[str]:
    """_summary_

    Args:
        file_path (str): file path of the requirements.txt file

    Returns:
        List[str]: the packages needed for development
    """
    
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n", "") for req in requirements]
        
        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)
    
    return requirements

setup(
    name='Labnostic Machine Learning Project',
    version='0.0.1',
    author='csharpmastr',
    author_email='csharp.mastr@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)