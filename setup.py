from setuptools import setup, find_packages
from typing import List

HYPER_E_DOT='-e .'
def get_requirements(file_path:str)->list[str]:
    '''
    this function will return requirements
    '''

    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPER_E_DOT in requirements:
            requirements.remove(HYPER_E_DOT)

    return requirements
    


setup(
    name='end_to_end_ml_project',               # Name of your project
    version='0.1.0',                            # Version number
    author='Durgesh-06',                       # Your name or username
    author_email='durgeshshirsath03@gmail.com',       # Your email
    description='An end-to-end machine learning project for model training and prediction.',
    long_description=open('README.md').read(),  # Optional: reads README content
    long_description_content_type='text/markdown',
    url='https://github.com/professorDS/end-to-end-ML-project',  # Your GitHub repo
    packages=find_packages(),                   # Finds all Python packages automatically
    install_requires=get_requirements('requirements.txt')                         # Dependencies required
        

    )
