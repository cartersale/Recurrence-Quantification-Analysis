from setuptools import setup, find_packages

setup(
    name='rqa_analysis',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
    ],
    description='A package for Recurrence Quantification Analysis (RQA)',
    author='Mike Richardson and Cathy Macpherson',
    author_email='michael.j.richardson@mq.edu.au',
    url='https://github.com/xkiwilabs/Recurrence-Quantification-Analysis',
)
