import os
import setuptools
import sys


# Load README to get long description.
with open('README.md') as f:
    _LONG_DESCRIPTION = f.read()


setuptools.setup(
    name='autoprompt',
    version='0.0.1',
    description='AutoPrompt',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='UCI NLP',
    url='https://github.com/ucinlp/autoprompt',
    packages=setuptools.find_packages(),
    install_requires=[ ],
    extras_require={
        'test': ['pytest']
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='text nlp machinelearning',
)
