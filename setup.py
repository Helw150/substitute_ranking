from setuptools import find_packages, setup
from version import Version


setup(name='substitute_ranking',
                 version=Version('0.0.1').number,
                 description='Lexical Substitution Ranking Package',
                 long_description=open('README.md').read().strip(),
                 author='William Held',
                 author_email='me@williamheld.com',
                 url='http://williamheld.com',
                 package_dir={"": "src"},
                 packages=find_packages("src"),
                 install_requires=[],
                 license='MIT License',
                 zip_safe=False,
                 keywords='lexical substitution',
                 classifiers=['NLP'])
