import setuptools
from packagename.version import Version


setuptools.setup(name='substitute_ranking',
                 version=Version('0.0.1').number,
                 description='Lexical Substitution Ranking Package',
                 long_description=open('README.md').read().strip(),
                 author='William Held',
                 author_email='me@williamheld.com',
                 url='http://williamheld.com',
                 py_modules=['substitute_ranking'],
                 install_requires=[],
                 license='MIT License',
                 zip_safe=False,
                 keywords='lexical substitution',
                 classifiers=['NLP'])
