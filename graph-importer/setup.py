from setuptools import setup, find_packages

setup(
    name='graph-importer',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A tool for importing JSON data into a Neo4j graph database.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'neo4j',
        'jsonschema',
        'pandas'
    ],
    entry_points={
        'console_scripts': [
            'graph-importer=main:main',
        ],
    },
)