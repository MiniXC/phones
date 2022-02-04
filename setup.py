
# -*- coding: utf-8 -*-
from setuptools import setup

long_description = None
INSTALL_REQUIRES = [
    'pandas>=0.25.3',
    'numpy>=1.19.5',
]

setup_kwargs = {
    'name': 'phones',
    'version': '0.0.1',
    'description': 'A collection of utilities for handling IPA phones.',
    'long_description': long_description,
    'license': 'MIT',
    'author': '',
    'author_email': 'Christoph Minixhofer <christoph.minixhofer@gmail.com>',
    'maintainer': None,
    'maintainer_email': None,
    'url': '',
    'packages': [
        'phones',
    ],
    'package_dir': {'': 'src'},
    'package_data': {'': ['*']},
    'install_requires': INSTALL_REQUIRES,
    'python_requires': '>=3.6',

}


setup(**setup_kwargs)
