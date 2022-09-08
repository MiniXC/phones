
# -*- coding: utf-8 -*-
from setuptools import setup

long_description = None
INSTALL_REQUIRES = [
    'pandas>=0.25.3',
    'numpy>=1.19.5',
    'tqdm>=4.62.3',
    'scipy>=1.5.4',
]
EXTRAS_REQUIRE = {
    'plots': [
        'plotly>=5.5.0',
        'scikit-learn>=0.24.2',
    ],
    'test': [
        'pytest>=7.0.0',
        'pytest-cov>=3.0.0',
    ],
}

setup_kwargs = {
    'name': 'phones',
    'version': '0.0.4',
    'description': 'A collection of utilities for handling IPA phones.',
    'long_description': long_description,
    'license': 'MIT',
    'author': '',
    'author_email': 'Christoph Minixhofer <christoph.minixhofer@gmail.com>',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://cdminix.me/phones',
    'packages': [
        'phones',
        'phones.phonecodes.src',
    ],
    'package_dir': {'': 'src'},
    'package_data': {'': ['*']},
    'install_requires': INSTALL_REQUIRES,
    'extras_require': EXTRAS_REQUIRE,
    'python_requires': '>=3.6',

}


setup(**setup_kwargs)
