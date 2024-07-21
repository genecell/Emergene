from setuptools import setup, find_packages

setup(
    name='Emergene',
    version='0.1.0',
    author='Min Dai',
    description='Individual cell-based differential transcriptomic analysis across conditions wth Emergene',
    packages=find_packages(),
    install_requires=[
        # list any dependencies your package requires here
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)