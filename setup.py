from setuptools import setup, find_packages

setup(
    name='argument_relation_transformer',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch==1.6.0',
        'pytorch-lightning==1.5.0',
        'transformers==4.30.0',
        'numpy==1.21.6',
        'scikit-learn==1.0.2',
    ]
)
