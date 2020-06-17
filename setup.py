#nsml: nsml/ml:cuda10.1-cudnn7-pytorch1.3keras2.3
from distutils.core import setup

setup(
    name='Category Match training on NSML',
    version='0.1',
    description='Category Match training ',
    install_requires=[
        'numpy',
        'minio',
        'pytorch_transformers',
        'torchvision',
    ],
)