from setuptools import setup, find_packages

VERSION = '0.1.0'
DESCRIPTION = 'Base Training'
LONG_DESCRIPTION = ''

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="pytorch_base",
    version=VERSION,
    author="Luis Barba",
    author_email="<luis.barba-flores@psi.ch>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=[]  # add any additional packages that
)