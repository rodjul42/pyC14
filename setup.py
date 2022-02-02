from distutils.core import setup
from setuptools import setup, find_packages
#This is a list of files to install, and where
#(relative to the 'root' dir, where setup.py is)
#You could be more specific.
files = ["c14/*"]

setup(name = "c14",
    version = "100",
    description = "yadda yadda",
    author = "myself and I",
    author_email = "email@someplace.com",
    url = "whatever",
    include_package_data=True,
    packages = ['c14'],
    package_dir={'c14': 'c14'},
    package_data = {'c14' : ['data/kudryavtsev_et_al_1993_table_2.xlsx'] },
    #'runner' is in the root.
    #scripts = ["runner"],
    long_description = """Really long text here.""",
    install_requires=['numpy>=1.18.1','scipy>=1.4.1','pandas>=0.25.3','emcee>=3.0.2','arviz>=0.11.2','matplotlib','xlrd','rpy2']
    #
    #This next part it for the Cheese Shop, look a little down the page.
    #classifiers = []
)
