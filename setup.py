from setuptools import setup, find_packages

from os import path

script_directory = path.abspath(path.dirname(__file__))

package_name = "clairvoyance"
version = None
with open(path.join(script_directory, package_name, '__init__.py')) as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith("__version__"):
            version = line.split("=")[-1].strip().strip('"')
assert version is not None, f"Check version in {package_name}/__init__.py"

with open(path.join(script_directory, 'README.md')) as f:
    long_description = f.read()

requirements = list()
with open(path.join(script_directory, 'requirements.txt')) as f:
    for line in f.readlines():
        line = line.strip()
        if line:
            if not line.startswith("#"):
                requirements.append(line)


setup(name='clairvoyance_feature_selection',
      version=version,
      description='AutoML simultaneous bayesian hyperparameter optimization and feature selection',
      url='https://github.com/jolespin/clairvoyance',
      author='Josh L. Espinoza',
      author_email='jol.espinoz@gmail.com',
      license='BSD-3',
      packages=["clairvoyance"],
      install_requires=requirements[::-1],
    scripts=['bin/clairvoyance_v1.py'],

)