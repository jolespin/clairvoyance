from setuptools import setup

# Version
version = None
with open("./clairvoyance/__init__.py", "r") as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith("__version__"):
            version = line.split("=")[-1].strip().strip('"')
assert version is not None, "Check version in clairvoyance/__init__.py"

setup(name='clairvoyance_feature_selection',
      version=version,
      description='Feature selection via recursive feature inclusion',
      url='https://github.com/jolespin/clairvoyance',
      author='Josh L. Espinoza',
      author_email='jespinoz@jcvi.org',
      license='BSD-3',
      packages=["clairvoyance"],
      install_requires=[
        'pandas >= 1.2.4',
        "numpy >= 1.13",
        "xarray >= 0.10.3",
        "matplotlib >= 2",
        "seaborn >= 0.10.1",
        "scipy >= 1.0",
        "scikit-learn >= 1.0",
        "soothsayer_utils >= 2022.6.24",
        "tqdm >=4.19",
      ],
    include_package_data=True,
    scripts=['bin/clairvoyance_v1.py'],

)