from setuptools import setup

setup(name='sigclust',
      version='0.1',
      description='Python SigClust',
      url='http://github.com/thomaskeefe/SigClust',
      author='Thomas Keefe',
      author_email='tkeefe@live.unc.edu',
      packages=['sigclust'],
      install_requires=[
        'scipy',
        'numpy',
        'pandas',
        'scikit-learn',
        'numba'
      ])
