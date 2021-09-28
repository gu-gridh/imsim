# -*- coding: UTF-8 -*
from setuptools import setup

setup(name='imsim',
      version='1.0',
      description='An application generating an image similarity index and 2D projection',
      url='http://github.com/CDH-DevTeam/imsim',
      author='Victor Wåhlstrand Skärström',
      license='MIT',
      packages=['imsim'],
      entry_points="""
        [console_scripts]
        imsim = imsim.__main__:main
        """,
      zip_safe=False,
      install_requires=['tensorflow',
                        'tensorflow_hub',
                        'numpy',
                        'umap-learn',
                        'scikit-learn',
                        'pyyaml',
                        'annoy',
                        'tqdm'])