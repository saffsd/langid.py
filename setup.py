from setuptools import setup, find_packages
import sys, os

version = '1.0'

setup(name='langid',
      version=version,
      description="langid.py is a standalone Language Identification (LangID) tool.",
      long_description="""\
""",
      classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='language detection',
      author='Marco Lui',
      author_email='saffsd@gmail.com',
      url='https://github.com/saffsd/langid.py',
      license='BSD',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          # -*- Extra requirements: -*-
          'numpy',
      ],
      entry_points="""
      # -*- Entry points: -*-
      """,
      )
