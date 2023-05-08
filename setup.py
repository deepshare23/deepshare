from setuptools import find_packages, setup

""" Installs DLCM packages 
Current `dlcm_packages` includes (2022.12.21)
- 'profiler', 'utils', 'profiler.scripts', 'profiler.configs', 'profiler.scripts.parser'

Run `pip install -e .` from where `setup.py` is located($DLCM_HOME) once, then `dlcm_packages` will be installed in 
editable mode. Editable mode deploys packages locally and only **links** package source code to `site-packages` 
without copy, thus additional `pip install` is not needed when the package source codes change.
"""
dlcm_packages = find_packages()
setup(
   name='dlcm',
   version='0.0',
   description='A resource sensitive elastic cluster manager',
   author='Junyeol Ryu',
   author_email='gajagajago@snu.ac.kr',
   packages=find_packages(),
   python_requires  = '>=3.9',
   install_requires=[
      # Add package only if DLCM, Slurm, or Hadoop needs it. 
      # Packages needed for misc benchmarks or scripts should be installed at execution time, 
      # with `pip install -r requirements.txt`
      'pip==23.0.1',
      'hdfs',
      'nvidia-cublas-cu11==11.10.3.66',
      'nvidia-cuda-nvrtc-cu11==11.7.99',
      'nvidia-cuda-runtime-cu11==11.7.99',
      'nvidia-cudnn-cu11==8.5.0.96',
      'psutil',
      'XlsxWriter',
      'torch==1.12.0',
   ])
