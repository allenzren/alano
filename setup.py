import setuptools

setuptools.setup(
    name='alano',
    version='1.0',
    description='Custom Python Toolbox',
    author='Allen Z. Ren',
    author_email='allen.ren@princeton.edu',
    packages=setuptools.find_packages(),
    install_requires=['torch', 'numpy'],
)
