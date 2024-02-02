'''
Read version from installed package.
'''


from importlib.metadata import version

__version__ = version('cellxhaustive')
# AT. Double-check: sometimes causing a problem when importing from the package root dir
