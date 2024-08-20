"""Use Fourier shell correlation to estimate the resolution of cryo-EM images and volumes"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ttfsc")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Johannes Elferich"
__email__ = "jojotux123@hotmail.com"

from .cli import cli