"""
Point in Polygon package for efficient spatial operations using GeoPandas and Oracle Spatial database.
"""

from point_in_polygon.main import PointInPolygonConfig, point_in_polygon
from point_in_polygon.setup_keyring import setup_keyring

__all__ = ['PointInPolygonConfig', 'point_in_polygon', 'setup_keyring']

__version__ = '0.1.0'
