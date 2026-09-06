"""Pure helpers, imported by module path.

Nothing is re-exported here.  Each module in this package reads whatever it is
handed -- a Lanelet2 map, an OpenDRIVE road network -- and the ambient forms,
which fetch those from the :class:`~..coordinate.map_manager.MapManager`
singleton, live in :mod:`..coordinate` alongside it.  Re-exporting the pure
names beside the ambient ones would put one name on two signatures.
"""
