"""Export a run's ODD samples as an ASAM OpenODD 1.0 COD (openodd's export).

``scenario-coverage --export-cod DIR`` writes, per run, the COD table
(section 8.3), its manifest (8.3.4) and the taxonomy; see :mod:`openodd.cod`.
"""

from openodd.cod import CodExport, export_cod, export_cods

__all__ = ["CodExport", "export_cod", "export_cods"]
