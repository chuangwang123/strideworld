"""Shared StrideWorld measurement utilities.

Import the concrete ``constants``, ``gt``, ``pose`` or ``se2`` module directly.
Keeping package initialisation empty avoids loading OpenCV when a caller only
needs the dependency-light constants or SE(2) helpers.
"""
