"""Lab trajectories.

Synthetic and deterministic on purpose. The cohort is served under a DUA that permits no
redistribution of raw laboratory events, so no laboratory table is read and none could be:
the series are generated from the admission's own feature row, through the same seam the
predictor uses. `synthetic.py` owns the generator and the anchoring guarantee that keeps it
honest — every value the model scored for an anchored analyte is reproduced exactly, at that
analyte's reporting precision.
"""
