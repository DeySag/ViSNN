"""Ablation tooling: tagged single runs, resumable matrices, aggregation.

Layout written per run (relative to the results root, by default ``results/``):

    <tag>/config.json     exact CLI args of the run
    <tag>/metrics.csv     one row per epoch, flushed incrementally
    <tag>/summary.json    final metrics + energy report (completion marker)
    <tag>/log.txt         full training stdout
    progress.log          appended by run_matrix.py

``summary.json`` doubles as the resume marker: a tag whose summary exists is
considered complete and is skipped by the matrix runner.
"""
