#!/bin/bash
jupyter nbconvert --to html 00_schematic_level_sets.ipynb --execute --output-dir notebook_outputs
jupyter nbconvert --to html 01_generate_training_screenshots.ipynb --execute --output-dir notebook_outputs
jupyter nbconvert --to html 02_generate_single_interpolation.ipynb --execute --output-dir notebook_outputs
jupyter nbconvert --to html 03_generate_structure_screenshots.ipynb --execute --output-dir notebook_outputs
jupyter nbconvert --to html 04_plot_derivatives.ipynb --execute --output-dir notebook_outputs
jupyter nbconvert --to html 05_plot_composed_derivatives.ipynb --execute --output-dir notebook_outputs
python generate_main_html.py