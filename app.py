# The code for this file was adapted from Code Institute Malaria Detection Project Dashboard
# https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/app.py

import streamlit as st
from app_pages.multipage import MultiPage
from app_pages.project_summary_page import project_summary_page_body
from app_pages.leaves_visualizer_page import leaves_visualizer_body
from app_pages.powdery_mildew_detection_page import powdery_mildew_detection_body
from app_pages.project_hypothesis_page import project_hypothesis_body
from app_pages.ml_performance_metrics_page import ml_perfomance_metrics_body

app = MultiPage(app_name = 'Cherry Leaves Powdery Mildew Detector')

app.add_page(title = 'Project Summary', func = project_summary_page_body)
app.add_page(title = 'Leaves Visualizer', func = leaves_visualizer_body)
app.add_page(title = 'Mildew Detector', func = powdery_mildew_detection_body)
app.add_page(title = 'Project Hypotheses', func = project_hypothesis_body)
app.add_page(title = 'ML Performance Metrics', func = ml_perfomance_metrics_body)

app.run()