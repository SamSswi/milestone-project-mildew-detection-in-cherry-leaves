# The code in this file was taken from Code Institute's Malaria Detection Sample Project
# https://github.com/Code-Institute-Solutions/WalkthroughProject01/blob/main/app_pages/multipage.py
import streamlit as st


class MultiPage:

    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name

    def add_page(self, title, func) -> None:
        self.pages.append({"title": title, "function": func})

    def run(self):
        st.title(self.app_name)
        page = st.sidebar.radio('Menu', self.pages, format_func=lambda page: page['title'])
        page['function']()