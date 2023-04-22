# Imports and setup

import streamlit as st
from rdkit import Chem

st.set_page_config(
    page_title="H3D Screening Cascade",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("H3D Screening Cascade")

# Side bar

st.sidebar.title("Models")

st.sidebar.header("Plasmodium falciparum")
texts = ["NF54", "K1"]
st.sidebar.text("\n".join(texts))

st.sidebar.header("Mycobacterium tuberculosis")
texts = ["MTb"]
st.sidebar.text("\n".join(texts))

# Input

input_molecules = st.text_area(label="Input molecules")

input_molecules = input_molecules.split("\n")


if input_molecules is None:
    st.header("Input molecules")
    
    st.info("These models are lightweight versions")

else:
    st.title("Work in progress")