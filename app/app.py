# Imports and setup

import streamlit as st
from rdkit import Chem
from eosce.models import ErsiliaCompoundEmbeddings
import os
import joblib
import pandas as pd
import collections
import csv
from scipy.stats import percentileofscore
from rdkit.Chem import AllChem, Draw

# Theming

st.set_page_config(
    page_title="H3D Screening Cascade",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("H3D Screening Cascade")

# Variables

ROOT = os.path.dirname(os.path.abspath(__file__))

model_names = {
    "caco": "Caco-2",
    "clintH": "CLintH",
    "clintM": "CLintM",
    "clintR": "CLintR",
    "cyp_all_cyp2c9": "CYP2C9",
    "cyp_all_cyp2c19": "CYP2C19",
    "cyp_all_cyp2d6": "CYP2D6",
    "cyp_all_cyp3a4": "CYP3A4",
    "k1": "K1",
    "mtb": "MTb",
    "nf54": "NF54",
    "sol_65": "Sol65",
    "sol": "Aq. Sol. (pH=7.4)",
    "cho": "CHO",
    "hepg2": "HEPG2",
}

model_keys = [
    "nf54", "k1",
    "mtb",
    "cho",
    "hepg2",
    "clintH",
    "clintM",
    "clintR",
    "caco",
    "sol",
    "cyp_all_cyp2c9",
    "cyp_all_cyp2c19",
    "cyp_all_cyp3a4",
    "cyp_all_cyp2d6",
]

# Load models

@st.cache_resource
def load_models():
    models = {}
    models_dir = os.path.join(ROOT, "..", "models")
    for fn in os.listdir(models_dir):
        mn = fn.split(".joblib")[0]
        if mn in model_names:
            models[mn] = joblib.load(os.path.join(models_dir, fn))
    return models

@st.cache_data
def load_precalculations():
    df = pd.read_csv(os.path.join(ROOT, "..", "data", "precalculations.csv"))
    data = {}
    columns = list(df.columns)[1:]
    for c in columns:
        data[c] = df[c].tolist()
    return data

models = load_models()
precalcs = load_precalculations()
embedder = ErsiliaCompoundEmbeddings()

# Side bar

st.sidebar.title("Lightweight Models")

st.sidebar.markdown("Lightweight models have been trained using the [Ersilia Compound Embedding](https://github.com/ersilia-os/compound-embedding-lite). Values in parenthesis show the **percentile (%)** of the prediction score with respect to a reference library of 200k compounds.")


st.sidebar.header("Plasmodium falciparum")
texts = ["NF54", "K1"]
st.sidebar.text("\n".join(texts))

st.sidebar.header("Mycobacterium tuberculosis")
texts = ["MTb"]
st.sidebar.text("\n".join(texts))

st.sidebar.header("Cytotoxicity")
texts = ["CHO", "HEPG2"]
st.sidebar.text("\n".join(texts))

st.sidebar.header("Clearance")
texts = ["CLintH", "CLintM", "CLintR"]
st.sidebar.text("\n".join(texts))

st.sidebar.header("Permeability and solubility")
texts = ["Caco-2", "Aq. Sol."]
st.sidebar.text("\n".join(texts))

st.sidebar.header("Cytochromes")
texts = ["CYP2C9", "CYP2C19", "CPY3A4", "CPY2D6"]
st.sidebar.text("\n".join(texts))


# Input

input_molecules = st.text_area(label="Input molecules")

input_molecules = input_molecules.split("\n")
input_molecules = [inp for inp in input_molecules if inp != ""]

def is_valid_input_molecules():
    if len(input_molecules) == 0:
        return False
    for input_molecule in input_molecules:
        mol = Chem.MolFromSmiles(input_molecule)
        if mol is None:
            st.error("Input {0} is not a valid SMILES".format(input_molecule))
            return False
    return True


def get_molecule_image(smiles):
    m = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(m)
    opts = Draw.DrawingOptions()
    opts.bgColor = None
    im = Draw.MolToImage(m, size=(200, 200), options=opts)
    return im


if is_valid_input_molecules():
    st.header("Input molecules")
    X = embedder.transform(input_molecules)
    results = collections.OrderedDict()
    results["smiles"] = input_molecules
    for k in model_keys:
        if k in models:
            v = models[k]
            results[k] = list(v.predict_proba(X)[:,1])
            results[k+"_norm"] = [percentileofscore(precalcs[k], x) for x in results[k]]
        else:
            results[k] = [None]*X.shape[0]
            results[k+"_norm"] = [None]*X.shape[0]
    data = pd.DataFrame(results)
    
    for v in data.iterrows():
        idx = v[0]
        r = v[1]
        st.markdown("### {0}: `{1}`".format(idx, r["smiles"]))
        cols = st.columns(5)
        image = get_molecule_image(r["smiles"])
        cols[0].image(image)
        texts = [
            "NF54    : {0:.3f} ({1:.1f}%)".format(r["nf54"], r["nf54_norm"]),
            "K1      : {0:.3f} ({1:.1f}%)".format(r["k1"], r["k1_norm"]),
            "",
            "MTb     : {0:.3f} ({1:.1f}%)".format(r["mtb"], r["mtb_norm"])
        ]
        cols[1].text("\n".join(texts))
        texts = [
            "CHO     : {0:.3f} ({1:.1f}%)".format(r["cho"], r["cho_norm"]),
            "HEPG2   : {0:.3f} ({1:.1f}%)".format(r["hepg2"], r["hepg2_norm"]),
            "",
            "CLintH  : {0:.3f} ({1:.1f}%)".format(r["clintH"], r["clintH_norm"]),
            "CLintM  : {0:.3f} ({1:.1f}%)".format(r["clintM"], r["clintM_norm"]),
            "CLintR  : {0:.3f} ({1:.1f}%)".format(r["clintR"], r["clintR_norm"]),
        ]
        cols[2].text("\n".join(texts))
        texts = [
            "Caco-2  : {0:.3f} ({1:.1f}%)".format(r["caco"], r["caco_norm"]),
            "Aq. Sol.: {0:.3f} ({1:.1f}%)".format(r["sol"], r["sol_norm"])
        ]
        cols[3].text("\n".join(texts))
        texts = [
            "CYP2C9  : {0:.3f} ({1:.1f}%)".format(r["cyp_all_cyp2c9"], r["cyp_all_cyp2c9_norm"]),
            "CYP2C19 : {0:.3f} ({1:.1f}%)".format(r["cyp_all_cyp2c19"], r["cyp_all_cyp2c19_norm"]),
            "CYP3A4  : {0:.3f} ({1:.1f}%)".format(r["cyp_all_cyp3a4"], r["cyp_all_cyp3a4_norm"]),
            "CYP2D6  : {0:.3f} ({1:.1f}%)".format(r["cyp_all_cyp2d6"], r["cyp_all_cyp2d6_norm"])
        ]
        cols[4].text("\n".join(texts))

    columns = [x for x in list(data.columns) if not x.endswith("_norm")]
    data = data[columns].rename(columns=model_names)

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode("utf-8")

    csv = convert_df(data)

    st.download_button(
        "Download as CSV", csv, "predictions.csv", "text/csv", key="download-csv"
    )

else:
    st.markdown("Input molecules as a list of SMILES strings. For example:")
    smiles = []
    with open(os.path.join(ROOT, "..", "data", "example.csv"), "r") as f:
        reader = csv.reader(f)
        for r in reader:
            smiles += [r[0]]
    st.text("\n".join(smiles))

    st.info("The H3D screening cascade models are related to [Turon*, Hlozek* et al](https://www.biorxiv.org/content/10.1101/2022.12.13.520154v1). Please note that, in this application, we provide **lightweight** versions of the models. For full models, please see the [ZairaChem](https://github.com/ersilia-os/zaira-chem) tool.")