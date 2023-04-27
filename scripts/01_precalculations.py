from eosce.models import ErsiliaCompoundEmbeddings
import csv
import os
import joblib
import collections
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(root, "..", "data", "reference_library.csv"), "r") as f:
    reader = csv.reader(f)
    smiles_list = []
    for r in reader:
        smiles_list += r

mdl = ErsiliaCompoundEmbeddings()

X = mdl.transform(smiles_list)

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

def load_models():
    models = {}
    models_dir = os.path.join(root, "..", "models")
    for fn in os.listdir(models_dir):
        mn = fn.split(".joblib")[0]
        if mn in model_keys:
            models[mn] = joblib.load(os.path.join(models_dir, fn))
    return models

models = load_models()

probas = collections.OrderedDict()

probas["smiles"] = smiles_list
for k in model_keys:
    mdl = models[k]
    probas[k] = list(mdl.predict_proba(X)[:,1])

df = pd.DataFrame(probas)

df.to_csv(os.path.join(root, "../data/precalculations.csv"), index=False)