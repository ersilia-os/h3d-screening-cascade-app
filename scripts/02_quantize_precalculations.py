from sklearn.preprocessing import QuantileTransformer
import pandas as pd
import joblib

df = pd.read_csv("../data/precalculations.csv")

trf = QuantileTransformer()
columns = list(df.columns)[1:]

trf.fit(df[columns])

joblib.dump((trf, columns), "../data/precalculations_quantizer.joblib")