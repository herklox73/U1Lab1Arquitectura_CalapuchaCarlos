import pandas as pd


class PrepareSampleFilter:
    """Convierte los valores validados en un DataFrame con el orden del modelo."""

    def __init__(self, features):
        self.features = features

    def run(self, values):
        return pd.DataFrame([values], columns=self.features)
