class PredictFilter:
    """Genera la predicción y la confianza con el modelo entrenado."""

    def __init__(self, model):
        self.model = model

    def run(self, sample):
        prediction = self.model.predict(sample)[0]

        probability = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(sample)[0]
            probability = round(float(max(probabilities)) * 100, 2)

        return {
            "label": str(prediction),
            "probability": probability
        }
