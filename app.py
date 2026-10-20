from pathlib import Path

import joblib
import pandas as pd

from flask import Flask, render_template, request, flash
from sklearn.model_selection import train_test_split

from filtros import (
    LoadDatasetFilter,
    ValidateDatasetFilter,
    PrepareDataFilter,
    TrainModelFilter,
    EvaluateModelFilter,
    ValidateInputFilter,
    InputValidationError,
    PrepareSampleFilter,
    PredictFilter,
    Pipeline
)

# Configuración de Flask
app = Flask(
    __name__,
    static_folder="public",
    static_url_path="/public"
)

app.secret_key = "cambia-esta-clave-en-produccion"

# Configuración del modelo
MODEL_PATH = Path("models/weather_model.joblib")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

TARGET = "rain"

# Máximo de opciones para mostrar una columna categórica como lista desplegable
MAX_OPTIONS = 50


def build_schema(X, info):
    """Describe cada característica para generar el formulario de predicción."""
    schema = []

    for column in X.columns:
        if column in info["numeric_columns"]:
            serie = pd.to_numeric(X[column], errors="coerce").dropna()
            schema.append({
                "name": column,
                "kind": "number",
                "min": round(float(serie.min()), 2) if len(serie) else None,
                "max": round(float(serie.max()), 2) if len(serie) else None,
                "example": round(float(serie.median()), 2) if len(serie) else None
            })
        else:
            options = sorted(X[column].dropna().astype(str).unique().tolist())
            schema.append({
                "name": column,
                "kind": "category",
                "options": options if len(options) <= MAX_OPTIONS else []
            })

    return schema


def render_page(results=None, schema=None, prediction_result=None,
                values=None, errors=None):
    """Renderiza la interfaz con todos los datos que necesita la plantilla."""
    return render_template(
        "index.html",
        results=results,
        prediction_features=[f["name"] for f in schema] if schema else None,
        prediction_schema=schema,
        prediction_result=prediction_result,
        prediction_values=values or {},
        field_errors=errors or {}
    )


@app.route("/", methods=["GET"])
def index():
    return render_page()

@app.route("/train", methods=["POST"])
def train():
    uploaded = request.files.get("dataset")

    if uploaded is None or not uploaded.filename:
        flash("Selecciona un archivo CSV.")
        return index()

    if not uploaded.filename.lower().endswith(".csv"):
        flash("El archivo debe tener extensión .csv.")
        return index()

    try:
        # Filtro 1: carga del dataset
        raw = LoadDatasetFilter().run(uploaded.stream)

        # Filtro 2: validación
        validated = ValidateDatasetFilter(TARGET).run(raw)

        # Filtro 3: preparación de los datos
        X, y, info = PrepareDataFilter(TARGET).run(validated)

        # Separación de datos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # Filtro 4: entrenamiento
        model = TrainModelFilter().run(
            X_train,
            y_train,
            info
        )

        # Filtro 5: evaluación
        metrics = EvaluateModelFilter().run(
            model,
            X_test,
            y_test
        )

        # Esquema de las características para el formulario de predicción
        schema = build_schema(X, info)

        results = {
            "rows": len(validated),
            "features": len(X.columns),
            "target": info["target"],
            "class_distribution": {
                str(key): int(value)
                for key, value in y.value_counts().items()
            },
            "metrics": metrics,
            "model_path": str(MODEL_PATH)
        }

        # Guardar modelo, configuración y resultados
        joblib.dump(
            {
                "model": model,
                "features": list(X.columns),
                "target": info["target"],
                "schema": schema,
                "results": results
            },
            MODEL_PATH
        )

        return render_page(results=results, schema=schema)

    except Exception as exc:
        app.logger.exception("Error durante el entrenamiento")
        flash(f"No se pudo procesar el dataset: {exc}")
        return index()

@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_PATH.exists():
        flash("Primero entrena un modelo subiendo un dataset.")
        return index()

    saved = joblib.load(MODEL_PATH)
    features = saved["features"]

    # Modelos guardados antes de esta mejora no tienen esquema
    schema = saved.get("schema") or [
        {"name": feature, "kind": "auto"} for feature in features
    ]
    results = saved.get("results")

    # Tubería de predicción: validar → preparar muestra → predecir
    prediction_pipeline = Pipeline([
        ValidateInputFilter(schema),
        PrepareSampleFilter(features),
        PredictFilter(saved["model"])
    ])

    try:
        prediction_result = prediction_pipeline.run(request.form)

        return render_page(
            results=results,
            schema=schema,
            prediction_result=prediction_result,
            values=request.form
        )

    except InputValidationError as exc:
        flash(str(exc))
        return render_page(
            results=results,
            schema=schema,
            values=request.form,
            errors=exc.errors
        )

    except Exception as exc:
        app.logger.exception("Error durante la predicción")
        flash(f"No se pudo generar la predicción: {exc}")
        return render_page(results=results, schema=schema, values=request.form)

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )
