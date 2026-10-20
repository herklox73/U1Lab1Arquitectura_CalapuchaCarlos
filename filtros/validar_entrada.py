class InputValidationError(ValueError):
    """Error con los mensajes de validación de cada campo del formulario."""

    def __init__(self, errors):
        self.errors = errors
        super().__init__("Revisa los campos marcados en el formulario.")


class ValidateInputFilter:
    """Valida los datos ingresados en el formulario de predicción."""

    def __init__(self, schema):
        # schema: lista de campos con "name", "kind" ("number", "category" o "auto")
        # y, para las categorías, la lista de "options" válidas.
        self.schema = schema

    def run(self, form_data):
        values = {}
        errors = {}

        for field in self.schema:
            name = field["name"]
            raw_value = str(form_data.get(name, "")).strip()

            if raw_value == "":
                errors[name] = "Este campo es obligatorio."
                continue

            if field["kind"] == "number":
                try:
                    values[name] = float(raw_value.replace(",", "."))
                except ValueError:
                    errors[name] = "Ingresa un valor numérico."
            elif field["kind"] == "auto":
                # Sin esquema: se usa número si se puede, si no, texto
                try:
                    values[name] = float(raw_value)
                except ValueError:
                    values[name] = raw_value
            else:
                options = field.get("options") or []
                if options and raw_value not in options:
                    errors[name] = "Selecciona una de las opciones de la lista."
                else:
                    values[name] = raw_value

        if errors:
            raise InputValidationError(errors)

        return values
