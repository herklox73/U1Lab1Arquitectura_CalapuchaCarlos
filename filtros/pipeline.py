class Pipeline:
    """Orquestador de la tubería: ejecuta los filtros en orden.

    La salida de cada filtro es la entrada del siguiente (Pipes and Filters).
    """

    def __init__(self, filters):
        self.filters = list(filters)

    def run(self, data):
        for current_filter in self.filters:
            data = current_filter.run(data)
        return data
