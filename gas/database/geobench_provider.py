
class GeobenchProvider:
    def __init__(self, geobench_path: str):
        self.geobench_path = geobench_path

    def get_geobench_path(self) -> str:
        return self.geobench_path

    def get_geobench_version(self) -> str:
        return "1.0.0"