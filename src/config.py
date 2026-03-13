import json
import logging


def configure_logging(
    logfile_path: str = "app.log", log_to_console: bool = True
) -> None:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=(
            [logging.FileHandler(logfile_path), logging.StreamHandler()]
            if log_to_console
            else [logging.FileHandler(logfile_path)]
        ),
    )


class Config:
    """Configuration class to hold application settings."""

    def __init__(self, config_file_path: str = "config.json"):
        self._configuration = json.load(open(config_file_path, "r"))

    @property
    def collection_name(self) -> str:
        try:
            return self._configuration["collection_name"]
        except KeyError:
            raise KeyError("Missing 'collection_name' in configuration.")

    @property
    def document_path(self) -> str:
        try:
            return self._configuration["document_path"]
        except KeyError:
            raise KeyError("Missing 'document_path' in configuration.")

    @property
    def log_folder(self) -> str:
        try:
            return self._configuration["log_folder"]
        except KeyError:
            raise KeyError("Missing 'log_folder' in configuration.")

    @property
    def raw_data_path(self) -> str:
        try:
            return self._configuration["raw_data_path"]
        except KeyError:
            raise KeyError("Missing 'raw_data_path' in configuration.")

    @property
    def model(self) -> str:
        try:
            return self._configuration["model"]
        except KeyError:
            raise KeyError("Missing 'model' in configuration.")

    @property
    def qdrant_host(self) -> str:
        try:
            return self._configuration["qdrant_host"]
        except KeyError:
            raise KeyError("Missing 'qdrant_host' in configuration.")

    @property
    def qdrant_port(self) -> int:
        try:
            return self._configuration["qdrant_port"]
        except KeyError:
            raise KeyError("Missing 'qdrant_port' in configuration.")
