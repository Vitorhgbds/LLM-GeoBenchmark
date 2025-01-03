import logging
from threading import Lock

from rich.logging import RichHandler


class Logger:
    _instance = None
    _lock = Lock()  # Thread-safe initialization

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls, *args, **kwargs)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return  # Prevent re-initialization
        self._initialized = True

        # Configure Rich logging
        logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
        self.logger = logging.getLogger("rich_logger")
        self.logger.setLevel(logging.DEBUG)  # Default log level

    def get_logger(self) -> logging.Logger:
        return self.logger

    def set_level(self, level: str) -> None:
        """Set the logging level dynamically.

        Args:
            level (str): The desired logging level. Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        """
        level = level.upper()
        if level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            self.logger.setLevel(getattr(logging, level))
            self.logger.info(f"Log level set to {level}")
        else:
            self.logger.error(
                f"Invalid log level: {level}. Use one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'."
            )

    def get_level(self) -> None:
        """Get the current logging level."""
        level = logging.getLevelName(self.logger.getEffectiveLevel())
        self.logger.info(f"Current log level: {level}")
        return level
