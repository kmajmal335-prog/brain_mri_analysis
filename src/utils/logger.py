import logging
import sys
from pathlib import Path
from datetime import datetime


class Logger:
    def __init__(self, name="BrainTumorAI", log_dir="experiments"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Create log directory
        log_path = Path(log_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_path.mkdir(parents=True, exist_ok=True)

        # File handler
        fh = logging.FileHandler(log_path / "training.log")
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)