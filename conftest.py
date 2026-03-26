import logging
import os
from pathlib import Path

def pytest_configure(config):
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    log_path = output_dir / "pipeline.log"

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)

    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
