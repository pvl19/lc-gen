"""Logging utilities for training"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "lcgen",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        format_string: Custom format string (optional)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Default format
    if format_string is None:
        format_string = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'

    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TrainingLogger:
    """
    Simple training logger for tracking metrics.

    Stores metrics in memory and optionally writes to file.
    """

    def __init__(self, log_dir: Optional[str] = None, experiment_name: str = "experiment"):
        self.metrics = {}
        self.log_dir = Path(log_dir) if log_dir else None
        self.experiment_name = experiment_name

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        else:
            self.log_file = None

    def log_metric(self, name: str, value: float, step: int):
        """Log a metric value at a given step"""
        if name not in self.metrics:
            self.metrics[name] = {'steps': [], 'values': []}

        self.metrics[name]['steps'].append(step)
        self.metrics[name]['values'].append(value)

        # Write to file if enabled
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"step={step}, {name}={value:.6f}\n")

    def log_metrics(self, metrics_dict: dict, step: int):
        """Log multiple metrics at once"""
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step)

    def get_metric(self, name: str) -> dict:
        """Retrieve all logged values for a metric"""
        return self.metrics.get(name, {'steps': [], 'values': []})

    def get_latest(self, name: str) -> Optional[float]:
        """Get the most recent value for a metric"""
        if name in self.metrics and len(self.metrics[name]['values']) > 0:
            return self.metrics[name]['values'][-1]
        return None

    def get_best(self, name: str, mode: str = 'min') -> Optional[float]:
        """
        Get the best value for a metric.

        Args:
            name: Metric name
            mode: 'min' or 'max'

        Returns:
            Best value or None if metric doesn't exist
        """
        if name in self.metrics and len(self.metrics[name]['values']) > 0:
            values = self.metrics[name]['values']
            return min(values) if mode == 'min' else max(values)
        return None

    def summary(self) -> str:
        """Generate a summary of all logged metrics"""
        lines = [f"Training Summary for {self.experiment_name}"]
        lines.append("=" * 50)

        for name, data in self.metrics.items():
            if len(data['values']) > 0:
                latest = data['values'][-1]
                best = min(data['values'])
                lines.append(f"{name}:")
                lines.append(f"  Latest: {latest:.6f}")
                lines.append(f"  Best:   {best:.6f}")
                lines.append(f"  Steps:  {len(data['values'])}")

        return "\n".join(lines)
