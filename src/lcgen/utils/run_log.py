"""Utility for logging run arguments to JSON files."""
import json
import datetime
from pathlib import Path


def log_run_args(args, log_file: str = 'logs/run_log.json'):
    """
    Append the current run's arguments to a JSON log file.
    
    Args:
        args: argparse.Namespace with run arguments
        log_file: Path to the JSON log file (relative to project root or absolute)
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing log entries or start fresh
    if log_path.exists():
        try:
            with open(log_path, 'r') as f:
                log_entries = json.load(f)
        except (json.JSONDecodeError, IOError):
            log_entries = []
    else:
        log_entries = []
    
    # Build the new entry
    entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'args': vars(args)  # Convert Namespace to dict
    }
    
    log_entries.append(entry)
    
    # Write back
    with open(log_path, 'w') as f:
        json.dump(log_entries, f, indent=2)
    
    print(f'Logged run arguments to {log_path}')
