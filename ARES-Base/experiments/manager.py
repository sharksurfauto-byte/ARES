"""
This module enforces total immutability. Whenever an experiment is initialized, it scans existing directories, increments the run counter
(exp_001, exp_002, etc.), snapshots all configuration files and git commit hashes, and establishes dual-stream logging 
(CSV for plotting/LaTeX tables, JSON for structured web rendering or programmatic parsing).
"""

import csv
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import yaml

class ExperimentManager:
    """
    Research-grade immutable experiment tracker for ARES-Base.
    Prevents overwrites, captures Git commit hashes, snapshots configuration files,
    and streams metrics to both CSV and structured JSON formats.
    """
    def __init__(self, base_dir:Union[str,Path]="experiments/runs", exp_name:str = "baseline"):
        self.base_dir=Path(base_dir)
        self.exp_name=exp_name.lower().replace(" ","_")
        self.exp_dir:Path = self._create_next_exp_dir()

        # File paths within the experiment directory
        self.metrics_csv_path = self.exp_dir / "metrics.csv"
        self.logs_json_path = self.exp_dir / "logs.json"
        self.samples_path = self.exp_dir / "generation_samples.txt"
        self.notes_path = self.exp_dir / "notes.md"
        self.config_dir = self.exp_dir / "configs"

        # Internal state tracking
        self.metrics_history: List[Dict[str, Any]] = []
        self._csv_headers_written = False

    def _create_next_exp_dir(self) -> Path:
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Find existing sequential run numbers
        existing_runs = [d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith("exp_")]
        max_id = 0
        for d in existing_runs:
            try:
                run_id = int(d.name.split("_")[1])
                if run_id > max_id:
                    max_id = run_id
            except (IndexError, ValueError):
                continue
                
        next_id = max_id + 1
        new_exp_dir = self.base_dir / f"exp_{next_id:03d}_{self.exp_name}"
        
        # Guardrail against race conditions or accidental overwrites
        if new_exp_dir.exists():
            raise FileExistsError(f"CRITICAL: Experiment directory already exists at {new_exp_dir}. Aborting to preserve immutability.")
            
        new_exp_dir.mkdir(parents=True, exist_ok=False)
        return new_exp_dir
    
    def _capture_git_hash(self) -> str:
        """Retrieves the current Git commit SHA to ensure cryptographic code traceability."""
        try:
            commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            return commit_hash.decode("utf-8").strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "untracked_or_git_unavailable"
        
    def init_experiment(self, config_paths: List[Union[str, Path]], notes: str = "") -> Path:
        """
        Snapshots configuration files, records metadata, and initializes logging files.
        """
        self.config_dir.mkdir(exist_ok=True)
        
        # 1. Snapshot all config files
        for cfg_path in config_paths:
            path_obj = Path(cfg_path)
            if path_obj.exists():
                shutil.copy2(path_obj, self.config_dir / path_obj.name)
            else:
                print(f"[ExperimentManager] Warning: Config file not found during snapshot: {cfg_path}")

        # 2. Build immutable experiment metadata
        metadata = {
            "experiment_id": self.exp_dir.name,
            "timestamp": datetime.now().isoformat(),
            "git_commit_hash": self._capture_git_hash(),
            "notes": notes
        }
        
        with open(self.exp_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 3. Initialize notes file
        with open(self.notes_path, "w", encoding="utf-8") as f:
            f.write(f"# Experiment Notes: {self.exp_dir.name}\n\n")
            f.write(f"**Created:** {metadata['timestamp']}\n")
            f.write(f"**Git Commit:** `{metadata['git_commit_hash']}`\n\n")
            if notes:
                f.write(f"## Initial Hypothesis\n{notes}\n\n")

        print(f"[ExperimentManager] Initialized immutable run: {self.exp_dir}")
        return self.exp_dir
    
    def log_metrics(self, step: int, epoch: int, metrics: Dict[str, Union[int, float]]) -> None:
        """
        Appends telemetry figures to CSV (for plotting) and JSON (for structured parsing).
        """
        entry = {"step": step, "epoch": epoch, "timestamp": datetime.now().isoformat(), **metrics}
        self.metrics_history.append(entry)

        # 1. Append to CSV
        file_exists = self.metrics_csv_path.exists()
        with open(self.metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            if not file_exists or not self._csv_headers_written:
                writer.writeheader()
                self._csv_headers_written = True
            writer.writerow(entry)

        # 2. Update JSON log
        with open(self.logs_json_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, indent=2)

    def log_generation_sample(self, step: int, prompt: str, generated_text: str) -> None:
        """Logs qualitative text generation samples at specific training milestones."""
        with open(self.samples_path, "a", encoding="utf-8") as f:
            f.write(f"=== Milestone Step: {step} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(f"[PROMPT]: {prompt}\n")
            f.write(f"[OUTPUT]: {generated_text}\n")
            f.write("=" * 70 + "\n\n")

    def append_note(self, heading: str, content: str) -> None:
        """Appends markdown observations to the experiment notes file."""
        with open(self.notes_path, "a", encoding="utf-8") as f:
            f.write(f"### {heading} ({datetime.now().strftime('%H:%M:%S')})\n")
            f.write(f"{content}\n\n")