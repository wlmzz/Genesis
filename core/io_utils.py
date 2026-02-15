from __future__ import annotations
import os
import csv
import json
from typing import Dict, Any

def ensure_dir(path: str):
    """Crea directory se non esiste"""
    os.makedirs(path, exist_ok=True)
    return path

def append_csv(path: str, row: Dict[str, Any], header_order: list[str]):
    """Appende riga a CSV, crea header se nuovo file"""
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header_order)
        if new_file:
            w.writeheader()
        w.writerow(row)

def append_jsonl(path: str, obj: Dict[str, Any]):
    """Appende oggetto JSON a file JSONL"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
