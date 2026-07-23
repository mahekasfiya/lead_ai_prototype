import json
from pathlib import Path
from typing import Dict, Any, List

class ServiceKnowledge:
    def __init__(self, data_path: Path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.raw = json.load(f)
        self.services = self.raw.get("services", [])
        self._index = {s["service_id"]: s for s in self.services}

    def get_service(self, service_id: str) -> Dict[str, Any]:
        return self._index.get(service_id)