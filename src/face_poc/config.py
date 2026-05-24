from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class RunConfig:
    input_dir: Path
    output_dir: Path
    device: str
    recursive: bool
    model_name: str
    det_size: int
    detector_threshold: float
    clusterer: str
    dbscan_eps: float
    dbscan_min_samples: int
    agglomerative_distance_threshold: float

    def to_json(self) -> dict[str, object]:
        data = asdict(self)
        data["input_dir"] = str(self.input_dir)
        data["output_dir"] = str(self.output_dir)
        return data
