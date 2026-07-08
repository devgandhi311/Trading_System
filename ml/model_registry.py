"""Simple filesystem model registry for trained ML artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import pickle
from pathlib import Path
from typing import Any

from utils.config_loader import ConfigLoader
from utils.file_utils import ensure_parent_dir
from utils.time_utils import now_ist


@dataclass(frozen=True)
class ModelRegistryEntry:
    """Metadata for a registered model artifact."""

    model_name: str
    model_type: str
    target: str
    artifact_path: str
    metrics_path: str | None
    feature_importance_path: str | None
    created_at: str
    metadata: dict[str, Any]


class ModelRegistry:
    """Persist and load model artifacts with JSON metadata."""

    def __init__(
        self,
        *,
        config_loader: ConfigLoader | None = None,
        model_dir: str | Path | None = None,
        registry_path: str | Path | None = None,
    ) -> None:
        """Create a registry rooted at data/model_data by default."""

        self.config_loader = config_loader or ConfigLoader()
        config = self.config_loader.load()
        self.model_dir = Path(model_dir or config.settings.data_paths["models"])
        self.registry_path = Path(registry_path or self.model_dir / "model_registry.json")

    def save_model(
        self,
        model: Any,
        *,
        model_name: str,
        model_type: str,
        target: str,
        metrics_path: str | Path | None = None,
        feature_importance_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelRegistryEntry:
        """Save a pickled model and registry entry."""

        artifact_path = ensure_parent_dir(self.model_dir / f"{model_name}.pkl")
        with artifact_path.open("wb") as file:
            pickle.dump(model, file)
        entry = ModelRegistryEntry(
            model_name=model_name,
            model_type=model_type,
            target=target,
            artifact_path=str(artifact_path),
            metrics_path=str(metrics_path) if metrics_path else None,
            feature_importance_path=str(feature_importance_path) if feature_importance_path else None,
            created_at=now_ist().isoformat(),
            metadata=metadata or {},
        )
        self._append_entry(entry)
        return entry

    def load_model(self, model_name: str) -> Any:
        """Load a pickled model by name."""

        with (self.model_dir / f"{model_name}.pkl").open("rb") as file:
            return pickle.load(file)

    def list_models(self) -> list[ModelRegistryEntry]:
        """Return all registry entries."""

        if not self.registry_path.exists():
            return []
        data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        return [ModelRegistryEntry(**entry) for entry in data]

    def _append_entry(self, entry: ModelRegistryEntry) -> None:
        entries = [asdict(existing) for existing in self.list_models()]
        entries.append(asdict(entry))
        ensure_parent_dir(self.registry_path)
        self.registry_path.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
