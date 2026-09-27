"""Tenant-isolated hot-swappable LoRA adapter registry."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Mapping


@dataclass(frozen=True)
class AdapterRecord:
    adapter_id: str
    tenant_id: str
    version: str
    state: Mapping[str, Any]
    created_at: float


class LoRAAdapterRegistry:
    """Atomic adapter lifecycle with tenant/version isolation.

    Model execution is deliberately delegated to the serving runtime; this
    registry never mutates a live model in-place.
    """
    def __init__(self, *, max_adapters_per_tenant: int = 32) -> None:
        self.max_adapters_per_tenant = max_adapters_per_tenant
        self._records: dict[tuple[str, str], AdapterRecord] = {}
        self._active: dict[str, str] = {}
        self._lock = threading.RLock()

    def publish(self, tenant_id: str, adapter_id: str, version: str, state: Mapping[str, Any]) -> AdapterRecord:
        if not tenant_id or not adapter_id or not version:
            raise ValueError("tenant_id, adapter_id and version are required")
        with self._lock:
            count = sum(1 for (tenant, _), _record in self._records.items() if tenant == tenant_id)
            key = (tenant_id, adapter_id)
            if key not in self._records and count >= self.max_adapters_per_tenant:
                raise RuntimeError("tenant adapter quota exceeded")
            record = AdapterRecord(adapter_id, tenant_id, version, dict(state), time.time())
            self._records[key] = record
            return record

    def activate(self, tenant_id: str, adapter_id: str) -> AdapterRecord:
        with self._lock:
            record = self._records[(tenant_id, adapter_id)]
            self._active[tenant_id] = adapter_id
            return record

    def deactivate(self, tenant_id: str) -> None:
        with self._lock:
            self._active.pop(tenant_id, None)

    def active(self, tenant_id: str) -> AdapterRecord | None:
        with self._lock:
            adapter_id = self._active.get(tenant_id)
            return self._records.get((tenant_id, adapter_id)) if adapter_id else None


    def list(self, tenant_id: str) -> list[AdapterRecord]:
        with self._lock:
            return [record for (tenant, _adapter), record in self._records.items() if tenant == tenant_id]

    def remove(self, tenant_id: str, adapter_id: str) -> None:
        with self._lock:
            if self._active.get(tenant_id) == adapter_id:
                raise RuntimeError("cannot remove the active adapter; deactivate first")
            self._records.pop((tenant_id, adapter_id), None)


__all__ = ["AdapterRecord", "LoRAAdapterRegistry"]
