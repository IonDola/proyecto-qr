"""
red/implementation/routing.py
Implementaciones concretas de IRoutingTable e ICircuitTable.
RoutingTable  — tabla de ruteo distance-vector en memoria.
CircuitTable  — tabla de circuitos virtuales efímeros para anonimato.
Ambas son thread-safe: usan threading.Lock para proteger el estado interno.
"""
from __future__ import annotations

import threading
import time

from ..interfaces import IRoutingTable, ICircuitTable, RouteEntry


class RoutingTable(IRoutingTable):
    """
    Tabla de ruteo distance-vector simple, almacenada en memoria.
    Cada entrada: dst_id → RouteEntry(next_hop_mac, cost, updated_at).
    La política de actualización es "gana el menor costo"; si el costo
    entrante es igual, se actualiza el timestamp (keep-alive de ruta).
    Thread-safety: un Lock protege _table en todas las operaciones.
    """

    def __init__(self) -> None:
        self._table: dict[bytes, RouteEntry] = {}
        self._lock  = threading.Lock()

    # IRoutingTable
    def add_route(self, dst_id: bytes, next_hop_mac: bytes, cost: int) -> None:
        """
        Añade o actualiza la ruta hacia dst_id.
        Solo sobreescribe si el nuevo costo es menor o igual al existente.
        """
        with self._lock:
            existing = self._table.get(dst_id)
            if existing is None or cost <= existing.cost:
                self._table[dst_id] = RouteEntry(
                    dst_id       = dst_id,
                    next_hop_mac = next_hop_mac,
                    cost         = cost,
                    updated_at   = time.monotonic(),
                )

    def lookup(self, dst_id: bytes) -> RouteEntry | None:
        with self._lock:
            return self._table.get(dst_id)

    def remove_route(self, dst_id: bytes) -> None:
        with self._lock:
            self._table.pop(dst_id, None)

    def all_routes(self) -> list[RouteEntry]:
        with self._lock:
            return list(self._table.values())

    def purge_stale(self, max_age_s: float) -> int:
        # Elimina rutas cuyo updated_at sea más antiguo que max_age_s segundos.
        cutoff = time.monotonic() - max_age_s
        with self._lock:
            stale = [dst for dst, r in self._table.items() if r.updated_at < cutoff]
            for dst in stale:
                del self._table[dst]
        return len(stale)

    # helpers

    def __len__(self) -> int:
        with self._lock:
            return len(self._table)

    def __repr__(self) -> str:
        with self._lock:
            return f"RoutingTable({len(self._table)} rutas)"


# CircuitTable
class CircuitTable(ICircuitTable):
    """
    Tabla de circuitos virtuales efímeros para ruteo anónimo.
    Cada entrada: circuit_id → (prev_hop_mac | None, next_hop_mac | None)
    None en prev_hop_mac  → este nodo es el originador del circuito.
    None en next_hop_mac  → este nodo es el destinatario final.
    Un nodo intermediario tiene ambos campos definidos y solo sabe
    "de dónde viene" y "a dónde reenviar". Nunca conoce al originador
    ni al destinatario final.
    Thread-safety: un Lock protege _circuits en todas las operaciones.
    """

    def __init__(self) -> None:
        self._circuits: dict[int, tuple[bytes | None, bytes | None]] = {}
        self._lock = threading.Lock()

    # ICircuitTable
    def add_circuit(
        self,
        circuit_id:   int,
        prev_hop_mac: bytes | None,
        next_hop_mac: bytes | None,
    ) -> None:
        with self._lock:
            self._circuits[circuit_id] = (prev_hop_mac, next_hop_mac)

    def lookup(self, circuit_id: int) -> tuple[bytes | None, bytes | None] | None:
        with self._lock:
            return self._circuits.get(circuit_id)

    def remove_circuit(self, circuit_id: int) -> None:
        with self._lock:
            self._circuits.pop(circuit_id, None)

    def all_circuits(self) -> list[int]:
        with self._lock:
            return list(self._circuits.keys())

    # helpers
    def __len__(self) -> int:
        with self._lock:
            return len(self._circuits)
    def __repr__(self) -> str:
        with self._lock:
            return f"CircuitTable({len(self._circuits)} circuitos activos)"
