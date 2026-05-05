"""
red/implementation/directory.py
NodeDirectory — implementación concreta de INodeDirectory.
Directorio en memoria de todos los nodos conocidos en la mesh.
QRNetNode lo consulta para obtener la lista de pares alcanzables
y lo actualiza cada vez que recibe un paquete HELLO o ROUTE_UPDATE.
Thread-safety: un Lock protege _nodes en todas las operaciones.
"""
from __future__ import annotations

import threading
import time
from ..interfaces import INodeDirectory, NodeInfo

class NodeDirectory(INodeDirectory):
    """
    Registro en memoria de nodos conocidos en la mesh.
    Cada entrada es un NodeInfo identificado por su node_id (16 bytes).
    Si llega un HELLO de un node_id ya conocido, se actualiza su last_seen
    y cost (mantiene el menor costo observado).
    Uso típico:
        directory = NodeDirectory()
        directory.register(NodeInfo(node_id=..., adapter_mac=..., cost=1, last_seen=time.time()))
        info = directory.lookup(some_id)
    """

    def __init__(self) -> None:
        self._nodes: dict[bytes, NodeInfo] = {}
        self._lock  = threading.Lock()

    # INodeDirectory

    def register(self, node: NodeInfo) -> None:
        """
        Registra o actualiza un nodo.
        Actualiza last_seen siempre. Actualiza cost solo si el nuevo es menor.
        """
        with self._lock:
            existing = self._nodes.get(node.node_id)
            if existing is None:
                self._nodes[node.node_id] = node
            else:
                # Mantener el menor costo; actualizar siempre last_seen
                best_cost = min(existing.cost, node.cost)
                self._nodes[node.node_id] = NodeInfo(
                    node_id     = node.node_id,
                    adapter_mac = node.adapter_mac,
                    cost        = best_cost,
                    last_seen   = node.last_seen,
                )

    def lookup(self, node_id: bytes) -> NodeInfo | None:
        with self._lock:
            return self._nodes.get(node_id)

    def all_nodes(self) -> list[NodeInfo]:
        with self._lock:
            return list(self._nodes.values())

    def expire_stale(self, max_age_s: float) -> int:
        """
        Elimina nodos que no han enviado HELLO en los últimos max_age_s segundos.
        Retorna el número de nodos eliminados.
        """
        cutoff = time.time() - max_age_s
        with self._lock:
            stale = [nid for nid, n in self._nodes.items() if n.last_seen < cutoff]
            for nid in stale:
                del self._nodes[nid]
        return len(stale)

    # helpers
    def __len__(self) -> int:
        with self._lock:
            return len(self._nodes)
    def __contains__(self, node_id: bytes) -> bool:
        with self._lock:
            return node_id in self._nodes
    def __repr__(self) -> str:
        with self._lock:
            return f"NodeDirectory({len(self._nodes)} nodos)"
