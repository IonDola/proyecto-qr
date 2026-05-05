"""
red/interfaces.py
Contratos de todas las abstracciones de Capa 2/3 (remote-QR-net).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .packet import Packet


# Tipos auxiliares
@dataclass(frozen=True)
class NodeInfo:
    """
    Registro de un nodo conocido en la mesh.
    Inmutable para poder usarlo como clave en sets/dicts.
    """
    node_id:     bytes # 16 bytes — identificador efímero del nodo
    adapter_mac: bytes # 6 bytes  — MAC del adaptador para alcanzarlo
    cost:        int # costo de salto (menor = más cercano)
    last_seen:   float # timestamp Unix de la última vez que respondió


@dataclass
class RouteEntry:
    """
    Entrada en la tabla de ruteo.
    next_hop_mac: la MAC del adaptador al que enviar para llegar a dst_id.
    """
    dst_id:       bytes # 16 bytes — destino final
    next_hop_mac: bytes # 6 bytes  — próximo salto
    cost:         int # costo acumulado hasta el destino
    updated_at:   float # timestamp de la última actualización


# IRoutingTable
class IRoutingTable(ABC):
    """
    Tabla de ruteo del nodo mesh.
    Relaciona node_id de destino con el next_hop_mac que debe usarse.
    QRNetNode es el único cliente de esta interfaz.
    """
    @abstractmethod
    def add_route(self, dst_id: bytes, next_hop_mac: bytes, cost: int) -> None:
        """
        Añade o actualiza la ruta hacia dst_id.
        Si ya existe una entrada con menor costo, no la sobreescribe.
        """
    @abstractmethod
    def lookup(self, dst_id: bytes) -> RouteEntry | None:
        """
        Busca la mejor ruta hacia dst_id.
        Retorna None si no hay ruta conocida.
        """
    @abstractmethod
    def remove_route(self, dst_id: bytes) -> None:
        """Elimina la ruta hacia dst_id si existe."""
    @abstractmethod
    def all_routes(self) -> list[RouteEntry]:
        """Retorna copia de todas las rutas actuales (útil para debug y HELLO broadcast)."""
    @abstractmethod
    def purge_stale(self, max_age_s: float) -> int:
        """
        Elimina rutas cuya updated_at sea anterior a (now - max_age_s).
        Retorna cuántas rutas fueron eliminadas.
        """


# INodeDirectory
class INodeDirectory(ABC):
    """
    Directorio de nodos conocidos en la mesh.
    Almacena NodeInfo de todos los pares descubiertos.
    QRNetNode lo consulta para saber a qué MACs puede enviar.
    """
    @abstractmethod
    def register(self, node: NodeInfo) -> None:
        """
        Registra o actualiza la información de un nodo.
        Si ya existe el node_id, actualiza last_seen y cost.
        """
    @abstractmethod
    def lookup(self, node_id: bytes) -> NodeInfo | None:
        """Retorna el NodeInfo del nodo con ese ID, o None si no se conoce."""
    @abstractmethod
    def all_nodes(self) -> list[NodeInfo]:
        """Retorna todos los nodos registrados (sin garantía de orden)."""
    @abstractmethod
    def expire_stale(self, max_age_s: float) -> int:
        """
        Elimina nodos cuyo last_seen sea anterior a (now - max_age_s).
        Retorna cuántos nodos fueron eliminados.
        """

# ICircuitTable

class ICircuitTable(ABC):
    """
    Tabla de circuitos virtuales efímeros para ruteo anónimo.

    Cada entrada relaciona un circuit_id con:
      - el MAC del hop anterior (para reenvíos de vuelta)
      - el MAC del hop siguiente (para reenvíos hacia adelante)

    Un nodo intermediario solo conoce sus vecinos inmediatos,
    nunca al originador ni al destinatario final.
    """
    @abstractmethod
    def add_circuit(
        self,
        circuit_id:   int,
        prev_hop_mac: bytes | None,
        next_hop_mac: bytes | None,
    ) -> None:
        """
        Registra un circuito.
        prev_hop_mac es None si este nodo es el originador.
        next_hop_mac es None si este nodo es el destinatario final.
        """
    @abstractmethod
    def lookup(self, circuit_id: int) -> tuple[bytes | None, bytes | None] | None:
        """
        Retorna (prev_hop_mac, next_hop_mac) para el circuit_id dado.
        Retorna None si el circuito no existe.
        """
    @abstractmethod
    def remove_circuit(self, circuit_id: int) -> None:
        """Elimina un circuito (tras FIN o timeout)."""
    @abstractmethod
    def all_circuits(self) -> list[int]:
        """Retorna todos los circuit_ids activos."""

# IQRNetNode
class IQRNetNode(ABC):
    """
    Nodo de la mesh remote-QR-net.
    Capa 7 (AnonChatApp) solo conoce esta interfaz.
    Responsabilidades:
    1. Unirse a la mesh y anunciar su presencia (join_mesh / discover_peers).
    2. Rutear paquetes entrantes hacia su destino (route_packet).
    3. Enviar mensajes con anonimato mediante circuitos efímeros (send_anonymous).
    4. Negociar circuitos virtuales punto a punto (negotiate_circuit).
    """
    @property
    @abstractmethod
    def node_id(self) -> bytes:
        """Identificador efímero de 16 bytes de este nodo."""
    @abstractmethod
    def join_mesh(self) -> None:
        """
        Anuncia la presencia de este nodo en la mesh enviando un HELLO broadcast.
        Bloquea hasta recibir al menos una respuesta o alcanzar el timeout.
        """
    @abstractmethod
    def leave_mesh(self) -> None:
        """
        Envía un BYE broadcast para que los vecinos actualicen sus tablas.
        Detiene los loops de ruteo y descubrimiento.
        """
    @abstractmethod
    def route_packet(self, pkt: Packet) -> None:
        """
        Procesa un paquete entrante:
          - Si dst_id es este nodo → entrega al buffer local.
          - Si TTL == 0 → descarta silenciosamente.
          - Si hay ruta → decrementa TTL y reenvía por el canal adecuado.
          - Si no hay ruta → descarta (no hay ICMP en QR-NET).
        """
    @abstractmethod
    def discover_peers(self) -> list[NodeInfo]:
        """
        Envía un HELLO y espera respuestas durante un tiempo configurable.
        Actualiza NodeDirectory y RoutingTable con los nodos descubiertos.
        Retorna la lista de NodeInfo nuevos o actualizados.
        """
    @abstractmethod
    def send_anonymous(self, msg: bytes, dst_id: bytes) -> None:
        """
        Envía msg hacia dst_id con anonimato:
          1. Negocia un circuit_id efímero si no existe.
          2. Encapsula msg en un Packet con src_id = circuit_id (no el node_id real).
          3. Envía por el canal seleccionado por AdapterSelector.
        """
    @abstractmethod
    def negotiate_circuit(self, dst_id: bytes) -> int:
        """
        Establece un circuito virtual hacia dst_id.
        Retorna el circuit_id asignado (uint32 aleatorio).
        Lanza RoutingError si no se puede alcanzar dst_id.
        """
    @abstractmethod
    def receive(self) -> bytes | None:
        """
        Retorna el próximo mensaje entregado a este nodo, o None si el buffer está vacío.
        La Capa 7 llama a esto en loop para leer mensajes entrantes.
        """
