"""
red/implementation/node.py
QRNetNode — implementación concreta de IQRNetNode.

Nodo de la mesh remote-QR-net. Coordina:
  - Descubrimiento de pares (HELLO broadcast)
  - Ruteo de paquetes (distance vector, TTL decrement)
  - Anonimato mediante circuitos virtuales efímeros
  - Entrega de mensajes a la Capa 7 mediante un buffer local

Diseño de hilos:
  - Hilo RX:     lee del INetworkAdapter y llama a _handle_incoming()
  - Hilo HELLO:  envía HELLO periódicamente y purga entradas obsoletas
  - Hilo principal (caller): envía paquetes via send_anonymous / route_packet
"""
from __future__ import annotations

import os
import queue
import struct
import threading
import time
import json

from common.network_policies import NodeID, make_node_id
from common.exceptions import RoutingError, AdapterError

from ..interfaces import (
    IQRNetNode,
    IRoutingTable,
    INodeDirectory,
    ICircuitTable,
    NodeInfo,
)
from ..packet import Packet, PacketType, BROADCAST_ID, DEFAULT_TTL
from .routing import RoutingTable, CircuitTable
from .directory import NodeDirectory

from transmision.interfaces import IAdapterSelector

# Constantes de comportamiento

HELLO_INTERVAL_S  = 30.0 # segundos entre broadcasts HELLO
STALE_ROUTE_S     = 90.0 # segundos sin actualización → ruta obsoleta
STALE_NODE_S      = 90.0 # segundos sin HELLO → nodo obsoleto
DISCOVER_TIMEOUT  = 5.0 # segundos de espera al descubrir pares
RX_TIMEOUT        = 2.0 # segundos de timeout en receive() del adaptador


class QRNetNode(IQRNetNode):
    """
    Nodo de la mesh remote-QR-net.

    Parámetros:
        node_id       : identificador de 16 bytes (se genera aleatoriamente si None)
        selector      : IAdapterSelector ya configurado con los adaptadores disponibles
        routing_table : IRoutingTable (default: RoutingTable())
        directory     : INodeDirectory (default: NodeDirectory())
        circuit_table : ICircuitTable (default: CircuitTable())

    Uso mínimo:
        node = QRNetNode(selector=my_selector)
        node.join_mesh()
        node.send_anonymous(b"hola", dst_id)
        msg = node.receive()
    """

    def __init__(
        self,
        selector:      IAdapterSelector,
        node_id:       bytes | None   = None,
        routing_table: IRoutingTable  | None = None,
        directory:     INodeDirectory | None = None,
        circuit_table: ICircuitTable  | None = None,
    ) -> None:
        self._node_id      = make_node_id(node_id if node_id else os.urandom(16))
        self._selector     = selector
        self._routing      = routing_table or RoutingTable()
        self._directory    = directory     or NodeDirectory()
        self._circuits     = circuit_table or CircuitTable()

        # Buffer de mensajes entregados a este nodo (para Capa 7)
        self._rx_queue: queue.Queue[bytes] = queue.Queue()

        # Control de hilos
        self._running        = False
        self._rx_thread:     threading.Thread | None = None
        self._hello_thread:  threading.Thread | None = None

    # IQRNetNode
    @property
    def node_id(self) -> bytes:
        return bytes(self._node_id)

    def join_mesh(self) -> None:
        """
        Inicia los hilos internos y envía un HELLO broadcast inicial.
        Debe llamarse una vez antes de send_anonymous / receive.
        """
        self._running = True
        self._rx_thread = threading.Thread(
            target=self._rx_loop, daemon=True, name="qrnet-net-rx"
        )
        self._hello_thread = threading.Thread(
            target=self._hello_loop, daemon=True, name="qrnet-net-hello"
        )
        self._rx_thread.start()
        self._hello_thread.start()
        self._send_hello()

    def leave_mesh(self) -> None:
        """Envía BYE broadcast y detiene los hilos internos."""
        self._send_packet(Packet.bye(self._node_id))
        self._running = False

    def route_packet(self, pkt: Packet) -> None:
        """
        Decide si entregar, reenviar o descartar un paquete entrante.

        Casos:
          1. dst_id == este nodo  → entregar al buffer local (_rx_queue).
          2. TTL == 0             → descartar silenciosamente.
          3. Paquete anónimo      → reenviar por circuito si existe.
          4. Ruta conocida        → decrementar TTL y reenviar.
          5. Sin ruta             → descartar (no hay mecanismo de error en QR-NET).
        """
        # Caso 1: entregar localmente
        if pkt.dst_id == bytes(self._node_id):
            self._rx_queue.put(pkt.payload)
            return

        # Caso 2: TTL agotado
        if pkt.ttl == 0:
            return

        forwarded = pkt.with_decremented_ttl()

        # Caso 3: paquete anónimo con circuit_id
        if pkt.is_anonymous:
            self._forward_circuit(forwarded)
            return

        # Caso 4: broadcast → reenviar a todos los adaptadores disponibles
        if pkt.is_broadcast:
            self._flood(forwarded)
            return

        # Caso 5: unicast → consultar tabla de ruteo
        route = self._routing.lookup(pkt.dst_id)
        if route is None:
            return   # sin ruta → descarte silencioso
        self._send_via_mac(forwarded, route.next_hop_mac)

    def discover_peers(self) -> list[NodeInfo]:
        """
        Envía HELLO y espera DISCOVER_TIMEOUT segundos para recibir respuestas.
        Retorna los nodos nuevos o actualizados durante la ventana.
        """
        before = {n.node_id for n in self._directory.all_nodes()}
        self._send_hello()
        time.sleep(DISCOVER_TIMEOUT)
        after  = self._directory.all_nodes()
        return [n for n in after if n.node_id not in before]

    def send_anonymous(self, msg: bytes, dst_id: bytes) -> None:
        """
        Encapsula msg en un Packet con anonimato y lo envía hacia dst_id.
        Si no existe un circuito previo, llama a negotiate_circuit primero.
        """
        circuit_id = self._find_circuit_for(dst_id)
        if circuit_id is None:
            circuit_id = self.negotiate_circuit(dst_id)

        # Usar el circuit_id como src_id (opacidad del originador)
        src_id_anon = struct.pack(">I", circuit_id) + b'\x00' * 12
        pkt = Packet(
            packet_type = PacketType.DATA,
            src_id      = src_id_anon,
            dst_id      = dst_id,
            circuit_id  = circuit_id,
            ttl         = DEFAULT_TTL,
            payload     = msg,
        )
        self._send_packet(pkt)

    def negotiate_circuit(self, dst_id: bytes) -> int:
        """
        Negocia un circuito virtual hacia dst_id.
        - Genera un circuit_id aleatorio (uint32).
        - Lo registra en CircuitTable como originador (prev_hop=None).
        - Envía CIRCUIT_SETUP hacia dst_id para que cada nodo intermedio
          registre su propio entry en su CircuitTable.
        Lanza RoutingError si no hay ruta hacia dst_id.
        """
        route = self._routing.lookup(dst_id)
        if route is None and dst_id != bytes(self._node_id):
            raise RoutingError(
                f"No hay ruta hacia {dst_id.hex()} — ejecuta discover_peers() primero"
            )

        circuit_id = struct.unpack(">I", os.urandom(4))[0]
        next_hop_mac = route.next_hop_mac if route else None

        self._circuits.add_circuit(
            circuit_id   = circuit_id,
            prev_hop_mac = None,           # este nodo es el originador
            next_hop_mac = next_hop_mac,
        )

        setup_pkt = Packet.circuit_setup(
            src_id     = bytes(self._node_id),
            dst_id     = dst_id,
            circuit_id = circuit_id,
        )
        self._send_packet(setup_pkt)
        return circuit_id

    def receive(self) -> bytes | None:
        """
        Retorna el próximo mensaje en el buffer local o None si no hay.
        No bloquea.
        """
        try:
            return self._rx_queue.get_nowait()
        except queue.Empty:
            return None

    # Hilos internos
    def _rx_loop(self) -> None:
        """Hilo RX: lee del adaptador activo y procesa paquetes entrantes."""
        while self._running:
            try:
                adapter = self._selector.select(BROADCAST_ID)
                raw = adapter.receive()
                if raw:
                    pkt = Packet.from_bytes(raw)
                    self._handle_incoming(pkt, adapter.get_mac())
            except AdapterError:
                time.sleep(0.5)
            except Exception:
                pass   # frame malformado o checksum inválido → ignorar
            time.sleep(0.01)

    def _hello_loop(self) -> None:
        """Hilo HELLO: envía HELLOs periódicos y purga entradas obsoletas."""
        while self._running:
            time.sleep(HELLO_INTERVAL_S)
            if self._running:
                self._send_hello()
                self._routing.purge_stale(STALE_ROUTE_S)
                self._directory.expire_stale(STALE_NODE_S)

    # Handlers de paquetes entrantes
    def _handle_incoming(self, pkt: Packet, via_mac: bytes) -> None:
        """Despacha un paquete entrante según su tipo."""
        handlers = {
            PacketType.HELLO:            self._on_hello,
            PacketType.BYE:              self._on_bye,
            PacketType.CIRCUIT_SETUP:    self._on_circuit_setup,
            PacketType.CIRCUIT_TEARDOWN: self._on_circuit_teardown,
            PacketType.ROUTE_UPDATE:     self._on_route_update,
            PacketType.DATA:             self._on_data,
        }
        handler = handlers.get(pkt.packet_type)
        if handler:
            handler(pkt, via_mac)

    def _on_hello(self, pkt: Packet, via_mac: bytes) -> None:
        """
        Registra al nodo que envió el HELLO en el directorio y en la tabla de ruteo.
        Cost = 1 (salto directo). Responde con un HELLO unicast de vuelta.
        """
        info = NodeInfo(
            node_id     = pkt.src_id,
            adapter_mac = via_mac,
            cost        = 1,
            last_seen   = time.time(),
        )
        self._directory.register(info)
        self._routing.add_route(
            dst_id       = pkt.src_id,
            next_hop_mac = via_mac,
            cost         = 1,
        )
        # Responder para que el remitente también nos registre
        if pkt.src_id != bytes(self._node_id):
            reply = Packet.hello(
                src_id  = bytes(self._node_id),
                payload = bytes(self._node_id), # nuestro ID en el payload
            )
            self._send_via_mac(reply, via_mac)

    def _on_bye(self, pkt: Packet, via_mac: bytes) -> None:
        """Elimina al nodo que se fue del directorio y de la tabla de ruteo."""
        self._directory.expire_stale(0)   # forzar expiración del nodo
        self._routing.remove_route(pkt.src_id)

    def _on_circuit_setup(self, pkt: Packet, via_mac: bytes) -> None:
        """
        Registra el circuito en CircuitTable.
        Si somos el destinatario, confirmamos. Si no, reenvíamos hacia el dst.
        """
        circuit_id = pkt.circuit_id
        if pkt.dst_id == bytes(self._node_id):
            # Somos el destinatario: registrar como extremo final
            self._circuits.add_circuit(
                circuit_id   = circuit_id,
                prev_hop_mac = via_mac,
                next_hop_mac = None, # nosotros somos el destino
            )
        else:
            # Nodo intermediario: registrar ambos hops y reenviar
            route = self._routing.lookup(pkt.dst_id)
            next_mac = route.next_hop_mac if route else None
            self._circuits.add_circuit(
                circuit_id   = circuit_id,
                prev_hop_mac = via_mac,
                next_hop_mac = next_mac,
            )
            if next_mac:
                self._send_via_mac(pkt.with_decremented_ttl(), next_mac)

    def _on_circuit_teardown(self, pkt: Packet, via_mac: bytes) -> None:
        """Elimina el circuito y reenvía el teardown si no somos el destino."""
        self._circuits.remove_circuit(pkt.circuit_id)
        if pkt.dst_id != bytes(self._node_id):
            self.route_packet(pkt)

    def _on_route_update(self, pkt: Packet, via_mac: bytes) -> None:
        """
        Procesa una actualización de tabla de ruteo (distance vector).
        El payload es una lista JSON de {dst_id_hex, cost}.
        """
        try:
            entries = json.loads(pkt.payload.decode())
            for entry in entries:
                dst_id = bytes.fromhex(entry["dst_id"])
                cost   = int(entry["cost"]) + 1   # +1 por el salto adicional
                self._routing.add_route(dst_id, via_mac, cost)
        except Exception:
            pass # payload malformado → ignorar

    def _on_data(self, pkt: Packet, via_mac: bytes) -> None:
        """Entrega datos localmente o reenvía según la tabla de ruteo/circuitos."""
        self.route_packet(pkt)

    # Envío
    def _send_packet(self, pkt: Packet) -> None:
        """Serializa y envía un paquete eligiendo el adaptador según dst_id."""
        dst_mac = self._resolve_mac(pkt.dst_id)
        self._send_via_mac(pkt, dst_mac)

    def _send_via_mac(self, pkt: Packet, dst_mac: bytes) -> None:
        """Envía un paquete a través del adaptador seleccionado para dst_mac."""
        try:
            adapter = self._selector.select(dst_mac)
            adapter.send(pkt.to_bytes())
        except AdapterError:
            pass # sin adaptador disponible → descarte silencioso

    def _flood(self, pkt: Packet) -> None:
        """Reenvía un paquete broadcast por todos los adaptadores disponibles."""
        try:
            adapter = self._selector.select(BROADCAST_ID)
            adapter.send(pkt.to_bytes())
        except AdapterError:
            pass

    def _forward_circuit(self, pkt: Packet) -> None:
        """Reenvía un paquete anónimo usando la CircuitTable."""
        entry = self._circuits.lookup(pkt.circuit_id)
        if entry is None:
            return   # circuito desconocido → descarte silencioso
        _, next_hop_mac = entry
        if next_hop_mac is None:
            # Somos el destinatario final
            self._rx_queue.put(pkt.payload)
        else:
            self._send_via_mac(pkt, next_hop_mac)

    def _send_hello(self) -> None:
        """Envía un HELLO broadcast con nuestro node_id en el payload."""
        try:
            pkt = Packet.hello(
                src_id  = bytes(self._node_id),
                payload = bytes(self._node_id),
            )
            self._flood(pkt)
        except Exception:
            pass

    # helpers
    def _resolve_mac(self, dst_id: bytes) -> bytes:
        """
        Obtiene el MAC de next hop para un dst_id.
        Para broadcast retorna el MAC broadcast.
        """
        if dst_id == BROADCAST_ID:
            return b'\xff' * 6
        route = self._routing.lookup(dst_id)
        if route:
            return route.next_hop_mac
        return b'\xff' * 6   # fallback: broadcast si no hay ruta

    def _find_circuit_for(self, dst_id: bytes) -> int | None:
        """Busca un circuit_id activo hacia dst_id, o None si no existe."""
        for cid in self._circuits.all_circuits():
            entry = self._circuits.lookup(cid)
            if entry is not None:
                _, next_hop_mac = entry
                route = self._routing.lookup(dst_id)
                if route and next_hop_mac == route.next_hop_mac:
                    return cid
        return None

    # contexto
    def __enter__(self) -> QRNetNode:
        self.join_mesh()
        return self
    def __exit__(self, *_) -> None:
        self.leave_mesh()
    def __repr__(self) -> str:
        return f"QRNetNode(id={self._node_id.hex()[:8]}…, rutas={len(self._routing)})"
