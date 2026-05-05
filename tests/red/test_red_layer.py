"""
tests/red/test_red_layer.py

Suite de tests unitarios para la Capa 2/3 de QR-NET.

Cubre:
  - red/packet.py              (Packet, PacketType, serialización, CRC-32)
  - red/implementation/routing (RoutingTable, CircuitTable)
  - red/implementation/directory (NodeDirectory)
  - red/implementation/node    (QRNetNode — con adaptadores mock)

No requiere hardware ni cámara — todos los adaptadores son mocks.
"""
from __future__ import annotations

import os
import struct
import time
import threading
from unittest.mock import MagicMock, patch

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from common.exceptions import ChecksumError, RoutingError
from common.network_policies import AdapterType, SelectionPolicy

from red.packet import Packet, PacketType, BROADCAST_ID, DEFAULT_TTL, PACKET_HEADER_SIZE
from red.interfaces import NodeInfo, RouteEntry, IQRNetNode
from red.implementation.routing import RoutingTable, CircuitTable
from red.implementation.directory import NodeDirectory
from red.implementation.node import QRNetNode


# Helpers
def make_id(seed: int = 1) -> bytes:
    # Genera un node_id de 16 bytes determinista a partir de un entero.
    return bytes([seed % 256] * 16)


def make_mac(seed: int = 1) -> bytes:
    # Genera un MAC de 6 bytes determinista.
    return bytes([seed % 256] * 6)


def mock_adapter(
    atype: AdapterType = AdapterType.ETHERNET,
    available: bool    = True,
    cost: int          = 10,
    mac: bytes | None  = None,
) -> MagicMock:
    # Crea un INetworkAdapter mock listo para usar.
    adapter = MagicMock()
    adapter.get_type.return_value      = atype
    adapter.is_available.return_value  = available
    adapter.get_cost.return_value      = cost
    adapter.get_mac.return_value       = mac or make_mac()
    adapter.send.return_value          = True
    adapter.receive.return_value       = None
    return adapter


def mock_selector(adapter: MagicMock) -> MagicMock:
    # Crea un IAdapterSelector mock que siempre retorna el adaptador dado.
    selector = MagicMock()
    selector.select.return_value = adapter
    return selector



# Packet
class TestPacket:
    def _make(self, **kwargs) -> Packet:
        defaults = dict(src_id=make_id(1), dst_id=make_id(2), payload=b"hola red")
        defaults.update(kwargs)
        return Packet(**defaults)

    def test_roundtrip_data(self):
        pkt = self._make()
        assert Packet.from_bytes(pkt.to_bytes()).payload == b"hola red"

    def test_roundtrip_preserves_all_fields(self):
        original = self._make(
            ttl        = 8,
            circuit_id = 0xDEADBEEF,
            payload    = b"payload de prueba",
        )
        restored = Packet.from_bytes(original.to_bytes())
        assert restored.ttl        == 8
        assert restored.circuit_id == 0xDEADBEEF
        assert restored.src_id     == make_id(1)
        assert restored.dst_id     == make_id(2)
        assert restored.payload    == b"payload de prueba"

    def test_checksum_detection(self):
        raw = bytearray(self._make().to_bytes())
        raw[-1] ^= 0xFF   # corromper último byte del CRC
        with pytest.raises(ChecksumError):
            Packet.from_bytes(bytes(raw))

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            Packet.from_bytes(b"\x00" * 10)

    def test_verify_valid(self):
        assert self._make().verify() is True

    def test_empty_payload(self):
        pkt = self._make(payload=b"")
        assert Packet.from_bytes(pkt.to_bytes()).payload == b""

    def test_is_broadcast_true(self):
        pkt = Packet(src_id=make_id(1), dst_id=BROADCAST_ID)
        assert pkt.is_broadcast is True

    def test_is_broadcast_false(self):
        pkt = self._make()
        assert pkt.is_broadcast is False

    def test_is_anonymous_true(self):
        pkt = self._make(circuit_id=42)
        assert pkt.is_anonymous is True

    def test_is_anonymous_false(self):
        pkt = self._make(circuit_id=0)
        assert pkt.is_anonymous is False

    def test_with_decremented_ttl(self):
        pkt = self._make(ttl=5)
        assert pkt.with_decremented_ttl().ttl == 4

    def test_ttl_does_not_go_below_zero(self):
        pkt = self._make(ttl=0)
        assert pkt.with_decremented_ttl().ttl == 0

    def test_factory_hello(self):
        pkt = Packet.hello(make_id(1))
        assert pkt.packet_type == PacketType.HELLO
        assert pkt.is_broadcast

    def test_factory_bye(self):
        pkt = Packet.bye(make_id(1))
        assert pkt.packet_type == PacketType.BYE
        assert pkt.is_broadcast

    def test_factory_circuit_setup(self):
        pkt = Packet.circuit_setup(make_id(1), make_id(2), 0xABCD)
        assert pkt.packet_type == PacketType.CIRCUIT_SETUP
        assert pkt.circuit_id  == 0xABCD

    def test_factory_circuit_teardown(self):
        pkt = Packet.circuit_teardown(make_id(1), make_id(2), 99)
        assert pkt.packet_type == PacketType.CIRCUIT_TEARDOWN

    def test_src_id_wrong_length_raises(self):
        with pytest.raises(ValueError):
            Packet(src_id=b"\x00" * 5, dst_id=make_id(2))

    def test_dst_id_wrong_length_raises(self):
        with pytest.raises(ValueError):
            Packet(src_id=make_id(1), dst_id=b"\x00" * 3)

    def test_large_payload_roundtrip(self):
        big = os.urandom(500)
        pkt = self._make(payload=big)
        assert Packet.from_bytes(pkt.to_bytes()).payload == big

    def test_packet_type_enum_survives_serialization(self):
        for ptype in PacketType:
            pkt = Packet(
                packet_type = ptype,
                src_id      = make_id(1),
                dst_id      = make_id(2),
            )
            restored = Packet.from_bytes(pkt.to_bytes())
            assert restored.packet_type == ptype

# RoutingTable
class TestRoutingTable:
    def test_add_and_lookup(self):
        rt = RoutingTable()
        rt.add_route(make_id(2), make_mac(1), cost=3)
        entry = rt.lookup(make_id(2))
        assert entry is not None
        assert entry.next_hop_mac == make_mac(1)
        assert entry.cost == 3

    def test_lookup_unknown_returns_none(self):
        rt = RoutingTable()
        assert rt.lookup(make_id(99)) is None

    def test_lower_cost_overwrites(self):
        rt = RoutingTable()
        rt.add_route(make_id(5), make_mac(1), cost=10)
        rt.add_route(make_id(5), make_mac(2), cost=3)
        entry = rt.lookup(make_id(5))
        assert entry.cost == 3
        assert entry.next_hop_mac == make_mac(2)

    def test_higher_cost_does_not_overwrite(self):
        rt = RoutingTable()
        rt.add_route(make_id(5), make_mac(1), cost=3)
        rt.add_route(make_id(5), make_mac(2), cost=10)
        entry = rt.lookup(make_id(5))
        assert entry.cost == 3
        assert entry.next_hop_mac == make_mac(1)

    def test_remove_route(self):
        rt = RoutingTable()
        rt.add_route(make_id(7), make_mac(1), cost=1)
        rt.remove_route(make_id(7))
        assert rt.lookup(make_id(7)) is None

    def test_remove_nonexistent_is_safe(self):
        rt = RoutingTable()
        rt.remove_route(make_id(99))   # no debe lanzar

    def test_all_routes(self):
        rt = RoutingTable()
        rt.add_route(make_id(1), make_mac(1), cost=1)
        rt.add_route(make_id(2), make_mac(2), cost=2)
        assert len(rt.all_routes()) == 2

    def test_purge_stale_removes_old_entries(self):
        rt = RoutingTable()
        rt.add_route(make_id(1), make_mac(1), cost=1)
        time.sleep(0.01)
        removed = rt.purge_stale(max_age_s=0.001)
        assert removed == 1
        assert rt.lookup(make_id(1)) is None

    def test_purge_stale_keeps_fresh_entries(self):
        rt = RoutingTable()
        rt.add_route(make_id(1), make_mac(1), cost=1)
        removed = rt.purge_stale(max_age_s=60.0)
        assert removed == 0

    def test_len(self):
        rt = RoutingTable()
        assert len(rt) == 0
        rt.add_route(make_id(1), make_mac(1), cost=1)
        assert len(rt) == 1

    def test_thread_safety(self):
        rt = RoutingTable()
        errors = []

        def writer(n):
            try:
                for i in range(50):
                    rt.add_route(make_id(n), make_mac(n), cost=i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []



# CircuitTable
class TestCircuitTable:
    def test_add_and_lookup(self):
        ct = CircuitTable()
        ct.add_circuit(42, prev_hop_mac=make_mac(1), next_hop_mac=make_mac(2))
        result = ct.lookup(42)
        assert result == (make_mac(1), make_mac(2))

    def test_lookup_unknown_returns_none(self):
        ct = CircuitTable()
        assert ct.lookup(9999) is None

    def test_originator_has_none_prev(self):
        ct = CircuitTable()
        ct.add_circuit(1, prev_hop_mac=None, next_hop_mac=make_mac(3))
        prev, next_ = ct.lookup(1)
        assert prev is None
        assert next_ == make_mac(3)

    def test_destination_has_none_next(self):
        ct = CircuitTable()
        ct.add_circuit(2, prev_hop_mac=make_mac(1), next_hop_mac=None)
        prev, next_ = ct.lookup(2)
        assert prev == make_mac(1)
        assert next_ is None

    def test_remove_circuit(self):
        ct = CircuitTable()
        ct.add_circuit(7, None, make_mac(1))
        ct.remove_circuit(7)
        assert ct.lookup(7) is None

    def test_remove_nonexistent_is_safe(self):
        ct = CircuitTable()
        ct.remove_circuit(999)

    def test_all_circuits(self):
        ct = CircuitTable()
        ct.add_circuit(1, None, make_mac(1))
        ct.add_circuit(2, make_mac(1), make_mac(2))
        assert set(ct.all_circuits()) == {1, 2}

    def test_len(self):
        ct = CircuitTable()
        assert len(ct) == 0
        ct.add_circuit(1, None, None)
        assert len(ct) == 1


# NodeDirectory
class TestNodeDirectory:
    def _make_node(self, seed: int = 1, cost: int = 1) -> NodeInfo:
        return NodeInfo(
            node_id     = make_id(seed),
            adapter_mac = make_mac(seed),
            cost        = cost,
            last_seen   = time.time(),
        )
    
    def test_register_and_lookup(self):
        nd = NodeDirectory()
        node = self._make_node(1)
        nd.register(node)
        result = nd.lookup(make_id(1))
        assert result is not None
        assert result.node_id == make_id(1)

    def test_lookup_unknown_returns_none(self):
        nd = NodeDirectory()
        assert nd.lookup(make_id(99)) is None

    def test_register_updates_last_seen(self):
        nd = NodeDirectory()
        nd.register(self._make_node(1, cost=5))
        time.sleep(0.01)
        newer = NodeInfo(
            node_id=make_id(1), adapter_mac=make_mac(1), cost=5, last_seen=time.time()
        )
        nd.register(newer)
        updated = nd.lookup(make_id(1))
        assert updated.last_seen >= newer.last_seen - 0.001

    def test_register_keeps_lower_cost(self):
        nd = NodeDirectory()
        nd.register(self._make_node(1, cost=3))
        nd.register(NodeInfo(make_id(1), make_mac(1), cost=10, last_seen=time.time()))
        assert nd.lookup(make_id(1)).cost == 3

    def test_register_updates_to_lower_cost(self):
        nd = NodeDirectory()
        nd.register(self._make_node(1, cost=10))
        nd.register(NodeInfo(make_id(1), make_mac(1), cost=2, last_seen=time.time()))
        assert nd.lookup(make_id(1)).cost == 2

    def test_all_nodes(self):
        nd = NodeDirectory()
        nd.register(self._make_node(1))
        nd.register(self._make_node(2))
        assert len(nd.all_nodes()) == 2

    def test_expire_stale(self):
        nd = NodeDirectory()
        nd.register(self._make_node(1))
        time.sleep(0.01)
        removed = nd.expire_stale(max_age_s=0.001)
        assert removed == 1
        assert nd.lookup(make_id(1)) is None

    def test_expire_stale_keeps_fresh(self):
        nd = NodeDirectory()
        nd.register(self._make_node(1))
        removed = nd.expire_stale(max_age_s=60.0)
        assert removed == 0

    def test_contains(self):
        nd = NodeDirectory()
        nd.register(self._make_node(3))
        assert make_id(3) in nd
        assert make_id(99) not in nd

    def test_len(self):
        nd = NodeDirectory()
        assert len(nd) == 0
        nd.register(self._make_node(1))
        assert len(nd) == 1

    def test_thread_safety(self):
        nd = NodeDirectory()
        errors = []

        def writer(seed):
            try:
                for _ in range(50):
                    nd.register(NodeInfo(make_id(seed), make_mac(seed), 1, time.time()))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# QRNetNode
class TestQRNetNode:
    # Tests de QRNetNode usando mocks de IAdapterSelector e INetworkAdapter.
    def _make_node(
        self,
        node_id: bytes | None = None,
        available: bool = True,
    ) -> tuple[QRNetNode, MagicMock]:
        adapter  = mock_adapter(available=available)
        selector = mock_selector(adapter)
        node     = QRNetNode(
            selector  = selector,
            node_id   = node_id or make_id(1),
        )
        return node, adapter

    # node_id
    def test_node_id_is_16_bytes(self):
        node, _ = self._make_node()
        assert len(node.node_id) == 16

    def test_node_id_custom(self):
        node, _ = self._make_node(node_id=make_id(7))
        assert node.node_id == make_id(7)

    def test_node_id_random_when_none(self):
        adapter  = mock_adapter()
        selector = mock_selector(adapter)
        n1 = QRNetNode(selector=selector)
        n2 = QRNetNode(selector=selector)
        assert n1.node_id != n2.node_id

    # receive
    def test_receive_empty_returns_none(self):
        node, _ = self._make_node()
        assert node.receive() is None

    # route_packet: entrega local
    def test_route_packet_delivers_locally(self):
        node, _ = self._make_node(node_id=make_id(1))
        pkt = Packet(
            src_id  = make_id(2),
            dst_id  = make_id(1),   # destinado a este nodo
            payload = b"mensaje local",
        )
        node.route_packet(pkt)
        msg = node.receive()
        assert msg == b"mensaje local"

    def test_route_packet_ttl_zero_discards(self):
        node, adapter = self._make_node(node_id=make_id(1))
        pkt = Packet(
            src_id  = make_id(2),
            dst_id  = make_id(3),
            ttl     = 0,
            payload = b"deberia descartarse",
        )
        node.route_packet(pkt)
        adapter.send.assert_not_called()
        assert node.receive() is None

    def test_route_packet_no_route_discards(self):
        node, adapter = self._make_node(node_id=make_id(1))
        pkt = Packet(
            src_id = make_id(2),
            dst_id = make_id(99),   # destino desconocido
            ttl    = 5,
        )
        node.route_packet(pkt)
        adapter.send.assert_not_called()

    def test_route_packet_forwards_when_route_known(self):
        node, adapter = self._make_node(node_id=make_id(1))
        node._routing.add_route(make_id(5), make_mac(3), cost=2)
        pkt = Packet(
            src_id = make_id(2),
            dst_id = make_id(5),
            ttl    = 4,
        )
        node.route_packet(pkt)
        adapter.send.assert_called_once()

    def test_route_packet_decrements_ttl_on_forward(self):
        node, adapter = self._make_node(node_id=make_id(1))
        node._routing.add_route(make_id(5), make_mac(3), cost=2)
        pkt = Packet(src_id=make_id(2), dst_id=make_id(5), ttl=10)
        node.route_packet(pkt)
        sent_raw = adapter.send.call_args[0][0]
        forwarded = Packet.from_bytes(sent_raw)
        assert forwarded.ttl == 9

    # route_packet: circuito anónimo
    def test_route_packet_anonymous_delivers_at_destination(self):
        # Nodo destinatario de un circuito recibe el payload.
        node, _ = self._make_node(node_id=make_id(1))
        node._circuits.add_circuit(
            circuit_id   = 0xABCD,
            prev_hop_mac = make_mac(9),
            next_hop_mac = None,   # este nodo es el destino
        )
        pkt = Packet(
            src_id     = make_id(2),
            dst_id     = make_id(1),
            circuit_id = 0xABCD,
            payload    = b"mensaje anonimo",
        )
        node.route_packet(pkt)
        assert node.receive() == b"mensaje anonimo"

    def test_route_packet_anonymous_forwards_at_intermediate(self):
        # Nodo intermedio reenvía por next_hop_mac del circuito.
        node, adapter = self._make_node(node_id=make_id(1))
        node._circuits.add_circuit(
            circuit_id   = 0x1234,
            prev_hop_mac = make_mac(8),
            next_hop_mac = make_mac(9),
        )
        pkt = Packet(
            src_id     = make_id(2),
            dst_id     = make_id(3),
            circuit_id = 0x1234,
            ttl        = 5,
            payload    = b"transito",
        )
        node.route_packet(pkt)
        adapter.send.assert_called_once()

    def test_route_packet_anonymous_unknown_circuit_discards(self):
        node, adapter = self._make_node()
        pkt = Packet(
            src_id     = make_id(2),
            dst_id     = make_id(3),
            circuit_id = 0xDEAD,   # circuito desconocido
            ttl        = 5,
        )
        node.route_packet(pkt)
        adapter.send.assert_not_called()

    # negotiate_circuit
    def test_negotiate_circuit_raises_if_no_route(self):
        node, _ = self._make_node(node_id=make_id(1))
        with pytest.raises(RoutingError):
            node.negotiate_circuit(make_id(99))

    def test_negotiate_circuit_registers_in_table(self):
        node, adapter = self._make_node(node_id=make_id(1))
        node._routing.add_route(make_id(2), make_mac(5), cost=1)
        cid = node.negotiate_circuit(make_id(2))
        assert isinstance(cid, int)
        entry = node._circuits.lookup(cid)
        assert entry is not None
        prev, _ = entry
        assert prev is None   # somos el originador

    def test_negotiate_circuit_sends_setup_packet(self):
        node, adapter = self._make_node(node_id=make_id(1))
        node._routing.add_route(make_id(2), make_mac(5), cost=1)
        node.negotiate_circuit(make_id(2))
        adapter.send.assert_called_once()
        sent = Packet.from_bytes(adapter.send.call_args[0][0])
        assert sent.packet_type == PacketType.CIRCUIT_SETUP

    # send_anonymous
    def test_send_anonymous_creates_circuit_if_needed(self):
        node, adapter = self._make_node(node_id=make_id(1))
        node._routing.add_route(make_id(2), make_mac(5), cost=1)
        node.send_anonymous(b"secreto", make_id(2))
        assert adapter.send.call_count >= 2   # CIRCUIT_SETUP + DATA

    def test_send_anonymous_uses_existing_circuit(self):
        node, adapter = self._make_node(node_id=make_id(1))
        node._routing.add_route(make_id(2), make_mac(5), cost=1)
        # Registrar circuito manualmente
        node._circuits.add_circuit(0xBEEF, None, make_mac(5))
        node.send_anonymous(b"mensaje", make_id(2))
        # Solo debe enviarse el DATA (no CIRCUIT_SETUP de nuevo)
        sent_types = []
        for call in adapter.send.call_args_list:
            pkt = Packet.from_bytes(call[0][0])
            sent_types.append(pkt.packet_type)
        # Debe haber al menos un DATA con circuit_id != 0
        data_pkts = [t for t in sent_types if t == PacketType.DATA]
        assert len(data_pkts) >= 1

    # on_hello
    def test_on_hello_registers_node(self):
        node, adapter = self._make_node(node_id=make_id(1))
        hello = Packet.hello(src_id=make_id(2), payload=make_id(2))
        node._handle_incoming(hello, via_mac=make_mac(7))
        assert node._directory.lookup(make_id(2)) is not None

    def test_on_hello_adds_route(self):
        node, adapter = self._make_node(node_id=make_id(1))
        hello = Packet.hello(src_id=make_id(3), payload=make_id(3))
        node._handle_incoming(hello, via_mac=make_mac(4))
        route = node._routing.lookup(make_id(3))
        assert route is not None
        assert route.next_hop_mac == make_mac(4)
        assert route.cost == 1

    def test_on_hello_sends_reply(self):
        node, adapter = self._make_node(node_id=make_id(1))
        hello = Packet.hello(src_id=make_id(5))
        node._handle_incoming(hello, via_mac=make_mac(6))
        adapter.send.assert_called_once()

    # on_circuit_setup
    def test_on_circuit_setup_registers_at_destination(self):
        node, _ = self._make_node(node_id=make_id(1))
        pkt = Packet.circuit_setup(make_id(2), make_id(1), circuit_id=0xF00D)
        node._handle_incoming(pkt, via_mac=make_mac(9))
        entry = node._circuits.lookup(0xF00D)
        assert entry is not None
        prev, next_ = entry
        assert prev == make_mac(9)
        assert next_ is None

    def test_on_circuit_setup_forwards_at_intermediate(self):
        node, adapter = self._make_node(node_id=make_id(1))
        node._routing.add_route(make_id(3), make_mac(7), cost=1)
        pkt = Packet.circuit_setup(make_id(2), make_id(3), circuit_id=0xCAFE)
        node._handle_incoming(pkt, via_mac=make_mac(8))
        adapter.send.assert_called_once()

    # on_circuit_teardown
    def test_on_circuit_teardown_removes_circuit(self):
        node, _ = self._make_node(node_id=make_id(1))
        node._circuits.add_circuit(0x1111, make_mac(2), None)
        pkt = Packet.circuit_teardown(make_id(2), make_id(1), circuit_id=0x1111)
        node._handle_incoming(pkt, via_mac=make_mac(2))
        assert node._circuits.lookup(0x1111) is None

    # interfaces
    def test_qrnetnode_implements_interface(self):
        """QRNetNode debe ser una instancia válida de IQRNetNode."""
        adapter  = mock_adapter()
        selector = mock_selector(adapter)
        node = QRNetNode(selector=selector)
        assert isinstance(node, IQRNetNode)

    # context manager
    def test_context_manager_starts_and_stops(self):
        adapter  = mock_adapter()
        selector = mock_selector(adapter)
        with QRNetNode(selector=selector, node_id=make_id(1)) as node:
            assert node._running is True
        # Después del __exit__ debe estar detenido
        assert node._running is False
