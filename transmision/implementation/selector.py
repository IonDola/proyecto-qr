"""
capa1/selector.py
AdapterSelector — implementación de IAdapterSelector.

Responsable de elegir el INetworkAdapter correcto para cada paquete.
QRNetNode llama a select(dst_mac) antes de enviar cualquier paquete.

Lógica de selección (por política):
  PREFER_QR       → QR_LIGHT primero, fallback a TCP
  PREFER_TCP      → ETHERNET/WIFI primero, fallback a QR
  LOWEST_COST     → el adaptador con menor get_cost() disponible
  FIRST_AVAILABLE → el primero en la lista que esté disponible

Caso especial: is_qr_required(dst_mac) fuerza el canal QR_LIGHT
cuando el destino es inter-ciudad (MAC broadcast o en tabla de gateways).
"""
from __future__ import annotations

from common.network_policies import AdapterType, SelectionPolicy
from common.exceptions import AdapterError
from ..interfaces import INetworkAdapter, IAdapterSelector


class AdapterSelector(IAdapterSelector):
    """
    Selector de canal con política configurable en tiempo de ejecución.

    Uso:
        selector = AdapterSelector(policy=SelectionPolicy.PREFER_QR)
        selector.register(luz_adaptador)
        selector.register(tcp_adaptador)

        adapter = selector.select(dst_mac)
        adapter.send(data)
    """

    # MACs que siempre requieren canal QR (inter-ciudad o broadcast)
    QR_FORCED_MACS: frozenset[bytes] = frozenset([
        b'\xff\xff\xff\xff\xff\xff',   # broadcast
    ])

    def __init__(
        self,
        policy: SelectionPolicy = SelectionPolicy.PREFER_QR,
    ) -> None:
        self._adapters: list[INetworkAdapter] = []
        self._policy = policy
        # Tabla de MACs remotas que requieren canal QR (gateways inter-ciudad)
        self._qr_gateway_macs: set[bytes] = set()

    # ── IAdapterSelector ──────────────────────────────────────────────────────

    def select(self, dst_mac: bytes) -> INetworkAdapter:
        """
        Retorna el adaptador más adecuado para dst_mac según la política activa.
        Lanza AdapterError si ningún adaptador está disponible.
        """
        available = [a for a in self._adapters if a.is_available()]
        if not available:
            raise AdapterError("Ningún adaptador disponible")

        # Caso especial: destino inter-ciudad siempre usa QR
        if self._is_qr_required(dst_mac):
            qr = self._first_of_type(available, AdapterType.QR_LIGHT)
            if qr:
                return qr
            raise AdapterError("Se requiere canal QR_LIGHT pero no está disponible")

        return self._apply_policy(available)

    def register(self, adapter: INetworkAdapter) -> None:
        """Añade un adaptador al pool. Los duplicados (mismo MAC) se reemplazan."""
        mac = adapter.get_mac()
        self._adapters = [a for a in self._adapters if a.get_mac() != mac]
        self._adapters.append(adapter)

    def set_policy(self, policy: SelectionPolicy) -> None:
        self._policy = policy

    # ── helpers públicos ──────────────────────────────────────────────────────

    def register_qr_gateway(self, remote_mac: bytes) -> None:
        """
        Marca una MAC remota como gateway inter-ciudad.
        Todo paquete hacia esa MAC usará forzosamente el canal QR.
        """
        self._qr_gateway_macs.add(remote_mac)

    def unregister_qr_gateway(self, remote_mac: bytes) -> None:
        self._qr_gateway_macs.discard(remote_mac)

    def available_adapters(self) -> list[INetworkAdapter]:
        """Retorna los adaptadores actualmente disponibles (útil para debug)."""
        return [a for a in self._adapters if a.is_available()]

    # ── lógica interna ────────────────────────────────────────────────────────

    def _is_qr_required(self, dst_mac: bytes) -> bool:
        """True si el destino exige canal QR (broadcast o gateway inter-ciudad)."""
        return dst_mac in self.QR_FORCED_MACS or dst_mac in self._qr_gateway_macs

    def _apply_policy(self, available: list[INetworkAdapter]) -> INetworkAdapter:
        """Aplica la política activa sobre los adaptadores disponibles."""
        if self._policy == SelectionPolicy.PREFER_QR:
            return (
                self._first_of_type(available, AdapterType.QR_LIGHT)
                or self._lowest_cost(available)
            )

        if self._policy == SelectionPolicy.PREFER_TCP:
            tcp = (
                self._first_of_type(available, AdapterType.ETHERNET)
                or self._first_of_type(available, AdapterType.WIFI)
            )
            return tcp or self._lowest_cost(available)

        if self._policy == SelectionPolicy.LOWEST_COST:
            return self._lowest_cost(available)

        # FIRST_AVAILABLE
        return available[0]

    @staticmethod
    def _first_of_type(
        adapters: list[INetworkAdapter], atype: AdapterType
    ) -> INetworkAdapter | None:
        return next((a for a in adapters if a.get_type() == atype), None)

    @staticmethod
    def _lowest_cost(adapters: list[INetworkAdapter]) -> INetworkAdapter:
        return min(adapters, key=lambda a: a.get_cost())