class QRNetError(Exception):
    """Raíz de todas las excepciones de QR-NET."""

class ChecksumError(QRNetError):
    """CRC-16 o CRC-32 no coincide con el payload recibido."""

class HandshakeError(QRNetError):
    """Fallo durante la negociación SYN / SYN-ACK / ACK."""

class AdapterError(QRNetError):
    """Error en un NetworkAdapter (cámara no disponible, socket caído, etc.)."""

class CameraError(AdapterError):
    """Error específico de un NetworkAdapter de tipo QR_LIGHT."""

class EthernetError(AdapterError):
    """Error específico de un NetworkAdapter de tipo ETHERNET."""

class WifiError(AdapterError):
    """Error específico de un NetworkAdapter de tipo WIFI."""

class RoutingError(QRNetError):
    """No se encontró ruta hacia el destino en la mesh."""

class CompressionError(QRNetError):
    """Fallo al comprimir un bloque de datos."""

class DecompressionError(QRNetError):
    """Fallo al descomprimir un bloque de datos."""

class GridDecodeError(QRNetError):
    """No se pudo decodificar la grilla 64x64 desde la imagen capturada."""

class FrameFormatError(QRNetError):
    """Formato de frame inválido (payload demasiado largo, tipo desconocido, etc.)."""

class TimeoutError(QRNetError):
    """Tiempo de espera agotado esperando un evento (respuesta, ACK, etc.)."""