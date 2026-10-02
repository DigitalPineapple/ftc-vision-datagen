"""
Minimal, dependency-free writer for TensorFlow's TFRecord container format and
the tf.train.Example protobuf schema used by the TensorFlow Object Detection
API (and expected by Roboflow / the Limelight Neural Network Trainer).

Implemented by hand-encoding the small, fixed Example/Features/Feature
protobuf schema with raw protobuf wire-format bytes, so this project doesn't
need a `tensorflow` or `protobuf` dependency just to produce a .tfrecord file.

Reference schema (tensorflow/core/example/{feature,example}.proto):
    message BytesList { repeated bytes value = 1; }
    message FloatList  { repeated float value = 1 [packed = true]; }
    message Int64List  { repeated int64 value = 1 [packed = true]; }
    message Feature {
        oneof kind {
            BytesList bytes_list = 1;
            FloatList float_list = 2;
            Int64List int64_list = 3;
        }
    }
    message Features { map<string, Feature> feature = 1; }
    message Example { Features features = 1; }

TFRecord container format (per record):
    uint64 length
    uint32 masked_crc32c(length bytes)
    byte   data[length]
    uint32 masked_crc32c(data)
"""
from __future__ import annotations

import struct

# --- CRC32C (Castagnoli), used (masked) for TFRecord framing checksums ---

_CRC32C_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (0x82F63B78 ^ (_c >> 1)) if (_c & 1) else (_c >> 1)
    _CRC32C_TABLE.append(_c)


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _masked_crc32c(data: bytes) -> int:
    crc = _crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def write_tfrecord(f, data: bytes) -> None:
    """Append one length-delimited, checksummed record to an open binary file."""
    length_bytes = struct.pack("<Q", len(data))
    f.write(length_bytes)
    f.write(struct.pack("<I", _masked_crc32c(length_bytes)))
    f.write(data)
    f.write(struct.pack("<I", _masked_crc32c(data)))


# --- Minimal protobuf wire-format helpers ---

def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _tag(field_number: int, wire_type: int) -> bytes:
    return _varint((field_number << 3) | wire_type)


def _len_delimited(field_number: int, payload: bytes) -> bytes:
    return _tag(field_number, 2) + _varint(len(payload)) + payload


def bytes_list_feature(values) -> bytes:
    """Feature{bytes_list=BytesList{value: [...]}} for a list of bytes/str."""
    payload = b""
    for v in values:
        if isinstance(v, str):
            v = v.encode("utf-8")
        payload += _len_delimited(1, v)
    return _len_delimited(1, payload)  # Feature.bytes_list = field 1


def float_list_feature(values) -> bytes:
    """Feature{float_list=FloatList{value: [...]}} (packed repeated float)."""
    packed = b"".join(struct.pack("<f", float(v)) for v in values)
    payload = _len_delimited(1, packed)
    return _len_delimited(2, payload)  # Feature.float_list = field 2


def int64_list_feature(values) -> bytes:
    """Feature{int64_list=Int64List{value: [...]}} (packed repeated varint)."""
    packed = b"".join(_varint(int(v)) for v in values)
    payload = _len_delimited(1, packed)
    return _len_delimited(3, payload)  # Feature.int64_list = field 3


def build_example(feature_dict: dict) -> bytes:
    """
    feature_dict: {name: encoded Feature bytes (from one of the *_feature
    helpers above)}. Returns a serialized Example message.
    """
    features_payload = b""
    for name, feature_bytes in feature_dict.items():
        entry = _len_delimited(1, name.encode("utf-8")) + _len_delimited(2, feature_bytes)
        features_payload += _len_delimited(1, entry)  # Features.feature map entry = field 1
    return _len_delimited(1, features_payload)  # Example.features = field 1
