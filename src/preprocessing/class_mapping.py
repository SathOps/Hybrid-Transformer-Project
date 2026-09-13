"""Explicit CICIoT2023 source-label mapping for the project taxonomy."""

from __future__ import annotations


import numpy as np

CANONICAL_TARGET_CLASSES = (
    "Benign",
    "BruteForce",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "Web-based",
)

ACTIVE_TARGET_CLASSES = (
    "Benign",
    "BruteForce",
    "DDoS",
    "DoS",
    "Mirai",
    "Spoofing",
    "Web-based",
)

ALLOWED_TARGET_CLASSES = CANONICAL_TARGET_CLASSES

NUM_CANONICAL_CLASSES = len(CANONICAL_TARGET_CLASSES)  # 8
NUM_ACTIVE_CLASSES = len(ACTIVE_TARGET_CLASSES)        # 7

CANONICAL_ID_TO_ACTIVE_INDEX = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6}
ACTIVE_INDEX_TO_CANONICAL_ID = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7}

_CANONICAL_TO_ACTIVE_LUT = np.array([0, 1, 2, 3, 4, -1, 5, 6], dtype=np.int64)
_ACTIVE_TO_CANONICAL_LUT = np.array([0, 1, 2, 3, 4, 6, 7], dtype=np.int64)


def canonical_to_active_labels(y: np.ndarray) -> np.ndarray:
    """Map canonical label IDs (0,1,2,3,4,6,7) to contiguous active model output targets (0..6)."""
    y_arr = np.asarray(y, dtype=np.int64)
    mapped = _CANONICAL_TO_ACTIVE_LUT[y_arr]
    if (mapped == -1).any():
        raise ValueError("Input labels contain inactive canonical label ID 5 (Recon).")
    return mapped


def active_to_canonical_labels(y: np.ndarray) -> np.ndarray:
    """Map active model output prediction targets (0..6) back to canonical label IDs (0,1,2,3,4,6,7)."""
    y_arr = np.asarray(y, dtype=np.int64)
    return _ACTIVE_TO_CANONICAL_LUT[y_arr]



SOURCE_TO_TARGET = {
    "BenignTraffic": "Benign",
    "Benign_Final": "Benign",
    "BrowserHijacking": "Web-based",
    "CommandInjection": "Web-based",
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-PSHACK_FLOOD": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-RSTFINFLOOD": "DDoS",
    "DDoS-RSTFINFlood": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS",
    "DNS_Spoofing": "Spoofing",
    "DictionaryBruteForce": "BruteForce",
    "DoS-HTTP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS",
    "DoS-TCP_Flood": "DoS",
    "DoS-UDP_Flood": "DoS",
    "MITM-ArpSpoofing": "Spoofing",
    "Mirai-greeth_flood": "Mirai",
    "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
    "Recon-HostDiscovery": "Recon",
    "Recon-OSScan": "Recon",
    "Recon-PingSweep": "Recon",
    "Recon-PortScan": "Recon",
    "SqlInjection": "Web-based",
    "Uploading_Attack": "Web-based",
    "XSS": "Web-based",
}


# No authoritative 34-to-8 rule was found for these labels. They are
# quarantined and excluded from the eight-class modeling dataset.
UNRESOLVED_SOURCE_LABELS = frozenset(
    {
        "Backdoor_Malware",
        "VulnerabilityScan",
    }
)


EXPECTED_SOURCE_LABELS = (
    "Backdoor_Malware",
    "BenignTraffic",
    "Benign_Final",
    "BrowserHijacking",
    "CommandInjection",
    "DDoS-ACK_Fragmentation",
    "DDoS-HTTP_Flood",
    "DDoS-ICMP_Flood",
    "DDoS-ICMP_Fragmentation",
    "DDoS-PSHACK_FLOOD",
    "DDoS-PSHACK_Flood",
    "DDoS-RSTFINFLOOD",
    "DDoS-RSTFINFlood",
    "DDoS-SYN_Flood",
    "DDoS-SlowLoris",
    "DDoS-SynonymousIP_Flood",
    "DDoS-TCP_Flood",
    "DDoS-UDP_Flood",
    "DDoS-UDP_Fragmentation",
    "DNS_Spoofing",
    "DictionaryBruteForce",
    "DoS-HTTP_Flood",
    "DoS-SYN_Flood",
    "DoS-TCP_Flood",
    "DoS-UDP_Flood",
    "MITM-ArpSpoofing",
    "Mirai-greeth_flood",
    "Mirai-greip_flood",
    "Mirai-udpplain",
    "Recon-HostDiscovery",
    "Recon-OSScan",
    "Recon-PingSweep",
    "Recon-PortScan",
    "SqlInjection",
    "Uploading_Attack",
    "VulnerabilityScan",
    "XSS",
)


def map_label(label: str) -> str | None:
    """Map a known source label or return ``None`` for quarantined labels."""
    if label in SOURCE_TO_TARGET:
        return SOURCE_TO_TARGET[label]
    if label in UNRESOLVED_SOURCE_LABELS:
        return None
    raise ValueError(f"Unexpected CICIoT2023 source label: {label!r}")