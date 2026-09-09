"""Explicit CICIoT2023 source-label mapping for the project taxonomy."""

from __future__ import annotations


ALLOWED_TARGET_CLASSES = (
    "Benign",
    "BruteForce",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "Web-based",
)


SOURCE_TO_TARGET = {
    "BenignTraffic": "Benign",
    "BrowserHijacking": "Web-based",
    "CommandInjection": "Web-based",
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS",
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
    "BrowserHijacking",
    "CommandInjection",
    "DDoS-ACK_Fragmentation",
    "DDoS-HTTP_Flood",
    "DDoS-ICMP_Flood",
    "DDoS-ICMP_Fragmentation",
    "DDoS-PSHACK_Flood",
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