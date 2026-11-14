"""Public SNR-to-MCS policy shared by accounting and actual file transmission."""
import math

from .digital_link import LinkConfig

POLICIES = ('fixed_16qam_v1', 'snr_qpsk_half_v1')


def side_link_config(snr_db, policy='fixed_16qam_v1', device='cuda'):
    if policy not in POLICIES or not math.isfinite(float(snr_db)):
        raise ValueError('known side-link policy and finite Es/N0 required')
    # This public rule uses only link CSI, never image quality or decoded CRCs.
    # Low-SNR packets retain the same 48-byte header and 508-byte payload.
    if policy == 'snr_qpsk_half_v1' and snr_db < 10.:
        return LinkConfig(k=4096, n=8192, bits_per_symbol=2, device=device)
    return LinkConfig(device=device)
