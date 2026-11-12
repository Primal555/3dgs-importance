"""Student RD accounting using the actual three-file delivery protocol."""
import torch
from comparison_baselines.digital_link import LinkConfig
from comparison_baselines.side_information_accounting import side_information_cost
from .budgeted_allocation_evaluation import ExactDeploymentCost
from .position_delivery import PositionCostMeter
from .successive_delivery_packet import packet_header
from .transport import encode_metadata

PROTOCOL = 'digital_side_file_v1'


class DigitalDeploymentCost(ExactDeploymentCost):
    def __init__(self, model, scene, codec_id, snr, bits):
        super().__init__(model, scene, codec_id, snr, bits)
        self.positions = PositionCostMeter(model.cfg, torch.cat([
            f[..., :3][ids>=0] for f,ids in zip(scene.groups,scene.group_ids)]))

    def details(self, original_q):
        row = super().details(original_q)
        q = original_q.detach().cpu()[self.scene.order]
        header = packet_header(self.model, self.scene, q, self.snr, codec_id=self.codec_id)
        row['metadata_bytes'] = len(encode_metadata(header, q.numpy()))
        accounting = side_information_cost(row)
        streams = accounting['streams']
        row.update(coordinate_channel_uses=streams['coordinates']['total_channel_uses'],
            metadata_channel_uses=streams['metadata']['total_channel_uses'],
            scale_channel_uses=streams['scales']['total_channel_uses'],
            total_channel_uses=accounting['total_channel_uses'], comparison_accounting=accounting,
            cost_protocol=PROTOCOL, rate_scope=accounting['rate_scope'],
            reliability='digital file cost measured; actual delivery tested separately')
        row.pop('net_bits_per_use_assumption', None)
        return row


def digital_side_proxy(cost, retained, fallback, count):
    """Amortized gradient proxy, exact at refresh; file padding is discrete."""
    link = LinkConfig()
    bits = (link.k-32)/(link.n/link.bits_per_symbol)
    scale_content = cost['scale_content_bits']/bits
    return dict(
        xyz_per_retained=cost['coordinate_channel_uses']/retained if retained else fallback['coordinate_channel_uses']/count,
        scale_per_retained=scale_content/retained if retained else 0.,
        fixed_uses=cost['metadata_channel_uses']+cost['scale_channel_uses']-scale_content+
            (cost['coordinate_channel_uses'] if not retained else 0.),
        bits_per_use=bits, cost_protocol=PROTOCOL,
        proxy_scope='amortized XYZ and scale content; exact files at refresh/evaluation; padding and compression changes are not differentiable')
