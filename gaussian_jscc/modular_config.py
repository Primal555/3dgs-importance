"""Opt-in modular experiment configuration; never changes legacy model hashes."""
from dataclasses import asdict, dataclass
import math
from .codec import CodecConfig


@dataclass
class ModularCodecConfig:
    architecture: str = 'modular_jscc_v1'
    latent_dim: int = 48
    sh_degree: int = 3
    hidden: int = 96
    grid_dim: int = 16
    levels: tuple = (4, 8)
    planes: bool = False
    depth: int = 2
    rates: tuple = (0, 4, 12, 24)
    block_size: int = 256
    morton_bits: int = 16
    decoder_window: int = 32
    attention_heads: int = 4
    decoder_geometry: str = 'xyz'
    power_floor: float = .01
    position_delivery: str = 'quantized'
    position_bits: int = 16
    position_compression: str = 'delta_zlib'
    position_compression_level: int = 6
    prefix_mode: str = 'progressive'
    decoder_mode: str = 'direct'
    loss_profile: str = 'learned_v1'
    position_head: str = 'learned_affine'
    individual_tiers: bool = True
    xyz_loss_scale: float = .05
    geometry_weight: float = 1.
    shape_weight: float = .25
    scale_weight: float = 1.
    opacity_weight: float = 1.
    dc_weight: float = 1.
    sh_weight: float = .25
    geometry_rates: tuple = ()
    geometry_floor: float = 1e-4
    context_variant: str = 'baseline'
    context_centers: tuple = (64, 16)
    context_neighbors: tuple = (16, 32)
    context_radii: tuple = (2., 4.)
    channel_variant: str = 'baseline'
    channel_hidden: int = 192
    channel_prefix_mode: str = 'suffix'
    channel_denoiser_mode: str = 'none'
    channel_denoiser_hidden: int = 64
    channel_denoiser_neighbors: int = 16
    channel_denoiser_scope: str = 'stage'
    channel_neighbor_mode: str = 'none'
    channel_neighbor_hidden: int = 64
    channel_neighbor_neighbors: int = 16

    def __post_init__(self):
        if self.architecture != 'modular_jscc_v1':
            raise ValueError('expected modular_jscc_v1 architecture')
        if type(self.latent_dim) is not int or self.latent_dim < 1:
            raise ValueError('latent_dim must be a positive integer')
        if (self.position_delivery != 'quantized' or self.position_bits != 16 or
                self.prefix_mode != 'progressive' or self.decoder_mode != 'direct'):
            raise ValueError('modular experiment requires reliable 16bit XYZ and progressive prefixes')
        self.rates = tuple(self.rates)
        if (len(self.rates) != 4 or any(type(r) is not int for r in self.rates)
                or self.rates[0] != 0 or any(a >= b for a, b in zip(self.rates, self.rates[1:]))):
            raise ValueError('modular rates require q0=0 and three strictly increasing positive integer budgets')
        # Composition reuses transport validation without subclassing legacy config.
        values = asdict(self)
        values.pop('latent_dim')
        for key in ('context_variant','context_centers','context_neighbors','context_radii','channel_variant','channel_hidden','channel_prefix_mode',
                    'channel_denoiser_mode','channel_denoiser_hidden','channel_denoiser_neighbors','channel_denoiser_scope',
                    'channel_neighbor_mode','channel_neighbor_hidden','channel_neighbor_neighbors'):
            values.pop(key)
        if (self.channel_neighbor_mode not in ('none','plain','feature') or
                type(self.channel_neighbor_hidden) is not int or self.channel_neighbor_hidden<1 or
                type(self.channel_neighbor_neighbors) is not int or self.channel_neighbor_neighbors<1):
            raise ValueError('invalid receiver neighbor refinement settings')
        if self.channel_neighbor_mode!='none' and (self.channel_variant!='progressive' or self.channel_prefix_mode!='full_prefix'):
            raise ValueError('neighbor refinement requires full-prefix progressive channel modules')
        if (self.channel_denoiser_mode not in ('none','pointwise','spatial') or
                type(self.channel_denoiser_hidden) is not int or self.channel_denoiser_hidden<1 or
                type(self.channel_denoiser_neighbors) is not int or self.channel_denoiser_neighbors<1):
            raise ValueError('invalid receiver denoiser settings')
        if self.channel_denoiser_mode!='none' and (self.channel_variant!='progressive' or self.channel_prefix_mode!='full_prefix'):
            raise ValueError('receiver denoising requires full-prefix progressive channel modules')
        if (self.channel_denoiser_scope not in ('stage','final') or
                (self.channel_denoiser_scope=='final' and self.channel_denoiser_mode=='none')):
            raise ValueError('final receiver denoising requires an enabled denoiser')
        if self.channel_variant not in ('baseline','residual','progressive'):
            raise ValueError('unknown channel variant')
        if (self.channel_prefix_mode not in ('suffix','full_prefix') or
                (self.channel_prefix_mode=='full_prefix' and self.channel_variant!='progressive')):
            raise ValueError('full-prefix reread requires progressive channel modules')
        if type(self.channel_hidden) is not int or self.channel_hidden<1:
            raise ValueError('channel_hidden must be a positive integer')
        if self.channel_variant=='baseline': self.channel_hidden=192
        if self.context_variant not in ('baseline','encoder','decoder','both'):
            raise ValueError('unknown spatial context variant')
        if self.context_variant in ('decoder','both') and self.decoder_geometry != 'xyz':
            raise ValueError('spatial decoder requires reliable XYZ')
        self.context_centers = tuple(self.context_centers)
        self.context_neighbors = tuple(self.context_neighbors)
        self.context_radii = tuple(self.context_radii)
        if (len(self.context_centers)!=2 or any(type(n) is not int or n<1 for n in self.context_centers)
                or self.context_centers[0]<self.context_centers[1]
                or len(self.context_neighbors)!=2 or any(type(n) is not int or n<1 for n in self.context_neighbors)
                or len(self.context_radii)!=2 or any(not math.isfinite(r) or r<=0 for r in self.context_radii)
                or self.context_radii[0]>=self.context_radii[1]):
            raise ValueError('invalid PointNet context centers/neighbors/radii')
        values['architecture'] = 'learned_joint'
        common = CodecConfig(**values)
        self.rates, self.levels, self.geometry_rates = common.rates, common.levels, common.geometry_rates

    @property
    def attr_dim(self):
        return 8 + 3 * (self.sh_degree + 1) ** 2

    def to_dict(self):
        values = asdict(self)
        # Baseline checkpoints/packet identities keep their historical syntax.
        if self.context_variant == 'baseline':
            for key in ('context_variant','context_centers','context_neighbors','context_radii'):
                values.pop(key)
        if self.channel_variant == 'baseline':
            values.pop('channel_variant'); values.pop('channel_hidden')
        if self.channel_prefix_mode == 'suffix': values.pop('channel_prefix_mode')
        if self.channel_denoiser_mode=='none':
            for key in ('channel_denoiser_mode','channel_denoiser_hidden','channel_denoiser_neighbors'): values.pop(key)
        if self.channel_denoiser_scope=='stage': values.pop('channel_denoiser_scope')
        if self.channel_neighbor_mode=='none':
            for key in ('channel_neighbor_mode','channel_neighbor_hidden','channel_neighbor_neighbors'):values.pop(key)
        return values

    @classmethod
    def from_dict(cls, values):
        return cls(**values)
