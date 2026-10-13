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

    def __post_init__(self):
        if self.architecture != 'modular_jscc_v1':
            raise ValueError('expected modular_jscc_v1 architecture')
        if type(self.latent_dim) is not int or self.latent_dim < 1:
            raise ValueError('latent_dim must be a positive integer')
        if (self.position_delivery != 'quantized' or self.position_bits != 16 or
                self.prefix_mode != 'progressive' or self.decoder_mode != 'direct' or
                tuple(self.rates) != (0, 4, 12, 24)):
            raise ValueError('modular experiment requires reliable 16bit XYZ and 4/12/24 progressive prefixes')
        # Composition reuses transport validation without subclassing legacy config.
        values = asdict(self)
        values.pop('latent_dim')
        for key in ('context_variant','context_centers','context_neighbors','context_radii'):
            values.pop(key)
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
        return values

    @classmethod
    def from_dict(cls, values):
        return cls(**values)
