"""Independent XYZ-in-payload identity; never masquerades as reliable XYZ."""
from dataclasses import asdict, dataclass
from .modular_config import ModularCodecConfig


@dataclass
class XYZPayloadConfig:
    architecture: str = 'xyz_payload_jscc_v1'
    attribute_latent_dim: int = 48
    latent_dim: int = 51
    sh_degree: int = 3
    hidden: int = 96
    grid_dim: int = 16
    levels: tuple = (4,8)
    planes: bool = False
    depth: int = 2
    block_size: int = 256
    morton_bits: int = 16
    decoder_window: int = 32
    attention_heads: int = 4
    context_variant: str = 'both'
    context_centers: tuple = (64,16)
    context_neighbors: tuple = (16,32)
    context_radii: tuple = (2.,4.)
    rates: tuple = (0,8,24,48)
    channel_variant: str = 'progressive'
    channel_hidden: int = 192
    channel_prefix_mode: str = 'full_prefix'
    channel_denoiser_mode: str = 'none'
    channel_neighbor_mode: str = 'none'
    power_floor: float = .01
    position_delivery: str = 'payload_xyz'
    decoder_geometry: str = 'xyz'
    prefix_mode: str = 'progressive'

    def __post_init__(self):
        if (self.architecture!='xyz_payload_jscc_v1' or self.position_delivery!='payload_xyz'
                or self.channel_variant!='progressive' or self.channel_prefix_mode!='full_prefix'
                or self.channel_denoiser_mode!='none' or self.channel_neighbor_mode!='none'
                or self.decoder_geometry!='xyz' or self.prefix_mode!='progressive'):
            raise ValueError('XYZ payload requires progressive full prefixes, no external XYZ or channel spatial modules')
        if (type(self.attribute_latent_dim) is not int or self.attribute_latent_dim<1
                or type(self.latent_dim) is not int or self.latent_dim!=3+self.attribute_latent_dim):
            raise ValueError('payload dimension must equal 3 + attribute_latent_dim')
        for name in ('rates','levels','context_centers','context_neighbors','context_radii'):
            setattr(self,name,tuple(getattr(self,name)))
        self.attribute_config()  # Reuse existing validated component settings.

    @property
    def attr_dim(self):
        return 8+3*(self.sh_degree+1)**2

    def attribute_config(self):
        names=('sh_degree','hidden','grid_dim','levels','planes','depth','block_size','morton_bits',
               'decoder_window','attention_heads','context_variant','context_centers','context_neighbors',
               'context_radii','rates','channel_hidden','power_floor')
        return ModularCodecConfig(**{n:getattr(self,n) for n in names},latent_dim=self.attribute_latent_dim,
                                  channel_variant='progressive',channel_prefix_mode='full_prefix')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)
