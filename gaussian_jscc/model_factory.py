"""Explicit opt-in dispatch; historical configuration serialization stays intact."""
from .codec import CodecConfig, GaussianCodec
from .modular_config import ModularCodecConfig
from .modular_codec import ModularGaussianCodec


def config_from_dict(values):
    if values.get('architecture') == 'modular_jscc_v1':
        return ModularCodecConfig.from_dict(values)
    return CodecConfig.from_dict(values)


def build_codec(cfg):
    if isinstance(cfg, ModularCodecConfig):
        if cfg.channel_variant=='successive':
            from .successive_codec import SuccessiveGaussianCodec
            return SuccessiveGaussianCodec(cfg)
        return ModularGaussianCodec(cfg)
    if isinstance(cfg, CodecConfig):
        return GaussianCodec(cfg)
    raise ValueError('unsupported codec config')
