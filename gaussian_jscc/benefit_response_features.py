"""Keep exact MSE response terms from the existing candidate render pass."""
import math
import time

import torch

from .benefit_footprint_features import FootprintWindow
from .pixel_benefit import response_density_labels
from .pixel_benefit_features import _label_chunks, pixel_coordinates

SCHEMA = 'exact_cross_energy_structure_v1'


@torch.no_grad()
def exact_response_terms(scene, changed_ids, projected, target, lambda_s, d_ref):
    if not math.isfinite(float(d_ref)) or d_ref <= 0:
        raise ValueError('positive finite d_ref required')
    ids = torch.as_tensor(changed_ids, device=scene.projected.device, dtype=torch.long)
    affected = scene.affected_tiles(ids, projected)
    tiles = scene.affected_tiles(ids, projected, halo_tiles=1)
    value = torch.zeros(3,3,dtype=torch.float64,device=scene.projected.device)
    for chosen, density in _label_chunks(scene, ids, projected, target, lambda_s, tiles, affected,
                                        density_fn=response_density_labels):
        x,y = pixel_coordinates(chosen, scene.width, halo=0)
        valid = ((x>=0)&(y>=0)&(x<scene.width)&(y<scene.height))[:,None,None]
        value += (density*valid).sum((0,3,4))/(scene.height*scene.width*d_ref)
    return value


class ResponseTermsWindow(FootprintWindow):
    def build_view(self, tile, changed, projected, target, *, source, block_ids, raw, with_labels):
        view = super().build_view(tile, changed, projected, target, source=source,
            block_ids=block_ids, raw=raw, with_labels=False)
        if with_labels:
            started = time.perf_counter()
            terms = exact_response_terms(tile, changed, projected, target,
                                         self.reference['lambda_s'],self.reference['d_ref']).cpu()
            view['response_terms'] = terms
            view['values'] = torch.stack((terms[:,0]+terms[:,1],terms[:,2]),-1)
            view['label_schema'] = SCHEMA
            view['teacher_seconds'] = time.perf_counter()-started
            view['tile_count'] = len(tile.affected_tiles(changed, projected, halo_tiles=1))
        return view
