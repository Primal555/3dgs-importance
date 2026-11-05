"""Local relation values in original PLY row order, with bounded CPU graph gathers."""

import torch
from torch import nn


class _RelationAttention(nn.Module):
    """Grouped vector attention; callers append a valid self token for isolated rows."""

    def __init__(self, width, groups, geometry_dim, dropout):
        super().__init__()
        self.groups = groups
        self.group_width = width // groups
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.geometry = nn.Sequential(nn.Linear(geometry_dim, width), nn.GELU(),
                                      nn.Linear(width, width))
        self.logits = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                    nn.Linear(width, groups))
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(width * 2, width))

    def forward(self, center, neighbors, geometry, valid):
        position = self.geometry(geometry)
        relation = self.query(center).unsqueeze(-2) - self.key(neighbors) + position
        scores = self.logits(relation).masked_fill(~valid.unsqueeze(-1), -torch.inf)
        weights = scores.softmax(dim=-2)
        values = (self.value(neighbors) + position).unflatten(-1, (self.groups, self.group_width))
        mixed = (weights.unsqueeze(-1) * values).sum(-3).flatten(-2)
        result = self.norm1(center + self.dropout(self.output(mixed)))
        return self.norm2(result + self.dropout(self.feedforward(result)))


class _CandidateInteraction(nn.Module):
    def __init__(self, width, groups, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, groups, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(width * 2, width))

    def forward(self, tokens):
        context, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        result = self.norm1(tokens + self.dropout(context))
        return self.norm2(result + self.dropout(self.feedforward(result)))


class RelationBenefitNetwork(nn.Module):
    """Signed q1--q3 values relative to the q0 anchor, using complete local context.

    The bank stays on CPU. A batch first collects projection neighbors, then two
    spatial hops for every collected row. Spatial layers only evaluate the rows
    whose exact predecessor context is present, so batching never cuts edges.
    """

    def __init__(self, node_dim, candidate_dim, view_dim, width=128, groups=8, dropout=0.,
                 *, context_node_dim=0, context_view_dim=0):
        super().__init__()
        if width <= 0 or groups <= 0 or width % groups:
            raise ValueError("width must be positive and divisible by groups")
        self.register_buffer("node_mean", torch.zeros(node_dim))
        self.register_buffer("node_std", torch.ones(node_dim))
        self.register_buffer("candidate_mean", torch.zeros(candidate_dim))
        self.register_buffer("candidate_std", torch.ones(candidate_dim))
        self.register_buffer("view_mean", torch.zeros(view_dim))
        self.register_buffer("view_std", torch.ones(view_dim))
        self.register_buffer("output_scale", torch.ones(2))
        if bool(context_node_dim)!=bool(context_view_dim) or min(context_node_dim,context_view_dim)<0:
            raise ValueError('both context feature dimensions must be positive or zero')
        self.contextual=bool(context_node_dim)
        if self.contextual:
            for name,dim in (('context_node',context_node_dim),('context_view',context_view_dim)):
                self.register_buffer(name+'_mean',torch.zeros(dim))
                self.register_buffer(name+'_std',torch.ones(dim))
                setattr(self,name+'_encoder',nn.Sequential(nn.Linear(dim,width),nn.GELU(),nn.Linear(width,width)))
        self.node_encoder = nn.Sequential(nn.Linear(node_dim, width), nn.GELU(),
                                          nn.Linear(width, width))
        self.candidate_encoder = nn.Sequential(nn.Linear(candidate_dim, width), nn.GELU(),
                                               nn.Linear(width, width))
        self.view_encoder = nn.Sequential(nn.Linear(view_dim + 7, width), nn.GELU(),
                                          nn.Linear(width, width))
        self.tier_embedding = nn.Parameter(torch.empty(4, width))
        nn.init.normal_(self.tier_embedding, std=.02)
        self.spatial_layers = nn.ModuleList([
            _RelationAttention(width, groups, 3, dropout) for _ in range(2)])
        self.projection_layer = _RelationAttention(width, groups, 10, dropout)
        self.candidate_interaction = _CandidateInteraction(width, groups, dropout)
        self.common_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
        self.advantage_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))

    @torch.no_grad()
    def fit_normalizers(self, bank, training_ids):
        """Fit only designated training PLY rows; views and tiers share scales."""
        ids = torch.as_tensor(training_ids, dtype=torch.long, device="cpu").flatten()
        if not ids.numel():
            raise ValueError("training_ids must contain at least one row")
        samples = {"node": bank["node"][ids],
                   "candidate": bank["candidate"][ids],
                   "view": bank["view_node"][:, ids]}
        if self.contextual:
            samples['context_node']=torch.stack([c['node'][ids] for c in bank['contexts']])
            samples['context_view']=torch.stack([c['view_node'][:,ids] for c in bank['contexts']])
        for name, sample in samples.items():
            flat = sample.reshape(-1, sample.shape[-1]).to(torch.float64)
            mean = getattr(self, name + "_mean")
            std = getattr(self, name + "_std")
            fitted_mean = flat.mean(0)
            floor = (fitted_mean.abs() * 1e-3).clamp_min(1e-3)
            fitted_std = torch.maximum(flat.std(0, unbiased=False), floor)
            mean.copy_(fitted_mean.to(device=mean.device, dtype=mean.dtype))
            std.copy_(fitted_std.to(device=std.device, dtype=std.dtype))
        return self

    @staticmethod
    def _include_neighbors(ids, index):
        neighbors = index[ids].flatten()
        return torch.unique(torch.cat((ids, neighbors[neighbors >= 0])), sorted=True)

    @staticmethod
    def _positions(sorted_ids, rows):
        # The closure is built on CPU; transfer compact local indices afterwards.
        return torch.searchsorted(sorted_ids, rows.clamp_min(0)).clamp_max(len(sorted_ids) - 1)

    def _take(self, tensor, ids):
        return tensor[ids].to(device=self.output_scale.device, dtype=self.output_scale.dtype)

    def _candidate_tokens(self, candidates):
        return self.candidate_encoder(self._normalize(candidates, "candidate")) + self.tier_embedding

    def _normalize(self, value, name):
        normalized = (value - getattr(self, name + "_mean")) / getattr(self, name + "_std")
        return torch.asinh(normalized)

    def _context(self,bank,context):
        if not self.contextual:return None
        if not isinstance(context,int) or not 0<=context<len(bank.get('contexts',[])):
            raise ValueError('explicit valid relation context required')
        return bank['contexts'][context]

    def _context_nodes(self,context,ids):
        return self.context_node_encoder(self._normalize(self._take(context['node'],ids),'context_node'))

    def _context_views(self,context,ids,view=None):
        rows=context['view_node'][:,ids] if view is None else context['view_node'][view,ids]
        rows=rows.to(self.output_scale)
        return self.context_view_encoder(self._normalize(rows,'context_view'))

    def _decode_candidates(self, context, candidates):
        tokens = context.unsqueeze(-2) + candidates
        prefix = tokens.shape[:-2]
        tokens = self.candidate_interaction(tokens.reshape(-1, 4, tokens.shape[-1]))
        tokens = tokens.reshape(*prefix, 4, tokens.shape[-1])
        common = self.common_head(tokens[..., 0, :]) * self.output_scale
        raw_advantage = self.advantage_head(tokens[..., 1:, :]) * self.output_scale
        return common, raw_advantage - raw_advantage.mean(-2, keepdim=True)

    def _spatial_step(self, layer, bank, target_ids, source_ids, source):
        neighbor_ids = bank["spatial_index"][target_ids]
        neighbor_ids = torch.cat((neighbor_ids, target_ids[:, None]), dim=1)
        valid = (neighbor_ids >= 0).to(self.output_scale.device)
        local = self._positions(source_ids, neighbor_ids).to(self.output_scale.device)
        center_local = self._positions(source_ids, target_ids).to(self.output_scale.device)
        xyz = self._take(bank["xyz"], source_ids)
        center_xyz = xyz[center_local]
        relative = xyz[local] - center_xyz[:, None]
        return layer(source[center_local], source[local], relative, valid)

    def forward(self, bank, root_ids, *, context=None):
        condition=self._context(bank,context)
        roots = torch.as_tensor(root_ids, dtype=torch.long, device="cpu").flatten()
        n = len(bank["node"])
        if roots.numel() and ((roots < 0).any() or (roots >= n).any()):
            raise IndexError("root_ids outside bank row range")
        views = bank["view_node"].shape[0]
        if not roots.numel():
            empty = self.output_scale.new_empty
            return {"per_view": empty((0, views, 3, 2)), "values": empty((0, 3, 2)),
                    "common": empty((0, 2)), "advantage": empty((0, 3, 2))}
        projection_ids = bank["view_index"][:, roots]
        seeds = torch.unique(torch.cat((roots, projection_ids[projection_ids >= 0])), sorted=True)
        one_hop = self._include_neighbors(seeds, bank["spatial_index"])
        closure = self._include_neighbors(one_hop, bank["spatial_index"])
        nodes = self._take(bank["node"], closure)
        candidates = self._take(bank["candidate"], closure)
        state = (self.node_encoder(self._normalize(nodes, "node")) +
                 self._candidate_tokens(candidates).mean(1))
        if condition is not None:state=state+self._context_nodes(condition,closure)
        state = self._spatial_step(self.spatial_layers[0], bank, one_hop, closure, state)
        state = self._spatial_step(self.spatial_layers[1], bank, seeds, one_hop, state)

        # Each projection key includes its own complete two-hop spatial state.
        device = self.output_scale.device
        root_local = self._positions(seeds, roots).to(device)
        neighbors = torch.cat((projection_ids, roots[None, :, None].expand(views, -1, -1)), -1)
        local = self._positions(seeds, neighbors).to(device)
        valid = (neighbors >= 0).to(device)
        view_features = bank["view_node"][:, seeds].to(device=device, dtype=state.dtype)
        view_geometry = (bank if condition is None else condition)["view_geom"][:, seeds].to(device=device, dtype=state.dtype)
        projected_state = state[None] + self.view_encoder(torch.cat((
            self._normalize(view_features, "view"), view_geometry), -1))
        if condition is not None:projected_state=projected_state+self._context_views(condition,seeds)
        view_ids = torch.arange(views, device=device)[:, None, None]
        neighbor_state = projected_state[view_ids, local]
        center_state = projected_state[:, root_local]
        xyz = self._take(bank["xyz"], seeds)
        relative_xyz = xyz[local] - xyz[root_local][None, :, None]
        relative_projection = view_geometry[view_ids, local] - view_geometry[:, root_local, None]
        context = self.projection_layer(center_state, neighbor_state,
            torch.cat((relative_xyz, relative_projection), -1), valid)

        root_candidates = self._candidate_tokens(self._take(bank["candidate"], roots))
        common_per_view, advantage_per_view = self._decode_candidates(context, root_candidates[None])
        per_view = (common_per_view[:, :, None] + advantage_per_view).transpose(0, 1)
        weights = bank["view_weights"].to(device=device, dtype=state.dtype)
        weights = weights / weights.sum()
        common = (common_per_view * weights[:, None, None]).sum(0)
        advantage = (advantage_per_view * weights[:, None, None, None]).sum(0)
        return {"per_view": per_view, "values": common[:, None] + advantage,
                "common": common, "advantage": advantage}

    @torch.no_grad()
    def predict_all(self, bank, batch_size, *, context=None, _shared=None):
        """Evaluate each scene layer once, with CPU caches and chunked device gathers."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.contextual and context is None:
            shared={};values=[];profiles=[]
            for index in range(len(bank['contexts'])):
                values.append(self.predict_all(bank,batch_size,context=index,_shared=shared))
                profiles.append(dict(self.last_inference_profile))
            self.last_inference_profile=dict(context_passes=len(profiles),
                aggregation='uniform_expected_signed_distortion',contexts=profiles,
                node_encoded_rows=sum(p['node_encoded_rows'] for p in profiles),
                candidate_encoded_rows=sum(p['candidate_encoded_rows'] for p in profiles))
            return torch.stack(values).mean(0)
        condition=self._context(bank,context)
        was_training = self.training
        self.eval()
        n = len(bank["node"])
        dtype = self.output_scale.dtype
        device = self.output_scale.device
        width = self.tier_embedding.shape[-1]
        views = bank["view_node"].shape[0]
        self.last_inference_profile = {"node_encoded_rows": 0, "candidate_encoded_rows": 0,
                                       "spatial_rows": [0, 0], "view_encoded_rows": 0,
                                       "projection_query_rows": 0}
        try:
            if not n:
                return torch.empty(0, 3, 2, dtype=dtype, device="cpu")
            batches = lambda: (torch.arange(start, min(start + batch_size, n))
                               for start in range(0, n, batch_size))
            state = torch.empty(n, width, dtype=dtype, device="cpu")
            if _shared is not None and 'base' in _shared:
                base,candidate_cache=_shared['base'],_shared['candidates']
            else:
                base=torch.empty_like(state)
                candidate_cache = torch.empty(n, 4, width, dtype=dtype, device="cpu")
                for ids in batches():
                    candidates = self._candidate_tokens(self._take(bank["candidate"], ids))
                    nodes = self.node_encoder(self._normalize(self._take(bank["node"], ids), "node"))
                    candidate_cache[ids] = candidates.cpu()
                    base[ids] = (nodes + candidates.mean(1)).cpu()
                    self.last_inference_profile["node_encoded_rows"] += len(ids)
                    self.last_inference_profile["candidate_encoded_rows"] += len(ids)
                if _shared is not None:_shared.update(base=base,candidates=candidate_cache)
            if condition is None:state=base
            else:
                for ids in batches():state[ids]=base[ids]+self._context_nodes(condition,ids).cpu()
            for layer_index, layer in enumerate(self.spatial_layers):
                next_state = torch.empty_like(state)
                for ids in batches():
                    sources = self._include_neighbors(ids, bank["spatial_index"])
                    gathered = state[sources].to(device)
                    next_state[ids] = self._spatial_step(layer, bank, ids, sources, gathered).cpu()
                    self.last_inference_profile["spatial_rows"][layer_index] += len(ids)
                state = next_state
            del next_state, gathered

            common = torch.zeros(n, 2, dtype=dtype, device="cpu")
            advantage = torch.zeros(n, 3, 2, dtype=dtype, device="cpu")
            weights = bank["view_weights"].to(dtype=dtype, device="cpu")
            weights = weights / weights.sum()
            projected = torch.empty_like(state)
            for view in range(views):
                # Every source row receives its view encoding exactly once.
                for ids in batches():
                    features = self._take(bank["view_node"][view], ids)
                    geometry = self._take((bank if condition is None else condition)["view_geom"][view], ids)
                    encoded = self.view_encoder(torch.cat((self._normalize(features, "view"), geometry), -1))
                    if condition is not None:encoded=encoded+self._context_views(condition,ids,view)
                    projected[ids] = (state[ids].to(device) + encoded).cpu()
                    self.last_inference_profile["view_encoded_rows"] += len(ids)
                for ids in batches():
                    neighbor_ids = torch.cat((bank["view_index"][view, ids], ids[:, None]), -1)
                    sources = torch.unique(neighbor_ids[neighbor_ids >= 0], sorted=True)
                    local = self._positions(sources, neighbor_ids).to(device)
                    root_local = self._positions(sources, ids).to(device)
                    valid = (neighbor_ids >= 0).to(device)
                    source_state = projected[sources].to(device)
                    xyz = self._take(bank["xyz"], sources)
                    geometry = self._take((bank if condition is None else condition)["view_geom"][view], sources)
                    relative = torch.cat((xyz[local] - xyz[root_local, None],
                                          geometry[local] - geometry[root_local, None]), -1)
                    context = self.projection_layer(source_state[root_local], source_state[local],
                                                    relative, valid)
                    per_common, per_advantage = self._decode_candidates(context, candidate_cache[ids].to(device))
                    common[ids] += per_common.cpu() * weights[view]
                    advantage[ids] += per_advantage.cpu() * weights[view]
                    self.last_inference_profile["projection_query_rows"] += len(ids)
            return common[:, None] + advantage
        finally:
            self.train(was_training)
