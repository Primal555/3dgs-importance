import copy
import io

import pytest
import torch


def make_bank(n=9, candidate_dim=6):
    generator = torch.Generator().manual_seed(41)
    spatial = torch.full((n, 3), -1, dtype=torch.long)
    spatial[:, 0] = (torch.arange(n) + 1) % n
    spatial[:, 1] = (torch.arange(n) + 2) % n
    projection = torch.full((2, n, 3), -1, dtype=torch.long)
    projection[0, :, 0] = (torch.arange(n) + 4) % n
    projection[1, :, 0] = (torch.arange(n) + 6) % n
    candidate = torch.randn(n, 4, candidate_dim, generator=generator)
    candidate[:, 0, -2:] = 0
    return {
        "node": torch.randn(n, 5, generator=generator),
        "candidate": candidate,
        "xyz": torch.randn(n, 3, generator=generator),
        "spatial_index": spatial,
        "view_node": torch.randn(2, n, 7, generator=generator),
        "view_geom": torch.randn(2, n, 7, generator=generator),
        "view_index": projection,
        "view_weights": torch.tensor([0.25, 0.75]),
        "metadata": {},
    }


def make_model(bank, **kwargs):
    from gaussian_jscc.relation_benefit_model import RelationBenefitNetwork
    torch.manual_seed(7)
    model = RelationBenefitNetwork(5, bank["candidate"].shape[-1], 7,
                                   width=16, groups=4, **kwargs)
    model.fit_normalizers(bank, torch.arange(len(bank["node"])))
    return model


def context_bank():
    bank=make_bank(13)
    g=torch.Generator().manual_seed(13)
    bank['contexts']=[dict(name=name,node=torch.randn(13,6,generator=g),
        view_node=torch.randn(2,13,8,generator=g),view_geom=torch.randn(2,13,7,generator=g))
        for name in ('q1','q2','q3','mixed')]
    return bank


def test_context_inputs_change_values_and_all_context_inference_matches_exact_mean():
    bank=context_bank()
    model=make_model(bank,context_node_dim=6,context_view_dim=8).eval()
    per=[model(bank,torch.arange(13),context=c)['values'].detach() for c in range(4)]
    assert not torch.allclose(per[0],per[1])
    expected=torch.stack(per).mean(0)
    torch.testing.assert_close(model.predict_all(bank,3),expected,atol=4e-7,rtol=4e-6)
    assert model.last_inference_profile['context_passes']==4
    for c in range(4):
        torch.testing.assert_close(model.predict_all(bank,2,context=c),per[c],atol=4e-7,rtol=4e-6)


def test_context_normalizers_fit_all_contexts_on_training_rows_only():
    bank=context_bank();model=make_model(bank,context_node_dim=6,context_view_dim=8)
    ids=torch.tensor([1,3,5]);model.fit_normalizers(bank,ids)
    changed=copy.deepcopy(bank)
    for ctx in changed['contexts']:
        ctx['node'][[0,2,4,6,7,8,9,10,11,12]]=1e8
        ctx['view_node'][:,[0,2,4,6,7,8,9,10,11,12]]=1e8
    other=make_model(bank,context_node_dim=6,context_view_dim=8);other.fit_normalizers(changed,ids)
    for name in ('context_node_mean','context_node_std','context_view_mean','context_view_std'):
        torch.testing.assert_close(getattr(model,name),getattr(other,name),atol=0,rtol=0)


def test_output_decomposition_and_fixed_view_weights():
    bank = make_bank(candidate_dim=11)
    model = make_model(bank).eval()
    result = model(bank, torch.tensor([6, 0, 3]))
    assert result["per_view"].shape == (3, 2, 3, 2)
    assert result["values"].shape == (3, 3, 2)
    torch.testing.assert_close(result["values"],
        (result["per_view"] * bank["view_weights"][None, :, None, None]).sum(1))
    torch.testing.assert_close(result["values"].mean(1), result["common"])
    torch.testing.assert_close(result["advantage"].mean(1), torch.zeros(3, 2), atol=1e-7, rtol=0)
    torch.testing.assert_close(result["values"], result["common"][:, None] + result["advantage"])


def test_node_permutation_preserves_original_row_correspondence():
    bank = make_bank()
    model = make_model(bank).eval()
    permutation = torch.tensor([8, 3, 1, 7, 0, 5, 2, 6, 4])
    inverse = torch.argsort(permutation)
    permuted = copy.deepcopy(bank)
    for key in ("node", "candidate", "xyz"):
        permuted[key] = bank[key][permutation]
    for key in ("view_node", "view_geom"):
        permuted[key] = bank[key][:, permutation]
    for key, rows in (("spatial_index", bank["spatial_index"][permutation]),
                      ("view_index", bank["view_index"][:, permutation])):
        permuted[key] = torch.where(rows >= 0, inverse[rows.clamp_min(0)], -1)
    expected = model(bank, torch.arange(9))["values"][permutation]
    torch.testing.assert_close(model(permuted, torch.arange(9))["values"], expected,
                               atol=2e-7, rtol=2e-6)


def test_padding_and_disconnected_rows_do_not_pollute_isolated_root():
    bank = make_bank()
    bank["spatial_index"][:] = -1
    bank["view_index"][:] = -1
    model = make_model(bank).eval()
    before = model(bank, [0])["values"]
    changed = copy.deepcopy(bank)
    for key in ("node", "candidate", "xyz"):
        changed[key][1:] = 100000
    for key in ("view_node", "view_geom"):
        changed[key][:, 1:] = 100000
    after = model(changed, [0])["values"]
    assert torch.isfinite(after).all()
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    wider = copy.deepcopy(bank)
    wider["spatial_index"] = torch.full((9, 9), -1, dtype=torch.long)
    wider["view_index"] = torch.full((2, 9, 9), -1, dtype=torch.long)
    torch.testing.assert_close(model(wider, [0])["values"], before, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("source, row", [("node", 2), ("node", 6), ("candidate", 0)])
def test_two_hop_and_projection_neighbor_context_and_q0_change_root(source, row):
    bank = make_bank()
    bank["spatial_index"][:] = -1
    bank["spatial_index"][0, 0] = 1
    bank["spatial_index"][1, 0] = 2
    bank["spatial_index"][4, 0] = 5
    bank["spatial_index"][5, 0] = 6
    bank["view_index"][:] = -1
    bank["view_index"][0, 0, 0] = 4
    model = make_model(bank).eval()
    before = model(bank, [0])["values"]
    changed = copy.deepcopy(bank)
    if source == "candidate":
        changed[source][row, 0, :3] += 3
    else:
        changed[source][row] += 3
    assert (model(changed, [0])["values"] - before).abs().max() > 1e-7


def test_batched_inference_matches_full_graph_and_restores_training_mode():
    bank = make_bank(19)
    model = make_model(bank).eval()
    expected = model(bank, torch.arange(19))["values"].detach()
    model.train()
    actual = model.predict_all(bank, batch_size=3)
    assert actual.device.type == "cpu" and not actual.requires_grad
    assert model.training
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-6)
    torch.testing.assert_close(model.predict_all(bank, batch_size=1), expected,
                               atol=3e-7, rtol=3e-6)


def test_small_root_batch_encodes_only_its_graph_closure():
    bank = make_bank(100)
    bank["spatial_index"][:] = -1
    bank["view_index"][:] = -1
    bank["spatial_index"][0, 0] = 1
    bank["spatial_index"][1, 0] = 2
    bank["view_index"][0, 0, 0] = 4
    bank["spatial_index"][4, 0] = 5
    bank["spatial_index"][5, 0] = 6
    model = make_model(bank).eval()
    encoded_rows = []
    handle = model.node_encoder.register_forward_pre_hook(
        lambda module, inputs: encoded_rows.append(len(inputs[0])))
    try:
        assert torch.isfinite(model(bank, [0])["values"]).all()
    finally:
        handle.remove()
    assert encoded_rows == [6]


def test_low_variance_large_features_are_compressed_with_relative_scale_floor():
    bank = make_bank()
    for key in ("node", "candidate", "view_node"):
        bank[key].fill_(1000.)
    bank["node"][0].fill_(1e9)
    bank["candidate"][0].fill_(1e9)
    bank["view_node"][:, 0].fill_(1e9)
    model = make_model(bank).eval()
    model.fit_normalizers(bank, torch.arange(1, 9))
    assert model.node_std.min() >= 1
    assert model.candidate_std.min() >= 1
    assert model.view_std.min() >= 1
    seen = {"node": [], "candidate": [], "view": []}
    handles = []
    for name, encoder in (("node", model.node_encoder),
                          ("candidate", model.candidate_encoder),
                          ("view", model.view_encoder)):
        handles.append(encoder.register_forward_pre_hook(
            lambda module, inputs, name=name: seen[name].append(inputs[0].detach())))
    try:
        values = model(bank, [0])["values"]
    finally:
        for handle in handles:
            handle.remove()
    assert torch.isfinite(values).all()
    for name, inputs in seen.items():
        # asinh((1e9 - 1000) / 1) is about 21.42; projection geometry stays small.
        assert inputs and max(value.abs().max() for value in inputs) < 22


def test_predict_all_encodes_each_scene_layer_once_across_root_batches():
    bank = make_bank(23)
    model = make_model(bank).eval()
    expected = model(bank, torch.arange(23))["values"].detach()
    counts = {"node": 0, "candidate": 0, "spatial1": 0,
              "spatial2": 0, "view": 0, "projection": 0}
    handles = []
    modules = (("node", model.node_encoder), ("candidate", model.candidate_encoder),
               ("spatial1", model.spatial_layers[0]), ("spatial2", model.spatial_layers[1]),
               ("view", model.view_encoder), ("projection", model.projection_layer))
    for name, module in modules:
        def count_rows(module, inputs, name=name):
            value = inputs[0]
            # Projection forward may arrange queries as [V,B,H] or [B,H].
            counts[name] += value.numel() // value.shape[-1] if name == "projection" else len(value)
        handles.append(module.register_forward_pre_hook(count_rows))
    try:
        actual = model.predict_all(bank, 3)
    finally:
        for handle in handles:
            handle.remove()
    assert counts == {"node": 23, "candidate": 23, "spatial1": 23,
                      "spatial2": 23, "view": 46, "projection": 46}
    assert model.last_inference_profile["node_encoded_rows"] == 23
    assert model.last_inference_profile["spatial_rows"] == [23, 23]
    assert model.last_inference_profile["projection_query_rows"] == 46
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-6)


def test_normalizers_use_only_training_rows_and_save_load_preserves_units():
    bank = make_bank()
    model = make_model(bank).eval()
    train_ids = torch.tensor([0, 2, 4])
    model.fit_normalizers(bank, train_ids)
    changed = copy.deepcopy(bank)
    for key in ("node", "candidate"):
        changed[key][[1, 3, 5, 6, 7, 8]] = 1e6
    changed["view_node"][:, [1, 3, 5, 6, 7, 8]] = 1e6
    other = make_model(bank).eval()
    other.fit_normalizers(changed, train_ids)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[key], atol=0, rtol=0)
    baseline = model(bank, [0, 4])["values"]
    model.output_scale.copy_(torch.tensor([2., 5.]))
    expected = baseline * torch.tensor([2., 5.])
    torch.testing.assert_close(model(bank, [0, 4])["values"], expected)
    expected = model(bank, [0, 4])["values"]
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    other.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(other(bank, [0, 4])["values"], expected, atol=0, rtol=0)


def test_toy_signed_targets_learn_common_and_tier_differences():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        bank = make_bank(4)
        model = make_model(bank)
        roots = torch.arange(4)
        common = torch.stack((bank["node"][:, 0] * .3 - .2,
                              bank["node"][:, 1] * .2 + .1), -1)
        advantage = torch.tensor([[.25, -.1], [-.3, .3], [.05, -.2]])
        truth = common[:, None] + advantage
        optimizer = torch.optim.Adam(model.parameters(), lr=.006)
        initial = (model(bank, roots)["values"] - truth).square().mean().item()
        for _ in range(140):
            result = model(bank, roots)
            loss = ((result["common"] - common).square().mean() +
                    (result["advantage"] - advantage).square().mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        result = model(bank, roots)
        assert (result["values"] - truth).square().mean() < initial * .015
        torch.testing.assert_close(result["common"], common, atol=.02, rtol=0)
        torch.testing.assert_close(result["advantage"], advantage.expand(4, -1, -1), atol=.02, rtol=0)
        assert result["values"].min() < 0 < result["values"].max()
    finally:
        torch.set_num_threads(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_bank_can_train_and_predict_on_cuda():
    bank = make_bank()
    model = make_model(bank).cuda()
    result = model(bank, torch.tensor([0, 4]))
    assert result["values"].is_cuda
    result["values"].square().sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert model.predict_all(bank, 3).device.type == "cpu"
