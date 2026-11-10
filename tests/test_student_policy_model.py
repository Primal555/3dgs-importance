from dataclasses import replace
import pytest
import torch
from gaussian_jscc.student_policy_graph import build_graph
from gaussian_jscc.student_policy_model import StudentConfig, StudentPolicy, hard_tiers


def inputs(n=37):
    torch.manual_seed(82)
    a, y, v = torch.randn(n, 12), torch.randn(n, 48), torch.randn(n, 43)
    xyz = torch.rand(n, 3); xyz[:2] = xyz[0]
    return a, y, v, build_graph(xyz, fine_points=4, coarse_points=3, neighbors=4)


def test_chunk_forward_and_all_branch_gradients_match():
    torch.set_num_threads(1)
    a, y, v, graph = inputs()
    cfg = StudentConfig(12, 48, 43, width=16, coarse_width=24, blocks=2, chunk_size=7)
    first = StudentPolicy(cfg); first.fit_normalizers(a, y, v)
    other = StudentPolicy(replace(cfg, chunk_size=100)); other.load_state_dict(first.state_dict())
    controls = torch.tensor([1., .5, -6., 4.])
    p = first(a, y, v, graph, controls); q = other(a, y, v, graph, controls)
    for left, right in zip(p, q): torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
    sum(x.square().mean() for x in p).backward()
    sum(x.square().mean() for x in q).backward()
    for (name, left), (_, right) in zip(first.named_parameters(), other.named_parameters()):
        assert left.grad is not None and torch.isfinite(left.grad).all(), name
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-6)
    assert first.global_encoder[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('n', [1, 2, 17])
def test_coincident_and_permuted_points_have_equivariant_actions(n):
    torch.set_num_threads(1)
    a, y, v, _ = inputs(n)
    xyz = torch.zeros(n, 3)
    graph = build_graph(xyz, fine_points=4, coarse_points=3, neighbors=4)
    net = StudentPolicy(StudentConfig(12, 48, 43, width=16, coarse_width=24, chunk_size=8))
    net.fit_normalizers(a, y, v); net.eval()
    permutation = torch.randperm(n); control = torch.tensor([1., .5, -6., 4.])
    with torch.no_grad():
        expected = net(a, y, v, graph, control)
        got = net(a[permutation], y[permutation], v[permutation],
                  build_graph(xyz[permutation], fine_points=4, coarse_points=3, neighbors=4), control)
    for old, new in zip(expected, got): torch.testing.assert_close(old[permutation], new)
    assert hard_tiers(*expected).shape == (n,)


def test_spatial_geometry_is_metric_and_partition_maps_original_rows():
    xyz = torch.tensor([[0., 0, 0], [1., 0, 0], [100., 0, 0], [101., 0, 0], [50., 1., 0]])
    graph = build_graph(xyz, fine_points=2, coarse_points=2, neighbors=3)
    assert len(graph['cluster1']) == len(xyz)
    assert len(graph['cluster2']) == len(graph['coords1'])
    for name in ('neighbors1', 'neighbors2'): assert graph[name].dtype == torch.long
    assert torch.isfinite(graph['coords1']).all()
    permutation = torch.tensor([4, 2, 0, 3, 1])
    new = build_graph(xyz[permutation], fine_points=2, coarse_points=2, neighbors=3)
    torch.testing.assert_close(new['cluster1'], graph['cluster1'][permutation])


def test_hard_gate_is_independent_of_positive_entropy():
    keep = torch.tensor([[0., .01], [.01, 0.], [0., 0.]])
    tier = torch.tensor([[0., 0., 0.], [0., 0., 100.], [0., 2., 0.]])
    assert hard_tiers(keep, tier).tolist() == [1, 0, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable locally')
def test_cuda_cpu_feature_chunks_recompute_with_live_global_gradients():
    from gaussian_jscc.student_policy_graph import graph_to
    a, y, v, graph = inputs()
    net = StudentPolicy(StudentConfig(12, 48, 43, width=16, coarse_width=24, chunk_size=7)).cuda()
    net.fit_normalizers(a, y, v)
    keep, tier = net(a, y, v, graph_to(graph, 'cuda'), torch.tensor([1., .5, -6., 4.], device='cuda'))
    (keep.square().mean()+tier.square().mean()).backward()
    assert torch.isfinite(net.global_encoder[0].weight.grad).all()
    assert net.global_encoder[0].weight.grad.abs().sum() > 0
