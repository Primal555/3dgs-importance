import unittest
import torch


class PayloadEvaluationTests(unittest.TestCase):
    def test_metrics_distinguish_coordinate_mse_distance_rmse_and_common_drift(self):
        from gaussian_jscc.xyz_payload_evaluation import position_statistics
        target=torch.zeros(4,3); pred=torch.tensor([[1.,0,0],[1.,0,0],[-1.,0,0],[-1.,0,0]])
        stats=position_statistics(target,pred,[2,2])
        self.assertAlmostEqual(stats['world_coordinate_mse'],1/3)
        self.assertEqual(stats['world_distance_rmse'],1.)
        self.assertEqual(stats['world_translation_removed_rmse'],1.)
        self.assertEqual(stats['world_block_translation_removed_rmse'],0.)
        self.assertEqual(stats['world_mean_error_xyz'],[0.,0.,0.])
        with self.assertRaises(ValueError): position_statistics(target,pred,[1,1])

    def test_saved_metrics_generate_position_and_phase_charts(self):
        import json
        import tempfile
        from pathlib import Path
        from gaussian_jscc.xyz_payload_plots import plot_run
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); (root/'metrics').mkdir()
            rows=[dict(step=s,phase='semantic_render',path='clean',layout='clean',actual_snr=None,
                       source_psnr=20.,source_ssim=.8,photo_psnr=19.,photo_ssim=.7,
                       world_distance_rmse=.001,world_distance_p95=.002,world_bias_norm=.0001,
                       payload_xyz_mse=0.,payload_attr_mse=0.) for s in (0,500)]
            (root/'metrics/validation.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            paths=plot_run(root)
            self.assertIn(root/'charts/position_precision.png',paths)
            self.assertTrue(all(p.stat().st_size>1000 for p in paths))


if __name__=='__main__': unittest.main()
