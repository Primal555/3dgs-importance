import json
import tempfile
import unittest
from pathlib import Path

from comparison_baselines.packing import pack_ply, unpack_ply
from comparison_baselines.run import parser, build_plan, summarize_trials, validate_resume


class PackingTests(unittest.TestCase):
    def test_stream_round_trip_and_existing_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.ply'
            content = b'ply\n' + bytes(range(256)) * 10003
            source.write_bytes(content)
            info = pack_ply(source, root / 'packed.bin')
            self.assertEqual(info['input_bytes'], len(content))
            self.assertEqual(info['packed_bytes'], (root/'packed.bin').stat().st_size)
            unpack_ply(root/'packed.bin', root/'received.ply')
            self.assertEqual((root/'received.ply').read_bytes(), content)
            with self.assertRaises(FileExistsError):
                unpack_ply(root/'packed.bin', root/'received.ply')

    def test_corrupt_stream_never_creates_final_ply(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'input').write_bytes(bytes(range(256))*100)
            pack_ply(root/'input', root/'archive')
            damaged = bytearray((root/'archive').read_bytes())
            damaged[-1] ^= 16
            (root/'archive').write_bytes(damaged)
            with self.assertRaises((ValueError, RuntimeError)):
                unpack_ply(root/'archive', root/'result.ply')
            self.assertFalse((root/'result.ply').exists())

    def test_trailing_data_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'input').write_bytes(b'point cloud')
            pack_ply(root/'input', root/'archive')
            with (root/'archive').open('ab') as stream:
                stream.write(b'not part of the stream')
            with self.assertRaises(ValueError):
                unpack_ply(root/'archive', root/'result.ply')


class PipelineTests(unittest.TestCase):
    def test_plan_uses_exact_codec_input_and_camera_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'photos'; source.mkdir()
            ply = root/'pruned.ply'; ply.write_bytes(b'ply\nend_header\n')
            run = root/'codec'; run.mkdir()
            (run/'training.json').write_text(json.dumps({
                'selected_scenes': [{'name':'truck', 'ply':str(ply), 'source':str(source)}],
                'settings': {'resolution':4, 'images':'images_2', 'white_background':True, 'test_views':24}
            }))
            args = parser().parse_args(['--run',str(run),'--out',str(root/'new')])
            plan = build_plan(args)
            self.assertEqual(plan['input']['ply'], str(ply.resolve()))
            self.assertEqual(plan['evaluation']['resolution'], 4)
            self.assertEqual(plan['evaluation']['images'], 'images_2')
            self.assertTrue(plan['evaluation']['white_background'])
            self.assertEqual(plan['evaluation']['views'], 24)
            self.assertEqual(plan['link']['k'],4096)
            self.assertEqual(plan['link']['n'],6144)
            self.assertEqual(plan['link']['bits_per_symbol'],4)

    def test_explicit_paths_and_missing_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'photos').mkdir()
            args=parser().parse_args(['--ply',str(root/'missing'),'--source',str(root/'photos'),
                                      '--out',str(root/'new')])
            with self.assertRaises(FileNotFoundError):build_plan(args)

    def test_protocol_and_top_level_records_preserve_camera_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'photos').mkdir();(root/'source.ply').write_bytes(b'ply\n')
            run=root/'codec';run.mkdir()
            options=dict(resolution=1,images='images_4',white_background=True,lpips=False)
            for key in ['protocol',None]:
                record={'selected_scenes':[dict(name='truck',ply=str(root/'source.ply'),source=str(root/'photos'))]}
                if key:record[key]=options
                else:record.update(options)
                (run/'training.json').write_text(json.dumps(record))
                args=parser().parse_args(['--run',str(run),'--out',str(root/'out')])
                with self.subTest(key=key):
                    actual=build_plan(args)['evaluation']
                    for name,value in options.items():self.assertEqual(actual[name],value)

    def test_distinct_snrs_have_distinct_storage_names(self):
        from comparison_baselines.run import snr_label
        self.assertNotEqual(snr_label(10.),snr_label(10.000001))

    def test_blender_photo_content_is_part_of_resume_identity(self):
        from comparison_baselines.run import _dataset_identity
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'test').mkdir()
            photo=root/'test/r_0.png';photo.write_bytes(b'original RGB')
            (root/'transforms_test.json').write_text(json.dumps({'frames':[{'file_path':'test/r_0'}]}))
            before=_dataset_identity(root,'images')
            photo.write_bytes(b'changed  RGB')
            self.assertNotEqual(before,_dataset_identity(root,'images'))

    def test_run_preserves_saved_camera_names_and_list_heldout_scene(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'room').mkdir();(root/'scene.ply').write_bytes(b'ply\n')
            run=root/'codec';run.mkdir()
            record=dict(selected_scenes=[],heldout_scene=[dict(name='room',ply=str(root/'scene.ply'),source=str(root/'room'))],
                        scene_records={'room':{'test_view_names':['v1','v2']}},settings={'test_views':2})
            (run/'training.json').write_text(json.dumps(record))
            args=parser().parse_args(['--run',str(run),'--scene','room','--out',str(root/'out')])
            plan=build_plan(args)
            self.assertEqual(plan['evaluation']['camera_names'],['v1','v2'])

    def test_failed_transmission_is_not_a_zero_quality_sample(self):
        rows=[{'method':'maskgaussian','snr_db':10.,'trial':0,'status':'success',
               'total_channel_uses':100,'metrics':{'photo_psnr':30.,'source_psnr':40.}},
              {'method':'maskgaussian','snr_db':10.,'trial':1,'status':'failed',
               'total_channel_uses':100,'metrics':None}]
        summary=summarize_trials(rows)[0]
        self.assertEqual(summary['trials'],2)
        self.assertEqual(summary['success_rate'],.5)
        self.assertEqual(summary['metrics']['photo_psnr'],30.)
        self.assertEqual(summary['total_channel_uses'],100)
        self.assertEqual(summary['quality_conditioning'],'successful_file_delivery')
        rows[0]['status']='failed';rows[0]['metrics']=None
        self.assertIsNone(summarize_trials(rows)[0]['metrics'])

    def test_resume_rejects_changed_input_or_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            plan={'version':'test','link':{'k':4096,'n':6144},'input':{'sha256':'abc'}}
            (root/'plan.json').write_text(json.dumps(plan))
            validate_resume(root,plan)
            with self.assertRaises(ValueError):
                validate_resume(root,dict(plan,input={'sha256':'different'}))


if __name__=='__main__':unittest.main()
