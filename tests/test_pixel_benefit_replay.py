import tempfile
import unittest
from pathlib import Path

import torch

from gaussian_jscc.pixel_benefit_replay import PixelReplay, sample_sources


def record(source):
    return {"source": source, "step": 1, "validation": False,
            "views": [{"features": torch.arange(10), "target": torch.ones(3)}]}


class ReplayTests(unittest.TestCase):
    def test_fixed_validation_and_complete_groups(self):
        with tempfile.TemporaryDirectory() as root:
            replay = PixelReplay(root, 5, 2, 2)
            replay.add(record("valid"), True)
            replay.freeze_validation()
            with self.assertRaises(ValueError):
                replay.add(record("later"), True)
            replay.add(record("train"), False)
            result = replay.sample(4)
            self.assertEqual([r["source"] for r in result], ["train"] * 4)
            self.assertTrue(torch.equal(result[0]["views"][0]["features"], torch.arange(10)))
            self.assertEqual([r["source"] for r in replay.validation_records()], ["valid"])

    def test_restore_rng_priorities_and_old_checkpoint_shards(self):
        with tempfile.TemporaryDirectory() as root:
            replay = PixelReplay(root, 7, 2, 1)
            first = replay.add(record("a"), False)
            replay.add(record("b"), False)
            replay.update_priority(first, 10)
            state = replay.state_dict()
            expected = [(r["source"], r["replay_probability"]) for r in replay.sample(20)]
            replay.add(record("c"), False)
            replay.load_state_dict(state)
            replay.add(record("d"), False)
            restored = PixelReplay(root, 99, 2, 1)
            restored.load_state_dict(state)
            actual = [(r["source"], r["replay_probability"]) for r in restored.sample(20)]
            self.assertEqual(actual, expected)
            self.assertEqual({r["source"] for r in restored.sample(100)}, {"a", "b"})

    def test_missing_shard_is_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            replay = PixelReplay(root, 0, 2, 1)
            replay.add(record("a"), False)
            state = replay.state_dict()
            next(Path(root).rglob("*.pt")).unlink()
            with self.assertRaises(FileNotFoundError):
                replay.load_state_dict(state)

    def test_training_records_follow_fifo_and_validation_capacity(self):
        with tempfile.TemporaryDirectory() as root:
            replay = PixelReplay(root, 0, 2, 1)
            replay.add(record("held"), True)
            self.assertIsNone(replay.add(record("extra"), True))
            for name in ("a", "b", "c"):
                replay.add(record(name), False)
            self.assertEqual([r["source"] for r in replay.training_records()], ["b", "c"])
            self.assertEqual([r["source"] for r in replay.validation_records()], ["held"])

    def test_truncated_shard_is_rejected_on_restore(self):
        with tempfile.TemporaryDirectory() as root:
            replay = PixelReplay(root, 0, 2, 1)
            replay.add(record("a"), False)
            state = replay.state_dict()
            path = next(Path(root).rglob("*.pt"))
            with path.open("r+b") as shard:
                shard.truncate(1)
            with self.assertRaises(ValueError):
                replay.load_state_dict(state)

    def test_gc_preserves_checkpoint_and_validation_shards_and_removes_only_owned_orphans(self):
        with tempfile.TemporaryDirectory() as root:
            replay = PixelReplay(root, 0, 1, 1)
            replay.add(record("held"), True)
            replay.add(record("old"), False)
            checkpoint = replay.manifest()
            saved = replay.state_dict()
            replay.add(record("evicted"), False)
            orphan = Path(root) / replay.state_dict()['training'][0]['path']
            orphan_size = orphan.stat().st_size
            replay.add(record("live"), False)
            foreign = Path(root) / 'training' / ('a' * 32 + '.pt')
            foreign.write_bytes(b'user data')
            user_in_store = Path(root) / 'training' / replay.store_id / ('b' * 32 + '.pt')
            user_in_store.write_bytes(b'user data in store')
            with self.assertRaises(ValueError):
                replay.garbage_collect([checkpoint, {'version': 1}])
            self.assertEqual(len(list((Path(root) / 'training' / replay.store_id).glob('*.pt'))), 4)
            self.assertEqual(replay.garbage_collect([checkpoint]), 1)
            self.assertGreater(orphan_size, 0)
            self.assertFalse(orphan.exists())
            self.assertEqual(foreign.read_bytes(), b'user data')
            self.assertEqual(user_in_store.read_bytes(), b'user data in store')
            self.assertEqual([r['source'] for r in replay.validation_records()], ['held'])
            self.assertEqual([r['source'] for r in replay.training_records()], ['live'])
            replay.load_state_dict(saved)
            self.assertEqual([r['source'] for r in replay.training_records()], ['old'])


    def test_source_mixture_probabilities_and_importance(self):
        draws = sample_sources(["a", "b"], [3, 0], [0, 2], 100,
                               torch.Generator().manual_seed(4))
        self.assertEqual({d["source"] for d in draws}, {"a", "b"})
        for draw in draws:
            self.assertAlmostEqual(draw["proposal_probability"], 0.5)
            self.assertAlmostEqual(draw["importance_weight"], 1.0)
        draws = sample_sources(["a", "b"], [1, 0], [1, 0], 100,
                               torch.Generator().manual_seed(4))
        for draw in draws:
            p = 0.75 if draw["source"] == "a" else 0.25
            self.assertAlmostEqual(draw["proposal_probability"], p)
            self.assertAlmostEqual(draw["importance_weight"], 1 / (2 * p))


if __name__ == "__main__":
    unittest.main()
