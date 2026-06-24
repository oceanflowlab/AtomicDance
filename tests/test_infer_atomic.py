import os
import tempfile
import unittest

import numpy as np
import torch

from infer_atomic import (
    GroundTruthPlanStore,
    IndexedAtomicMotionLibrary,
    decode_motion,
    infer_completion,
    infer_plan,
)


class PlannerStub:
    def sample(self, music, padding_mask=None, temperature=1.0, deterministic=False):
        return torch.ones(music.shape[:2], dtype=torch.long, device=music.device)


class CompletionStub:
    def sample(self, music, draft, noise_mask, guidance_weight=None):
        return draft


class AtomicInferenceTests(unittest.TestCase):
    def test_ground_truth_plan_store(self):
        with tempfile.TemporaryDirectory() as directory:
            for split, name in (("train", "song_slice0"), ("test", "other_slice0")):
                root = os.path.join(directory, split)
                os.makedirs(root)
                np.save(
                    os.path.join(root, "labels.npy"),
                    np.array([[1, 1, 0, 2]], dtype=np.uint8),
                )
                with open(os.path.join(root, "names.json"), "w") as handle:
                    import json

                    json.dump([name], handle)
            store = GroundTruthPlanStore(directory)
            self.assertTrue(store.has_sequence("song"))
            self.assertEqual(store.get("song", 3).tolist(), [1, 1, 0])

    def test_indexed_library_and_windowed_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            train = os.path.join(directory, "train")
            os.makedirs(train)
            motion = np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)
            labels = np.array([[1, 1, 1, 0, 2, 2], [2, 2, 2, 2, 1, 1]])
            np.save(os.path.join(train, "motion.npy"), motion)
            np.save(os.path.join(train, "labels.npy"), labels)

            library = IndexedAtomicMotionLibrary(directory)
            draft, mask = library.build_draft(torch.tensor([1, 1, 0, 2, 2]), 3)
            self.assertEqual(draft.shape, (5, 3))
            self.assertEqual(mask[:, 0].tolist(), [1.0, 1.0, 0.0, 1.0, 1.0])

            music = torch.zeros(8, 2)
            plan = infer_plan(
                PlannerStub(), music, 4, torch.device("cpu"), available_labels={1, 2}
            )
            self.assertEqual(plan.tolist(), [1] * 8)
            expected = torch.arange(24, dtype=torch.float32).reshape(8, 3)
            generated = infer_completion(
                CompletionStub(),
                music,
                expected,
                torch.ones(8, 1),
                4,
                2,
                torch.device("cpu"),
            )
            self.assertTrue(torch.allclose(generated, expected))

    def test_decode_motion_produces_full_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            raw = torch.cat((torch.zeros(7), identity_6d.repeat(24)))
            normalizer = os.path.join(directory, "normalizer.pt")
            torch.save({"data_min": raw, "data_max": raw}, normalizer)
            decoded = decode_motion(torch.zeros(5, 151), normalizer)
            self.assertEqual(decoded["smpl_poses"].shape, (5, 72))
            self.assertEqual(decoded["smpl_trans"].shape, (5, 3))
            self.assertEqual(decoded["full_pose"].shape, (5, 24, 3))
            self.assertTrue(np.isfinite(decoded["full_pose"]).all())


if __name__ == "__main__":
    unittest.main()
