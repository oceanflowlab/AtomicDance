import os
import pickle
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

from eval.eval_bas import alignment_score, calculate_bas
from eval.evaluate import evaluate, resolve_features
from eval.extract_aist_features import load_keypoints
from eval.metrics import normalize_separately


class EvaluationTests(unittest.TestCase):
    def make_feature_root(self, root, offset=0.0):
        for directory in (
            "kinetic_features",
            "manual_features",
            "music_features",
            "dance_features",
        ):
            os.makedirs(os.path.join(root, directory))
        for index in range(4):
            name = "sample{}".format(index)
            kinetic = np.array([index, index ** 2, index + 1], dtype=np.float32) + offset
            manual = np.array([index % 2, index / 2], dtype=np.float32) + offset
            music = np.zeros(30, dtype=bool)
            dance = np.zeros(30, dtype=bool)
            music[[5, 15, 25]] = True
            dance[[5, 15, 25]] = True
            np.save(os.path.join(root, "kinetic_features", name + ".npy"), kinetic)
            np.save(os.path.join(root, "manual_features", name + ".npy"), manual)
            np.save(os.path.join(root, "music_features", name + ".npy"), music)
            np.save(os.path.join(root, "dance_features", name + ".npy"), dance)

    def test_bas_edge_cases(self):
        self.assertEqual(alignment_score(np.zeros(5), np.zeros(5)), 0.0)
        beats = np.array([False, True, False, True])
        self.assertEqual(alignment_score(beats, beats), 0.5)

    def test_unified_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            prediction = os.path.join(directory, "prediction")
            ground_truth = os.path.join(directory, "ground_truth")
            self.make_feature_root(prediction)
            self.make_feature_root(ground_truth)

            metrics = evaluate(prediction, ground_truth)
            self.assertAlmostEqual(metrics["fid_k"], 0.0, places=5)
            self.assertAlmostEqual(metrics["fid_m"], 0.0, places=5)
            self.assertAlmostEqual(metrics["BAS_pred"], 0.5)
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
            self.assertAlmostEqual(calculate_bas(prediction), 0.5)

    def test_starter_normalizes_prediction_and_gt_separately(self):
        features = np.array(
            [[0.0, 1.0], [1.0, 3.0], [3.0, 7.0], [6.0, 13.0]],
            dtype=np.float64,
        )
        transformed = features * np.array([4.0, 2.0]) + np.array([50.0, -20.0])
        np.testing.assert_allclose(
            normalize_separately(features),
            normalize_separately(transformed),
            atol=1e-12,
        )

        with tempfile.TemporaryDirectory() as directory:
            prediction = os.path.join(directory, "prediction")
            ground_truth = os.path.join(directory, "ground_truth")
            self.make_feature_root(prediction, offset=100.0)
            self.make_feature_root(ground_truth)

            metrics = evaluate(prediction, ground_truth)
            self.assertAlmostEqual(metrics["fid_k"], 0.0, places=5)
            self.assertAlmostEqual(metrics["fid_m"], 0.0, places=5)

    def test_generated_motion_feature_extraction_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            motions = os.path.join(directory, "motions")
            audio = os.path.join(directory, "audio")
            cache = os.path.join(directory, "cache")
            os.makedirs(motions)
            os.makedirs(audio)

            rng = np.random.RandomState(7)
            full_pose = rng.normal(size=(40, 24, 3)).astype(np.float32)
            motion_name = "sample_0_gBR_sBM_cAll_d04_mBR0_ch01"
            motion_path = os.path.join(motions, motion_name + ".pkl")
            with open(motion_path, "wb") as handle:
                pickle.dump({"full_pose": full_pose}, handle)
            with open(os.path.join(motions, "unrelated.pkl"), "wb") as handle:
                pickle.dump({"full_pose": full_pose}, handle)
            samples = np.arange(44100, dtype=np.float32)
            waveform = 0.1 * np.sin(2 * np.pi * 220 * samples / 44100)
            wavfile.write(
                os.path.join(audio, "gBR_sBM_cAll_d04_mBR0_ch01.wav"),
                44100,
                waveform,
            )

            root = resolve_features(
                "prediction",
                None,
                motions,
                audio,
                cache,
                workers=1,
                include_names=[motion_name],
            )
            for feature_dir in (
                "kinetic_features",
                "manual_features",
                "dance_features",
                "music_features",
            ):
                self.assertTrue((root / feature_dir / (motion_name + ".npy")).is_file())
            self.assertEqual(
                load_keypoints(motion_path)[0, 0].tolist(),
                [full_pose[0, 0, 0], full_pose[0, 0, 2], -full_pose[0, 0, 1]],
            )
            self.assertEqual(
                resolve_features(
                    "prediction",
                    None,
                    motions,
                    audio,
                    cache,
                    workers=1,
                    include_names=[motion_name],
                ),
                root,
            )
