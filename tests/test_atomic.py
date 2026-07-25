import json
import os
import tempfile
import unittest

import numpy as np
import torch

from dataset.atomic import (
    AtomicMotionLibrary,
    labels_to_segments,
    majority_vote,
    plan_boundaries,
    refine_plan,
)
from dataset.atomic_dataset import AtomicSequenceDataset, collate_atomic_sequences
from model.atomic_completion import AtomicCompletionDecoder, AtomicCompletionDiffusion
from model.atomic_planner import AtomicPlannerTransformer, UniformD3PM


class AtomicPlanTests(unittest.TestCase):
    def test_segments_and_refinement(self):
        labels = torch.tensor([1, 1, 2, 1, 1, 0, 0, 3, 0, 0])
        voted = majority_vote(labels, window_size=3)
        self.assertEqual(voted.tolist()[:5], [1, 1, 1, 1, 1])
        refined = refine_plan(labels, vote_window=3, min_length=3)
        self.assertTrue(all(segment.length >= 3 for segment in labels_to_segments(refined)))

    def test_duration_nearest_retrieval_and_draft(self):
        library = AtomicMotionLibrary(
            {1: [torch.ones(2, 3), torch.full((5, 3), 5.0)], 2: [torch.full((3, 3), 2.0)]}
        )
        retrieved = library.retrieve(1, 4)
        self.assertEqual(tuple(retrieved.shape), (4, 3))
        self.assertTrue(torch.allclose(retrieved, torch.full((4, 3), 5.0)))
        draft, mask = library.build_draft(torch.tensor([1, 1, 0, 2, 2, 2]), feature_dim=3)
        self.assertEqual(mask.squeeze(-1).tolist(), [1, 1, 0, 1, 1, 1])
        self.assertTrue(torch.equal(draft[2], torch.zeros(3)))
        self.assertEqual(
            plan_boundaries(torch.tensor([1, 1, 0, 2, 2])).tolist(),
            [False, False, True, True, False],
        )


class D3PMTests(unittest.TestCase):
    def test_training_and_sampling_shapes(self):
        model = AtomicPlannerTransformer(
            num_atomic_classes=4,
            music_dim=6,
            latent_dim=16,
            num_layers=1,
            num_heads=4,
            ff_size=32,
            dropout=0.0,
            max_seq_len=12,
        )
        diffusion = UniformD3PM(model, num_steps=4)
        labels = torch.randint(0, 5, (2, 12))
        music = torch.randn(2, 12, 6)
        output = diffusion.training_step(labels, music, timesteps=torch.tensor([0, 3]))
        self.assertEqual(tuple(output.logits.shape), (2, 12, 5))
        self.assertTrue(torch.isfinite(output.loss))
        sampled = diffusion.sample(music, deterministic=True)
        self.assertEqual(tuple(sampled.shape), (2, 12))
        self.assertTrue(torch.all((sampled >= 0) & (sampled < 5)))

    def _tiny_planner(self, activation, num_layers=1):
        return AtomicPlannerTransformer(
            num_atomic_classes=4,
            music_dim=6,
            latent_dim=16,
            num_layers=num_layers,
            num_heads=4,
            ff_size=32,
            dropout=0.0,
            max_seq_len=12,
            activation=activation,
        )

    def test_activation_variants_forward(self):
        labels = torch.randint(0, 5, (2, 12))
        music = torch.randn(2, 12, 6)
        for activation in ("square", "gaussian", "poly2"):
            diffusion = UniformD3PM(self._tiny_planner(activation), num_steps=4)
            output = diffusion.training_step(labels, music, timesteps=torch.tensor([0, 3]))
            self.assertEqual(tuple(output.logits.shape), (2, 12, 5))
            self.assertTrue(torch.isfinite(output.loss))
            sampled = diffusion.sample(music, deterministic=True)
            self.assertEqual(tuple(sampled.shape), (2, 12))

    def test_parabolic_activation_learnable_per_layer(self):
        diffusion = UniformD3PM(self._tiny_planner("poly2", num_layers=2), num_steps=4)
        coeffs = {
            name: param
            for name, param in diffusion.named_parameters()
            if name.endswith("activation.coeffs")
        }
        self.assertEqual(len(coeffs), 2)  # independent parabola per encoder layer
        output = diffusion.training_step(
            torch.randint(0, 5, (2, 12)), torch.randn(2, 12, 6), timesteps=torch.tensor([0, 3])
        )
        output.loss.backward()
        for param in coeffs.values():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())

    def test_default_activation_keeps_checkpoint_layout(self):
        state = self._tiny_planner("gelu").state_dict()
        self.assertFalse(any("activation" in key for key in state))


class CompletionTests(unittest.TestCase):
    def test_completion_loss_and_sampling_shapes(self):
        decoder = AtomicCompletionDecoder(
            motion_dim=7,
            seq_len=8,
            music_dim=6,
            latent_dim=16,
            ff_size=32,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )
        diffusion = AtomicCompletionDiffusion(decoder, num_steps=3)
        clean = torch.randn(2, 8, 7)
        music = torch.randn(2, 8, 6)
        draft = torch.randn(2, 8, 7)
        mask = torch.full((2, 8, 1), 0.25)
        boundaries = torch.zeros(2, 8, dtype=torch.bool)
        boundaries[:, 4] = True
        losses = diffusion.training_step(
            clean, music, draft, mask, boundaries, timesteps=torch.tensor([0, 2])
        )
        self.assertTrue(torch.isfinite(losses.total))
        sampled = diffusion.sample(music, draft, mask, guidance_weight=1.0)
        self.assertEqual(tuple(sampled.shape), tuple(clean.shape))


class AtomicDatasetTests(unittest.TestCase):
    def test_indexed_array_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            split = "{}/train".format(directory)
            os.makedirs(split)
            np.save("{}/motion.npy".format(split), np.zeros((2, 3, 7), dtype=np.float32))
            np.save("{}/music.npy".format(split), np.zeros((2, 3, 6), dtype=np.float32))
            np.save("{}/labels.npy".format(split), np.array([[0, 1, 1], [2, 2, 0]], dtype=np.uint8))
            with open("{}/names.json".format(split), "w") as handle:
                json.dump(["first", "second"], handle)

            dataset = AtomicSequenceDataset(directory, split="train")
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[1]["name"], "second")
            batch = collate_atomic_sequences([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["motion"].shape), (2, 3, 7))
            self.assertEqual(batch["labels"].dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
