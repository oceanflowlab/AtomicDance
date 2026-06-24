# Music-to-Dance Generation via Atomic Movements

<!-- Replace the # targets below when the public resources are available. -->
[![Paper](https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red)](#)
[![arXiv](https://img.shields.io/badge/arXiv-Coming_Soon-b31b1b.svg)](#)
[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=githubpages&logoColor=blue)](#)
[![Dataset](https://img.shields.io/badge/Google_Drive-Storage-dfa12b?style=flat&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1ETsaetMMWeKV3_E3Lr40BdybAsUAG8WM/view?usp=sharing)
[![YouTube](https://img.shields.io/badge/YouTube-Coming_Soon-red?style=plastic&logo=youtube&logoColor=red)](#)

This repository is the official PyTorch implementation of the paper
**Music-to-Dance Generation via Atomic Movements**.

**Xinhao Cai**, **Yixuan Sun**, **Minghang Zheng**, **Qingchao Chen**,
**Xin Jin**, **Song-chun Zhu**, and **Yang Liu**

[Paper](#) | [arXiv](#) | [Project](#) | [Dataset](https://drive.google.com/file/d/1ETsaetMMWeKV3_E3Lr40BdybAsUAG8WM/view?usp=sharing) | [YouTube](#)

Music-driven dance generation should produce motion that is rhythmically
synchronized with music while preserving coherent choreographic structure.
Existing end-to-end methods usually model dance as a continuous signal and
overlook its compositional nature. We instead represent choreography as a
sequence of semantically interpretable and reusable **atomic movements**.

We first construct an atomic movement vocabulary by segmenting dance sequences,
clustering recurring motion patterns, and refining their semantics with
LLM-assisted relabeling. We then introduce a two-stage generation framework
that mirrors the choreography process. A full-music-aware planner predicts the
type, timing, and duration of atomic movements. A transition-aware diffusion
model retrieves suitable movement prototypes, re-creates them with variations,
and synthesizes smooth, musically aligned transitions. The explicit symbolic
plan also enables users to replace movements, adjust durations, and edit dance
structure without retraining.

Our paper was accepted by ECCV 2026.

<!-- Add the public teaser/framework image here when available.
<div align="center">
  <img src="assets/teaser.png" width="90%">
</div>
-->

## Environment Setup

### Installation

The code was validated on Linux with Python 3.7.12, PyTorch 1.12.1, and CUDA
11.6. A CUDA GPU with at least 16 GB memory is recommended for training and
inference.

1. Create the Conda environment.

```bash
conda create -n atomicdance python=3.7 -y
conda activate atomicdance
```

2. Install PyTorch and the base Python dependencies.

```bash
pip install -r requirements.txt
```

3. Install the packages that import or compile against PyTorch.

```bash
pip install git+https://github.com/rodrigo-castellon/jukemirlib.git@a91d87fcae0dd89085752421e794ea7e1b300735
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.1
```

If its installation fails, install PyTorch3D 0.7.1 separately with the matching CUDA toolchain.

### Data Preparation

Download the processed atomic dataset from [Dataset](https://drive.google.com/file/d/1ETsaetMMWeKV3_E3Lr40BdybAsUAG8WM/view?usp=sharing) and extract it under
`data/atomic_aistpp/`. No additional label preprocessing is required.

```text
data/atomic_aistpp/
  manifest.json
  normalizer.pt
  train/
    motion.npy
    music.npy
    labels.npy
    names.json
  test/
    motion.npy
    music.npy
    labels.npy
    names.json
```

The released `atomic_aistpp` package is the only project-specific dataset that
needs to be downloaded. It contains the frame-aligned motion, 35-dimensional
music features, and atomic labels used for training and inference. Atomic labels
`1..100` represent movement categories; label `0` represents a transition.

Evaluation against AIST++ ground truth additionally expects motion PKLs and WAVs
under `data/edge_aistpp/{motions,wavs}`. Obtain AIST++ from its
[official website](https://google.github.io/aistplusplus_dataset/) rather than
from this project release. Feature extraction also requires the licensed SMPL
model at `smpl/SMPL_MALE.pkl`; obtain it from the
[official SMPL website](https://smpl.is.tue.mpg.de/).

## Training

### Atomic Movement Planner

```bash
python train_atomic.py \
  --stage planner \
  --data-root data/atomic_aistpp \
  --output-dir runs/atomic_planner \
  --device cuda \
  --epochs 20 \
  --batch-size 16
```

### Dance Completion Model


```bash
python train_atomic.py \
  --stage completion \
  --data-root data/atomic_aistpp \
  --output-dir runs/atomic_completion \
  --device cuda \
  --epochs 200 \
  --batch-size 8
```

Training reports mean loss every five epochs and saves a resumable checkpoint
every 20 epochs. Use `--resume CHECKPOINT` to continue training. Add
`--max-steps 10` for a bounded debugging run.

## Evaluation

The unified evaluator performs motion generation, feature extraction, caching,
and metric computation. It reports kinematic/manual-feature FID and diversity
and Beat Alignment Score (BAS). Prediction and ground-truth feature
distributions are standardized independently following the provided evaluation
starter.

The commands below evaluate the sequences in
`data/splits/crossmodal_test.txt`.

### Planner Plan + Dance Completion

This is the full two-stage inference setting. Atomic labels are generated by
the planner rather than read from ground truth.

```bash
python -m eval.evaluate \
  --ground-truth-motions data/edge_aistpp/motions \
  --audio-dir data/edge_aistpp/wavs \
  --sequence-list data/splits/crossmodal_test.txt \
  --plan-source planner \
  --planner-checkpoint runs/atomic_planner/<name>.pt \
  --completion-checkpoint runs/atomic_completion/<name>.pt \
  --atomic-data-root data/atomic_aistpp \
  --smpl-model smpl/SMPL_MALE.pkl \
  --device cuda:0 \
  --max-inference-frames 150 \
  --inference-batch-size 4 \
  --workers 4 \
  --inference-output eval/generated_planner \
  --cache-dir eval/cache_planner \
  --output eval/results_planner.json
```



Add `--overwrite-inference --force-extract` to regenerate motions and features
instead of reusing existing caches.

### Pretrained Checkpoints

Pretrained checkpoints will be released at [Checkpoints](https://drive.google.com/drive/folders/1r707t1FKhs_FkHNkNbqtDxIaXiYMUZuq?usp=sharing). The expected
layout is:

```text
runs/
  atomic_planner/
    planner_*.pt
  atomic_completion/
    completion_*.pt
```

## Citation

If you find this project useful, please consider citing our work. The entry
below will be updated when the final publication metadata is available.

```bibtex
@inproceedings{cai2026atomicdance,
  title={Music-to-Dance Generation via Atomic Movements},
  author={Cai, Xinhao and Sun, Yixuan and Zheng, Minghang and Chen, Qingchao and
          Jin, Xin and Zhu, Song-chun and Liu, Yang},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## Acknowledgements

This implementation is built on
[EDGE](https://github.com/Stanford-TML/EDGE). We also thank the authors of
[AIST++](https://google.github.io/aistplusplus_dataset/),
[PyTorch3D](https://github.com/facebookresearch/pytorch3d),
[SMPL](https://smpl.is.tue.mpg.de/), and the related music-to-dance generation
projects used in our experiments.

## License

This project is released under the license in [LICENSE](LICENSE). AIST++, SMPL,
pretrained models, and other third-party assets remain subject to their
respective licenses.
