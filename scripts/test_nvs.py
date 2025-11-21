"""Automated smoke tests for VGGT/NVS components."""

import os
import sys
from typing import Tuple

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vggt.heads.dpt_head import DPTHead
from vggt.heads.nvs_head import PluckerEncoder
from vggt.models.aggregator import NVSAggregator
from vggt.models.vggt_nvs import VGGT_NVS


IMG_SIZE = 518
PATCH_SIZE = 14


def _dummy_camera_params(batch: int, n_views: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create simple intrinsics/extrinsics for random-view tests."""

    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0).expand(batch, n_views, 3, 3).clone()

    extrinsics = torch.eye(4, dtype=torch.float32)[:3].unsqueeze(0).unsqueeze(0)
    extrinsics = extrinsics.expand(batch, n_views, 3, 4).clone()

    return intrinsics, extrinsics


def test_aggregator_tokens():
    """Ensure NVSAggregator processes source+target frames and keeps token shapes consistent."""

    batch = 2
    s_in = 3
    s_out = 2
    embed_dim = 128

    aggregator = NVSAggregator(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=embed_dim,
        depth=6,
        num_heads=4,
        patch_embed="conv",
    )

    patches = (IMG_SIZE // PATCH_SIZE) ** 2
    input_images = torch.randn(batch, s_in, 3, IMG_SIZE, IMG_SIZE)
    target_tokens = torch.randn(batch, s_out, patches, embed_dim)

    output_list, patch_start_idx = aggregator(input_images, target_tokens)

    expected_seq_len = patch_start_idx + patches
    assert len(output_list) == aggregator.aa_block_num, "Output list length should match aa_block_num"

    for layer in output_list:
        assert layer.shape[0] == batch
        assert layer.shape[1] == s_in + s_out
        assert layer.shape[2] == expected_seq_len
        assert layer.shape[3] == 2 * embed_dim

    assert patch_start_idx == aggregator.patch_start_idx

    print("✓ NVSAggregator emits the expected token shapes")
    return True


def test_rgb_head_target_only():
    """Verify the RGB head only receives aggregated tokens for target frames."""

    batch = 1
    s_in = 4
    s_out = 2
    embed_dim = 128

    aggregator = NVSAggregator(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=embed_dim,
        depth=24,
        num_heads=4,
        patch_embed="conv",
    )

    plucker_encoder = PluckerEncoder(img_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=embed_dim)
    rgb_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="sigmoid", conf_activation="expp1")

    input_images = torch.randn(batch, s_in, 3, IMG_SIZE, IMG_SIZE)
    target_plucker_images = torch.randn(batch, s_out, 6, IMG_SIZE, IMG_SIZE)

    target_tokens = plucker_encoder(target_plucker_images)
    aggregated_tokens_list, patch_start_idx = aggregator(input_images, target_tokens)

    target_only_tokens = [layer[:, s_in:, :, :] for layer in aggregated_tokens_list]
    target_images = torch.randn(batch, s_out, 3, IMG_SIZE, IMG_SIZE)

    preds, conf = rgb_head(target_only_tokens, target_images, patch_start_idx)

    assert preds.shape == (batch, s_out, IMG_SIZE, IMG_SIZE, 3)
    assert conf.shape == (batch, s_out, IMG_SIZE, IMG_SIZE)

    print("✓ RGB head consumes only target-frame tokens")
    return True


def test_vggt_nvs_integration():
    """Run VGGT_NVS end-to-end with simple camera parameters to smoke-test the forward pass."""

    batch = 1
    s_in = 4
    s_out = 2

    model = VGGT_NVS(img_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=1024)
    model.aggregator = NVSAggregator(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        patch_embed="conv",
    )
    model.plucker_encoder = PluckerEncoder(img_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=1024)

    model.eval()

    input_images = torch.rand(batch, s_in, 3, IMG_SIZE, IMG_SIZE)
    intrinsics, extrinsics = _dummy_camera_params(batch, s_out)

    outputs = model(input_images, intrinsics, extrinsics)

    assert "rgb" in outputs and "rgb_conf" in outputs
    assert outputs["rgb"].shape == (batch, s_out, IMG_SIZE, IMG_SIZE, 3)
    assert outputs["rgb_conf"].shape == (batch, s_out, IMG_SIZE, IMG_SIZE)

    print("✓ VGGT_NVS forward pass runs with random data")
    return True


def main():
    tests = [
        # ("NVSAggregator tokens", test_aggregator_tokens),
        # ("RGB head target-only", test_rgb_head_target_only),
        ("VGGT_NVS integration", test_vggt_nvs_integration),
    ]

    results = []
    print("Running automated VGGT/NVS smoke tests")
    print("=" * 60)

    for name, test_fn in tests:
        test_fn()

    print("=" * 60)


if __name__ == "__main__":
    main()
