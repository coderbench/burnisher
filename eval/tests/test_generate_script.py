#!/usr/bin/env python3
"""`scripts/generate.py` turns the decoder's pixels into an image a person can open.

The runtime itself is not run here. What is checked is the part that would otherwise only be found
by looking at a broken PNG on a GPU box: the decoder's [-1, 1] range, the channel order, and a PNG
that any reader accepts.
"""
from __future__ import annotations

import struct
import subprocess
import sys
import unittest
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import generate  # noqa: E402


def read_png(data):
    """Just enough of a PNG reader to check what png_bytes writes: 8-bit RGB, filter 0."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, {}
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        tag, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        (crc,) = struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])
        assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF, tag
        chunks.setdefault(tag, b"")
        chunks[tag] += body
        pos += 12 + n
    w, h, depth, colour = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert (depth, colour) == (8, 2)
    raw = zlib.decompress(chunks[b"IDAT"])
    rows = [raw[y * (3 * w + 1):(y + 1) * (3 * w + 1)] for y in range(h)]
    assert all(r[0] == 0 for r in rows)
    return np.frombuffer(b"".join(r[1:] for r in rows), dtype=np.uint8).reshape(h, w, 3)


class TestGenerateScript(unittest.TestCase):
    def test_the_decoder_range_maps_onto_bytes(self):
        pixels = np.array([-1.0, 0.0, 1.0, 3.0], dtype=np.float32).reshape(1, 1, 1, 4)
        pixels = np.repeat(pixels, 3, axis=1)
        self.assertEqual(generate.to_rgb8(pixels)[0, :, 0].tolist(), [0, 128, 255, 255])

    def test_channels_are_red_green_blue_in_that_order(self):
        pixels = -np.ones((1, 3, 2, 2), dtype=np.float32)
        pixels[0, 0] = 1.0
        rgb = generate.to_rgb8(pixels)
        self.assertEqual(rgb.shape, (2, 2, 3))
        self.assertEqual(rgb[1, 1].tolist(), [255, 0, 0])

    def test_a_latent_is_refused_rather_than_written_as_an_image(self):
        with self.assertRaises(ValueError):
            generate.to_rgb8(np.zeros((1, 4, 128, 128), dtype=np.float32))

    def test_the_png_round_trips(self):
        rgb = np.random.default_rng(0).integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
        np.testing.assert_array_equal(read_png(generate.png_bytes(rgb)), rgb)

    def test_both_scripts_answer_help_without_a_gpu_stack(self):
        for script in ("generate.py", "pytorch_baseline.py"):
            subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--help"],
                           check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
