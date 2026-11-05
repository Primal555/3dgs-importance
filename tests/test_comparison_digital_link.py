"""数字文件链路的定向测试；集成测试需要独立环境中的 Sionna 2.2.0。"""

import importlib
import hashlib
import io
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest


def link_module():
    # 缺失实现必须表现为明确的功能失败。
    assert Path("comparison_baselines/digital_link.py").is_file(), "数字文件链路尚未实现"
    return importlib.import_module("comparison_baselines.digital_link")


def real_link():
    link = link_module()
    pytest.importorskip("sionna.phy.fec.ldpc")
    return link


def test_rejects_invalid_block_and_modulation_config():
    link = link_module()
    for kwargs in ({"k": 33}, {"n": 6143}, {"bits_per_symbol": 3},
                   {"batch_size": 0}, {"decoder_iterations": 0}, {"k": 6144}):
        with pytest.raises((ValueError, TypeError)):
            link.LinkConfig(**kwargs)


@pytest.mark.parametrize("noiseless,snr_db", [(True, 10.0), (False, 30.0)])
def test_real_ldpc_recovers_bytes_and_charges_header_crc_padding(tmp_path, noiseless, snr_db):
    link = real_link()
    source = tmp_path / "源文件.bin"
    source.write_bytes(bytes(range(121)))
    output = tmp_path / "received.bin"
    progress = []
    result = link.transmit_file(
        source, output,
        config=link.LinkConfig(k=512, n=768, batch_size=2, device="cpu"),
        snr_db=snr_db, seed=12, noiseless=noiseless, progress=progress.append,
    )
    assert result["status"] == "success"
    assert output.read_bytes() == source.read_bytes()
    # 48 字节文件头；每个 64 字节信息块有 4 字节 CRC，净载荷为 60 字节。
    assert result["codewords"] == 3
    assert result["header_bytes"] == 48
    assert result["crc_bytes"] == 12
    assert result["padding_bytes"] == 11
    assert result["total_channel_uses"] == 576
    assert result["coded_bits"] == 2304
    assert result["information_bits"] == 1536
    expected_cost = link.file_channel_cost(121, link.LinkConfig(k=512, n=768))
    for key, value in expected_cost.items():
        assert result[key] == value
    assert result["crc_error_blocks"] == result["bit_errors"] == 0
    assert result["ber"] == 0.0
    assert result["snr_definition"] == "Es/N0"
    assert result["scrambler"] == {
        "algorithm": "splitmix64-counter-v1", "seed": 0,
        "counter_unit": "64 coded bits", "bit_order": "little",
        "reset": "file", "shared_public_configuration": True,
    }
    assert 0.85 < result["mean_complex_power"] < 1.15
    assert result["versions"]["sionna"] == "2.2.0"
    assert progress[-1]["completed_codewords"] == 3
    assert result["elapsed_seconds"] > 0


def test_low_snr_fails_without_publishing_file_and_is_seeded(tmp_path):
    link = real_link()
    source = tmp_path / "payload.bin"
    source.write_bytes(bytes(range(121)))
    config = link.LinkConfig(k=512, n=768, batch_size=2, device="cpu")
    first = link.transmit_file(source, tmp_path / "a.bin", config=config, snr_db=-20, seed=77)
    second = link.transmit_file(source, tmp_path / "b.bin", config=config, snr_db=-20, seed=77)
    assert first["status"] == second["status"] == "failed"
    assert not (tmp_path / "a.bin").exists()
    assert not (tmp_path / "b.bin").exists()
    assert first["crc_error_blocks"] > 0
    assert first["bit_errors"] > 0
    assert first["bit_errors"] == second["bit_errors"]
    assert first["total_channel_uses"] == second["total_channel_uses"] == 576
    assert first["failure_reasons"]


def test_empty_file_is_transmitted_and_charged(tmp_path):
    link = real_link()
    source = tmp_path / "empty"
    source.write_bytes(b"")
    result = link.transmit_file(source, tmp_path / "output", config=link.LinkConfig(
        k=512, n=768, device="cpu"), snr_db=10, seed=0, noiseless=True)
    assert result["status"] == "success"
    assert (tmp_path / "output").read_bytes() == b""
    assert result["total_channel_uses"] == 192
    assert result["padding_bytes"] == 12


def test_rejects_source_destination_alias(tmp_path):
    link = link_module()
    source = tmp_path / "source"
    source.write_bytes(b"keep")
    with pytest.raises(ValueError, match="source|源"):
        link.transmit_file(source, source, config=link.LinkConfig(device="cpu"), snr_db=10, seed=1)
    assert source.read_bytes() == b"keep"


def test_public_scrambler_sequence_is_independent_of_batch_boundaries():
    import numpy as np
    link = link_module()
    # SplitMix64 从 seed=0 的两个标准输出，以每字低位优先展开。
    expected = bytes.fromhex("afcd1d7b39a820e2f465b9a16a9e786e")
    complete = link._scrambling_bits(np, start_bit=0, bit_count=128)
    assert np.packbits(complete, bitorder="little").tobytes() == expected
    chunks = [link._scrambling_bits(np, start_bit=0, bit_count=5),
              link._scrambling_bits(np, start_bit=5, bit_count=123)]
    assert np.array_equal(np.concatenate(chunks), complete)


def test_reports_power_of_actual_mapped_symbols(tmp_path, monkeypatch):
    import numpy as np
    link = real_link()
    import sionna.phy.mapping as mapping
    original_mapper = mapping.Mapper
    transmitted = []

    def recording_mapper(*args, **kwargs):
        actual_mapper = original_mapper(*args, **kwargs)

        def map_bits(bits):
            symbols = actual_mapper(bits)
            transmitted.append(symbols.detach().cpu().numpy().copy())
            return symbols

        return map_bits

    monkeypatch.setattr(mapping, "Mapper", recording_mapper)
    source = tmp_path / "zeros"
    source.write_bytes(bytes(121))
    result = link.transmit_file(source, tmp_path / "received", config=link.LinkConfig(
        k=512, n=768, batch_size=2, device="cpu"), snr_db=30, seed=15, noiseless=True)
    samples = np.concatenate([batch.reshape(-1) for batch in transmitted]).astype(np.complex128)
    measured = float(np.mean(samples.real ** 2 + samples.imag ** 2))
    assert result["status"] == "success"
    assert result["mean_complex_power"] == pytest.approx(measured, abs=1e-6)


def test_cli_self_test_uses_real_link(tmp_path):
    real_link()
    command = [sys.executable, "-B", "-X", "utf8", "-m", "comparison_baselines.digital_link",
               "self-test", "--device", "cpu", "--k", "512", "--n", "768"]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=True)
    result = json.loads(completed.stdout)
    assert result["status"] == "success"
    assert result["total_channel_uses"] > 0


@pytest.mark.parametrize("damage,reason", [
    ("sha", "sha256_mismatch"), ("magic", "invalid_magic"),
    ("length", "invalid_length"), ("crc", "crc32_mismatch"),
    ("padding", "nonzero_padding"),
])
def test_receiver_rejects_corrupt_received_metadata_or_payload(tmp_path, damage, reason):
    link = link_module()
    payload = b"abc"
    magic = b"BADMAGIC" if damage == "magic" else b"MGDL0001"
    length = 1000 if damage == "length" else 3
    wire = struct.pack(">8sQ32s", magic, length, hashlib.sha256(payload).digest())
    wire += b"abd" if damage == "sha" else payload
    wire += b"\x01" + b"\x00" * 8 if damage == "padding" else b"\x00" * 9
    output = tmp_path / "received"
    output.write_bytes(b"previous verified file")
    result = link._receive_file(io.BytesIO(wire), output, payload_bytes=60, codewords=1,
                                crc_error_blocks=int(damage == "crc"))
    assert reason in result["failure_reasons"]
    assert output.read_bytes() == b"previous verified file"


def test_cli_failed_trial_reports_cost_and_exits_zero(tmp_path):
    real_link()
    source, output, report = tmp_path / "source", tmp_path / "received", tmp_path / "report.json"
    source.write_bytes(bytes(range(241)))
    command = [sys.executable, "-B", "-X", "utf8", "-m", "comparison_baselines.digital_link",
               "transmit", "--source", str(source), "--destination", str(output),
               "--report", str(report), "--device", "cpu", "--k", "512", "--n", "768",
               "--snr-db", "-20", "--seed", "77", "--batch-size", "2",
               "--progress-every-seconds", "10"]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=True)
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result == json.loads(completed.stdout)
    assert result["status"] == "failed"
    assert result["total_channel_uses"] == 960
    assert not output.exists()
    progress_lines = [line for line in completed.stderr.splitlines() if "[digital-link]" in line]
    assert len(progress_lines) == 2
    assert "codewords=2/5" in progress_lines[0]
    assert "codewords=5/5" in progress_lines[-1]
    assert all("crc_errors=" in line and "elapsed=" in line and "eta=" in line for line in progress_lines)
