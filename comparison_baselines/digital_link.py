"""基于 Sionna 2.2.0 PyTorch 的流式 LDPC/QAM 文件传输。

帧格式为 ``magic(8) | uint64 大端长度 | SHA256(32) | 文件``，共 48 字节头。
每个 k 位信息块保存 ``净载荷及零填充 | CRC32(4)``。CRC 覆盖整个净载荷区。
SNR 定义为 Es/N0；Gray QAM 星座的平均 Es=1，复高斯噪声 E[|w|²]=N0。
编码位按固定公开 SplitMix64 计数器序列加扰；接收端按帧顺序生成同一序列解扰。
接收成功仅取决于收到的 CRC、帧头、长度和 SHA256。BER 使用发送端数据作诊断。
生产安装须使用独立环境：Sionna==2.2.0、Python>=3.11、torch>=2.9.1。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import struct
import sys
import tempfile
import time
from typing import Callable
import zlib


_MAGIC = b"MGDL0001"
_HEADER = struct.Struct(">8sQ32s")
_CRC = struct.Struct(">I")
_SCRAMBLER = {
    "algorithm": "splitmix64-counter-v1", "seed": 0,
    "counter_unit": "64 coded bits", "bit_order": "little",
    "reset": "file", "shared_public_configuration": True,
}


def _scrambling_bits(np, *, start_bit: int, bit_count: int):
    """公开基带协议序列，按整个文件的编码位编号，与分批和噪声种子无关。

    第 j 个 uint64 字的输入为 seed+(j+1)*0x9e3779b97f4a7c15（模 2**64），
    再执行 SplitMix64 的两轮异或移位和乘法，最后低位优先展开。
    接收端只需要共享这些常量、帧顺序和码字长度。
    """
    first_word, skip = divmod(start_bit, 64)
    count = (skip + bit_count + 63) // 64
    words = np.arange(first_word, first_word + count, dtype=np.uint64)
    words = np.uint64(_SCRAMBLER["seed"]) + (words + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)
    words = (words ^ (words >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    words = (words ^ (words >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    words = words ^ (words >> np.uint64(31))
    bits = np.unpackbits(words.astype("<u8", copy=False).view(np.uint8), bitorder="little")
    return bits[skip:skip + bit_count]


@dataclass(frozen=True)
class LinkConfig:
    k: int = 4096
    n: int = 6144
    bits_per_symbol: int = 4
    decoder_iterations: int = 20
    batch_size: int = 32
    device: str = "cuda"

    def __post_init__(self):
        for name in ("k", "n", "bits_per_symbol", "decoder_iterations", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if self.k <= 32 or self.k > 8448 or self.k % 8:
            raise ValueError("k 必须是 32 < k <= 8448 的 8 的倍数，CRC32 包含在 k 中")
        if self.n <= self.k or self.n > 316 * 384:
            raise ValueError("n 必须大于 k，且不超过 5G LDPC 上限 121344")
        if self.bits_per_symbol % 2 or self.n % self.bits_per_symbol:
            raise ValueError("QAM bits_per_symbol 必须为正偶数，且整除 n")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device 必须是非空设备字符串")


def _receive_file(wire, destination: Path, *, payload_bytes: int, codewords: int,
                  crc_error_blocks: int) -> dict:
    """独立接收端：只读取实际解码的数据和 CRC 结果。"""
    failures = []
    if crc_error_blocks:
        failures.append("crc32_mismatch")
    wire.seek(0)
    header = wire.read(_HEADER.size)
    if len(header) != _HEADER.size:
        return {"failure_reasons": failures + ["truncated_header"], "received_bytes": None,
                "received_sha256": None}
    magic, length, expected_hash = _HEADER.unpack(header)
    if magic != _MAGIC:
        failures.append("invalid_magic")
    capacity = payload_bytes * codewords
    if length > capacity - _HEADER.size or (_HEADER.size + length + payload_bytes - 1) // payload_bytes != codewords:
        failures.append("invalid_length")
    result = {"failure_reasons": failures, "received_bytes": length,
              "received_sha256": None, "received_header_sha256": expected_hash.hex()}
    if failures:
        return result
    # 临时文件与目的文件位于同一目录；成功后原子替换。
    staged_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".digital-link-",
                                         suffix=".part", delete=False) as staged:
            staged_name = Path(staged.name)
            digest = hashlib.sha256()
            remaining = length
            while remaining:
                chunk = wire.read(min(1024 * 1024, remaining))
                if not chunk:
                    failures.append("truncated_payload")
                    break
                digest.update(chunk)
                staged.write(chunk)
                remaining -= len(chunk)
            result["received_sha256"] = digest.hexdigest()
            if digest.digest() != expected_hash:
                failures.append("sha256_mismatch")
            if any(wire.read()):
                failures.append("nonzero_padding")
        if not failures:
            staged_name.replace(destination)
    finally:
        if staged_name is not None:
            staged_name.unlink(missing_ok=True)
    return result


def transmit_file(source, destination, *, config: LinkConfig, snr_db: float, seed: int,
                  noiseless: bool = False, progress: Callable[[dict], None] | None = None) -> dict:
    """传输一个文件；有效试验返回 success/failed，依赖或程序错误抛出异常。

    ``progress`` 接收含 completed_codewords、total_codewords 和 crc_error_blocks 的字典。
    无噪声模式仍实际执行 LDPC 编解码、QAM 映射和软解调。
    """
    started = time.perf_counter()
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or (destination.exists() and source.samefile(destination)):
        raise ValueError("source 源文件与 destination 目的文件必须不同")
    if not source.is_file():
        raise FileNotFoundError(source)
    if not isinstance(config, LinkConfig):
        raise TypeError("config 必须是 LinkConfig")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed 必须是 [0, 2**64) 范围的整数")
    snr_db = float(snr_db)
    if not math.isfinite(snr_db):
        raise ValueError("snr_db 必须是有限数值")
    try:
        n0 = 10.0 ** (-snr_db / 10.0)
    except OverflowError as error:
        raise ValueError("snr_db 超出浮点噪声功率范围") from error
    if not math.isfinite(n0) or not 1e-30 <= n0 <= 1e30:
        raise ValueError("snr_db 对应的噪声功率必须位于 [1e-30, 1e30]")

    try:
        import numpy as np
        import torch
        from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
        from sionna.phy.mapping import Constellation, Mapper, Demapper
        from sionna.phy.channel import AWGN
    except ImportError as error:
        raise RuntimeError("数字链路需要独立环境中的 Sionna==2.2.0 PyTorch 版本，请安装所列依赖") from error
    version = importlib.metadata.version("sionna")
    if version != "2.2.0":
        raise RuntimeError(f"数字链路要求 Sionna==2.2.0，当前为 {version}")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"请求设备 {device}，当前 PyTorch 无可用 CUDA")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("数字链路 device 仅支持 cpu 或 cuda")
    encoder = LDPC5GEncoder(config.k, config.n, num_bits_per_symbol=config.bits_per_symbol,
                            precision="single", device=str(device))
    decoder = LDPC5GDecoder(encoder, num_iter=config.decoder_iterations, hard_out=True,
                            return_infobits=True, precision="single", device=str(device))
    constellation = Constellation("qam", config.bits_per_symbol, normalize=True,
                                   precision="single", device=str(device))
    mapper = Mapper(constellation=constellation, device=str(device))
    demapper = Demapper("app", constellation=constellation, device=str(device))
    channel = AWGN(device=str(device))
    generator = channel.torch_rng
    previous_rng_state = generator.get_state()
    generator.manual_seed(seed)
    payload_bytes = config.k // 8 - _CRC.size
    destination.parent.mkdir(parents=True, exist_ok=True)
    bit_errors = error_blocks = crc_errors = completed = coded_bit_errors = 0
    transmitted_energy = 0.0
    try:
        with source.open("rb") as input_file, tempfile.TemporaryFile(mode="w+b", dir=destination.parent) as wire:
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
            input_file.seek(0)
            source_hash = digest.hexdigest()
            pending = _HEADER.pack(_MAGIC, size, digest.digest())
            codewords = (_HEADER.size + size + payload_bytes - 1) // payload_bytes
            with torch.inference_mode():
                for offset in range(0, codewords, config.batch_size):
                    count = min(config.batch_size, codewords - offset)
                    packets = []
                    for _ in range(count):
                        payload, pending = pending[:payload_bytes], pending[payload_bytes:]
                        payload += input_file.read(payload_bytes - len(payload))
                        payload = payload.ljust(payload_bytes, b"\x00")
                        packets.append(payload + _CRC.pack(zlib.crc32(payload)))
                    packed = np.frombuffer(b"".join(packets), dtype=np.uint8).reshape(count, config.k // 8)
                    bits = torch.from_numpy(np.unpackbits(packed, axis=-1)).to(device=device, dtype=torch.float32)
                    coded = encoder(bits)
                    mask = torch.from_numpy(_scrambling_bits(
                        np, start_bit=offset * config.n, bit_count=count * config.n,
                    ).reshape(count, config.n)).to(device=device)
                    scrambled = torch.logical_xor(coded.bool(), mask.bool()).to(dtype=coded.dtype)
                    symbols = mapper(scrambled)
                    transmitted_energy += float(symbols.abs().square().sum(dtype=torch.float64).item())
                    received = symbols if noiseless else channel(symbols, n0)
                    # 零噪声时用有限正值计算软信息，避免除零；信道没有加噪声。
                    llrs = demapper(received, 1e-12 if noiseless else n0)
                    llrs = llrs * (1 - 2 * mask.to(dtype=llrs.dtype))
                    decoded = decoder(llrs)
                    differences = decoded != bits
                    bit_errors += int(differences.sum().item())
                    error_blocks += int(differences.any(dim=-1).sum().item())
                    coded_bit_errors += int(((llrs > 0) != coded.bool()).sum().item())
                    recovered = np.packbits(decoded.to(dtype=torch.uint8).cpu().numpy(), axis=-1)
                    for packet in recovered:
                        packet = packet.tobytes()
                        payload, received_crc = packet[:-_CRC.size], packet[-_CRC.size:]
                        crc_errors += int(zlib.crc32(payload) != _CRC.unpack(received_crc)[0])
                        wire.write(payload)
                    completed += count
                    if progress is not None:
                        progress({"completed_codewords": completed, "total_codewords": codewords,
                                  "crc_error_blocks": crc_errors})
            received_info = _receive_file(wire, destination, payload_bytes=payload_bytes,
                                          codewords=completed, crc_error_blocks=crc_errors)
    finally:
        generator.set_state(previous_rng_state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    torch_release = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:3])
    return {
        "status": "failed" if received_info["failure_reasons"] else "success",
        "source": str(source), "destination": str(destination),
        "source_bytes": size, "source_sha256": source_hash, **received_info,
        "config": asdict(config), "snr_db": snr_db, "snr_definition": "Es/N0",
        "constellation_average_energy": 1.0, "noise_variance": 0.0 if noiseless else n0,
        "mean_complex_power": transmitted_energy / (completed * (config.n // config.bits_per_symbol)),
        "scrambler": dict(_SCRAMBLER),
        "noiseless": noiseless, "seed": seed, "codewords": completed,
        "header_bytes": _HEADER.size, "crc_bytes": completed * _CRC.size,
        "payload_bytes_per_codeword": payload_bytes,
        "padding_bytes": completed * payload_bytes - _HEADER.size - size,
        "information_bits": completed * config.k, "coded_bits": completed * config.n,
        "total_channel_uses": completed * (config.n // config.bits_per_symbol),
        "bit_errors": bit_errors, "ber": bit_errors / (completed * config.k),
        "error_blocks": error_blocks, "block_error_rate": error_blocks / completed,
        "crc_error_blocks": crc_errors, "coded_bit_errors": coded_bit_errors,
        "coded_ber": coded_bit_errors / (completed * config.n),
        "elapsed_seconds": time.perf_counter() - started,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "sionna": version, "numpy": np.__version__},
        "sionna_torch_requirement": ">=2.9.1", "torch_meets_requirement": torch_release >= (2, 9, 1),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("transmit", "self-test"):
        command = subparsers.add_parser(name)
        for field in ("k", "n", "bits_per_symbol", "decoder_iterations", "batch_size"):
            command.add_argument("--" + field.replace("_", "-"), type=int, default=getattr(LinkConfig(), field))
        command.add_argument("--device", default="cuda")
        command.add_argument("--snr-db", type=float, default=30.0 if name == "self-test" else 10.0)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--noiseless", action="store_true")
        command.add_argument("--report", type=Path)
        if name == "transmit":
            command.add_argument("--source", type=Path, required=True)
            command.add_argument("--destination", type=Path, required=True)
            command.add_argument("--progress-every-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    config = LinkConfig(**{field: getattr(args, field) for field in LinkConfig.__dataclass_fields__})
    if args.command == "self-test":
        with tempfile.TemporaryDirectory(prefix="digital-link-self-test-") as directory:
            source, output = Path(directory) / "source.bin", Path(directory) / "received.bin"
            source.write_bytes(bytes(range(256)) * 2 + b"LDPC/QAM self-test")
            result = transmit_file(source, output, config=config, snr_db=args.snr_db,
                                   seed=args.seed, noiseless=args.noiseless)
            if result["status"] == "success" and output.read_bytes() != source.read_bytes():
                raise RuntimeError("self-test 文件恢复内容不匹配")
    else:
        if not math.isfinite(args.progress_every_seconds) or args.progress_every_seconds <= 0:
            parser.error("--progress-every-seconds 必须是有限正数")
        progress_started = time.perf_counter()
        last_progress = None

        def report_progress(state):
            nonlocal last_progress
            now = time.perf_counter()
            completed, total = state["completed_codewords"], state["total_codewords"]
            if last_progress is None or completed == total or now - last_progress >= args.progress_every_seconds:
                elapsed = now - progress_started
                eta = elapsed * (total - completed) / completed
                print(f"[digital-link] codewords={completed}/{total} "
                      f"crc_errors={state['crc_error_blocks']} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                      file=sys.stderr, flush=True)
                last_progress = now

        result = transmit_file(args.source, args.destination, config=config,
                               snr_db=args.snr_db, seed=args.seed, noiseless=args.noiseless,
                               progress=report_progress)
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 1 if args.command == "self-test" and result["status"] != "success" else 0


if __name__ == "__main__":
    raise SystemExit(main())
