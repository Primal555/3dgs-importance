"""Streaming, byte-exact PLY packing; the complete header travels on the link."""
import hashlib
import os
from pathlib import Path
import struct
import tempfile
import zlib

HEADER = struct.Struct('!8sQ32s')
MAGIC = b'GSPLY001'
CHUNK = 1 << 20


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def _temporary(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.packing-', dir=destination.parent)
    return destination, os.fdopen(fd, 'w+b'), Path(name)


def pack_ply(source, destination, *, level=6):
    source = Path(source).resolve()
    before = source.stat()
    if not source.is_file() or not 0 <= level <= 9:
        raise ValueError('packing needs a regular file and zlib level 0..9')
    destination, output, temporary = _temporary(destination)
    digest = hashlib.sha256(); count = 0
    try:
        with output, source.open('rb') as stream:
            output.write(bytes(HEADER.size))
            compressor = zlib.compressobj(level)
            for block in iter(lambda: stream.read(CHUNK), b''):
                count += len(block); digest.update(block)
                output.write(compressor.compress(block))
            output.write(compressor.flush())
            output.seek(0)
            output.write(HEADER.pack(MAGIC, count, digest.digest()))
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('source changed while it was being packed')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(input_bytes=count, packed_bytes=destination.stat().st_size,
                source_sha256=digest.hexdigest(), packed_sha256=sha256_file(destination),
                codec='zlib', level=level, header_bytes=HEADER.size,
                reconstruction='byte-exact original PLY')


def unpack_ply(archive, destination, *, max_output_bytes=4 << 30):
    archive = Path(archive)
    destination, output, temporary = _temporary(destination)
    digest = hashlib.sha256(); count = 0
    try:
        with output, archive.open('rb') as stream:
            header = stream.read(HEADER.size)
            if len(header) != HEADER.size:
                raise ValueError('truncated PLY packing header')
            magic, length, expected = HEADER.unpack(header)
            if magic != MAGIC or length > max_output_bytes:
                raise ValueError('invalid PLY packing header or output size')
            decompressor = zlib.decompressobj()
            while block := stream.read(CHUNK):
                pending = block
                while pending:
                    try:
                        restored = decompressor.decompress(pending, min(CHUNK, length-count+1))
                    except zlib.error as exc:
                        raise ValueError('corrupt compressed PLY') from exc
                    count += len(restored)
                    if count > length:
                        raise ValueError('decompressed PLY exceeds its declared length')
                    output.write(restored); digest.update(restored)
                    if decompressor.unused_data:
                        raise ValueError('unexpected bytes after compressed PLY')
                    pending = decompressor.unconsumed_tail
            if not decompressor.eof or count != length or digest.digest() != expected:
                raise ValueError('incomplete PLY or SHA256 mismatch')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(bytes=count, sha256=digest.hexdigest())
