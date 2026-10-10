"""Small review downloads without deleting training artifacts or reducing metrics."""
import argparse
import json
import os
import zipfile
from pathlib import Path


def representative_views(count, limit):
    if limit is None or limit >= count:
        return set(range(count))
    if limit < 1:
        raise ValueError('image view limit must be positive')
    if limit == 1:
        return {0}
    return {round(i*(count-1)/(limit-1)) for i in range(limit)}


def build_review(root):
    root = Path(root).resolve()
    required = [root/'complete.json', root/'evaluation/initial/evaluation.json',
                root/'evaluation/final/evaluation.json']
    if not all(p.is_file() for p in required):
        raise ValueError('training plus initial/final evaluation must complete before packaging')
    files = []
    excluded = {'checkpoints', 'allocation_latest', '__pycache__'}
    for folder, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in excluded and not (Path(folder)/d).is_symlink()
                   and (Path(folder)/d).resolve().is_relative_to(root)]
        for name in names:
            path = Path(folder)/name
            if (path.suffix not in ('.json', '.jsonl', '.csv', '.png') or
                    name == 'review_manifest.json' or path.is_symlink() or
                    not path.resolve().is_relative_to(root)):
                continue
            files.append(path)
    manifest = {'files': [p.relative_to(root).as_posix() for p in sorted(files)],
                'uncompressed_bytes': sum(p.stat().st_size for p in files),
                'excluded': 'checkpoints, arrays, packets, PLY, PDF duplicates and symlinks; server files retained'}
    archive = root/'review.zip'
    temporary = root/'review.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(files):
            z.write(path, path.relative_to(root).as_posix())
        z.writestr('review_contents.json', json.dumps(manifest, indent=2))
    os.replace(temporary, archive)
    manifest['archive_bytes'] = archive.stat().st_size
    (root/'review_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    print(build_review(parser.parse_args().root))
