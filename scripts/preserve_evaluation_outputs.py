"""Snapshot or verify existing evaluation artifacts without rewriting them."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
MANIFEST = OUT / 'evaluation_preservation_manifest.json'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        saved = json.loads(MANIFEST.read_text())
        for relative, record in saved.items():
            path = ROOT / relative
            stat = path.stat()
            assert stat.st_size == record['bytes'] and stat.st_mtime_ns == record['mtime_ns'], relative
            if 'sha256' in record:
                assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], relative
        print(f'Preserved {len(saved)} original files: sizes/timestamps and artifact hashes verified.')
        return
    if MANIFEST.exists():
        raise FileExistsError('Preservation manifest already exists; use --verify')
    paths = [p for p in OUT.iterdir() if p.is_file() and p.name != 'final_evaluation_report.md']
    paths += list((OUT/'figures').glob('*')) + list((ROOT/'data').glob('*'))
    paths += list(OUT.rglob('*.pt'))
    records = {}
    for path in paths:
        if not path.is_file():
            continue
        stat = path.stat()
        record = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if path.suffix != '.pt':
            record['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        records[str(path.relative_to(ROOT))] = record
    MANIFEST.write_text(json.dumps(records, indent=2))
    print(f'Snapshotted {len(records)} original files for preservation verification.')

if __name__ == '__main__':
    main()
