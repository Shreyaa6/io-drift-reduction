"""Download a pinned AirIO EuRoC IMU/reference subset; no images or GPS."""
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = 'https://github.com/Air-IO/Air-IO/releases/download/datasets/EuRoC-Dataset.zip'
SHA256 = 'de09fdd03e9147461fd678aa6763454bdf8d1f07172753a9ec80073e9be38e4a'
ROOT = Path(__file__).resolve().parents[1]


def extract(archive, output):
    archive, output = Path(archive), Path(output)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f'Archive checksum mismatch: {digest}; review upstream change before updating pin')
    count = 0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            parts = Path(info.filename).parts
            if len(parts) < 4 or parts[0] != 'EuRoC-Dataset' or '..' in parts:
                continue
            if parts[-1] not in {'data.csv', 'sensor.yaml'}:
                continue
            if parts[3] not in {'imu0', 'state_groundtruth_estimate0'}:
                continue
            dest = output.joinpath(*parts[1:])
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, dest.open('wb') as target:
                shutil.copyfileobj(src, target)
            count += 1
    output.mkdir(parents=True, exist_ok=True)
    (output / 'provenance.json').write_text(json.dumps({'url': URL, 'sha256': digest,
        'original': 'https://projects.asl.ethz.ch/datasets/euroc-mav/', 'files': count}, indent=2))
    return count


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'data/euroc')
    args = p.parse_args()
    archive = args.archive or ROOT / 'data/EuRoC-Dataset.zip'
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix('.part')
        urllib.request.urlretrieve(URL, temporary)
        temporary.replace(archive)
    print(f'Extracted {extract(archive, args.output)} IMU/reference files to {args.output}')
