"""Create a small allowlisted project ZIP for Colab, excluding secrets/weights."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def prepare(destination=None):
    destination = destination or ROOT / 'artifacts/colab-project.zip'
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / 'README.md', ROOT / 'training/system_prompt.txt', ROOT / 'training/requirements-baseline.txt']
    for directory, pattern in [('evaluation', '*.py'), ('data', '*.jsonl'), ('data', '*.json')]:
        paths.extend(sorted((ROOT / directory).glob(pattern)))
    for pattern in ('*.py', 'requirements-training.txt', 'baseline_reference.json', 'final_test_reference.json'):
        paths.extend(sorted((ROOT / 'training').glob(pattern)))
    with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, 'ticket-triage/' + path.relative_to(ROOT).as_posix())
    return destination


if __name__ == '__main__':
    print(prepare())
