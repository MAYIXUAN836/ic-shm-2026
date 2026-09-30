"""Exercise the embedded download/extraction code without network or ML libraries."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / 'reproduce.sh').read_text().split('download_bundle() {', 1)[1].split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]


class DownloadTests(unittest.TestCase):
    def fixture(self, directory, group, content, name):
        root = Path(directory)
        (root / 'config').mkdir(exist_ok=True)
        cache = root / 'downloads' / group
        cache.mkdir(parents=True, exist_ok=True)
        (cache / name).write_bytes(content)
        row = {'name': name, 'id': 'unused', 'group': group, 'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
        (root / 'config/download_manifest.json').write_text(json.dumps({'drive_folder': 'unused', 'files': [row]}))
        # A verified cache must never call the network. Unverified data must not
        # be extracted when downloading a replacement fails.
        wrapper = "import types,sys\nm=types.ModuleType('gdown')\ndef fail(**kw): raise RuntimeError('network disabled in test')\nm.download=fail\nsys.modules['gdown']=m\n"
        return subprocess.run([sys.executable, '-c', wrapper + SCRIPT, str(root), group], capture_output=True, text=True)

    def test_dataset_extract_and_reject_escape(self):
        for path, succeeds in [('official110/image/example.jpg', True), ('../outside.txt', False)]:
            with self.subTest(path=path), tempfile.TemporaryDirectory() as directory:
                content = io.BytesIO()
                with zipfile.ZipFile(content, 'w') as archive:
                    archive.writestr(path, b'fixture')
                result = self.fixture(directory, 'datasets', content.getvalue(), 'data.zip')
                self.assertEqual(result.returncode == 0, succeeds, result.stderr)
                if succeeds:
                    self.assertEqual((Path(directory) / 'datasets' / path).read_bytes(), b'fixture')
                else:
                    self.assertIn('Unsafe ZIP member', result.stderr)

    def test_adapter_allowlist_and_no_code_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config').mkdir()
            relative = 'models/adapters/test/adapter_model.safetensors'
            (root / 'config/adapter_checksums.json').write_text(json.dumps({relative: hashlib.sha256(b'fixture').hexdigest()}))
            (root / 'run.py').write_text('preserve this code')
            content = io.BytesIO()
            with tarfile.open(fileobj=content, mode='w') as archive:
                for name, data in [('repro_package/' + relative, b'fixture'), ('repro_package/run.py', b'old code')]:
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
            result = self.fixture(directory, 'weights', content.getvalue(), 'repro_package.tar.part000')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / relative).read_bytes(), b'fixture')
            self.assertEqual((root / 'run.py').read_text(), 'preserve this code')

    def test_corrupt_cache_is_not_extracted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # First create a verified ZIP and the generated manifest.
            content = io.BytesIO()
            with zipfile.ZipFile(content, 'w') as archive:
                archive.writestr('sample.txt', b'fixture')
            self.assertEqual(self.fixture(directory, 'datasets', content.getvalue(), 'data.zip').returncode, 0)
            (root / 'datasets/sample.txt').unlink()
            (root / 'downloads/datasets/data.zip').write_bytes(b'corrupt')
            wrapper = "import types,sys\nm=types.ModuleType('gdown')\ndef fail(**kw): raise RuntimeError('network disabled in test')\nm.download=fail\nsys.modules['gdown']=m\n"
            result = subprocess.run([sys.executable, '-c', wrapper + SCRIPT, str(root), 'datasets'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / 'datasets/sample.txt').exists())

    def test_manifest_counts_and_sizes(self):
        manifest = json.loads((ROOT / 'config/download_manifest.json').read_text())
        rows = manifest['files']
        self.assertEqual(len(rows), 32)
        self.assertEqual(len({r['id'] for r in rows}), 32)
        self.assertEqual(sum(r['size'] for r in rows), 1786093167)
        self.assertEqual(sum(r['group'] == 'datasets' for r in rows), 25)
        self.assertEqual(sorted(r['name'] for r in rows if r['group'] == 'weights'), [f'repro_package.tar.part{i:03d}' for i in range(7)])
        for row in rows:
            self.assertRegex(row['sha256'], r'^[0-9a-f]{64}$')


if __name__ == '__main__':
    unittest.main()
