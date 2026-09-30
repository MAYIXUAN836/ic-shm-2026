"""Check frozen interfaces without importing ML libraries or loading models."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.stage1_infer import parse_output
from run import collect_inventory


class PipelineTests(unittest.TestCase):
    def test_recognition_parser(self):
        ok, labels, error = parse_output('{"damage_categories":["Exposed Rebar","crack","crack"]}')
        self.assertEqual((ok, labels, error), (True, ['crack', 'exposed_rebar'], None))
        self.assertEqual(parse_output('{"damage_categories":[]}'), (True, [], None))
        for raw in ('broken', '[]', '{}', '{"damage_categories":"crack"}', '{"damage_categories":["unknown"]}'):
            with self.subTest(raw=raw):
                self.assertFalse(parse_output(raw)[0])

    def test_recursive_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                collect_inventory(root)
            (root / 'sub').mkdir()
            (root / 'sub/b.PNG').touch()
            (root / 'a.jpg').touch()
            (root / 'notes.txt').touch()
            self.assertEqual([r['image_id'] for r in collect_inventory(root)], ['a.jpg', 'sub/b.PNG'])

    def test_routes_and_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write(name, rows):
                path = root / name
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                return str(path)
            def read(name):
                return [json.loads(line) for line in (root / name).read_text().splitlines()]
            def call(script, *args, success=True):
                result = subprocess.run([sys.executable, str(ROOT / 'pipeline' / script), *args], capture_output=True, text=True)
                if success:
                    self.assertEqual(result.returncode, 0, result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
            cases = [[], ['spalling'], ['crack'], ['exposed_rebar'], ['crack', 'corrosion'], ['crack', 'spalling'], ['corrosion', 'exposed_rebar']]
            stage = write('stage.jsonl', [{'image_id': str(i), 'image_path': '/unused/image.jpg', 'predicted_categories': c} for i, c in enumerate(cases)])
            call('prepare_routes.py', '--stage1', stage, '--output-dir', str(root / 'routes'))
            self.assertEqual([r['image_id'] for r in read('routes/expert_b_crack_only.jsonl')], ['2', '5'])
            self.assertEqual([r['image_id'] for r in read('routes/expert_b_with_rebar.jsonl')], ['3', '6'])
            def expert_row(i, cats, text):
                return {'image_id': str(i), 'damage_categories': cats, 'description': text}
            crack = write('crack.jsonl', [expert_row(i, ['crack'], 'v07 crack.') for i in (2, 5)])
            other = write('other.jsonl', [expert_row(i, cases[i], 'v08 other.') for i in (3, 4, 6)])
            rebar = write('rebar.jsonl', [expert_row(i, cases[i], 'v07 rebar.') for i in (3, 6)])
            selected = str(root / 'selected.jsonl')
            call('combine_b_routes.py', '--stage1', stage, '--crack-only', crack, '--other', other, '--rebar-comparison', rebar, '--output', selected)
            b = {r['image_id']: r for r in read('selected.jsonl')}
            self.assertEqual(b['3']['description'], 'v07 rebar.')
            self.assertEqual(b['6']['description'], 'v08 other.')
            a = write('a.jsonl', [expert_row(i, ['spalling'], 'Large-area concrete spalling.') for i in (1, 5)])
            merged = str(root / 'merged.jsonl')
            call('merge.py', '--stage1', stage, '--expert-a', a, '--expert-b', selected, '--output', merged, '--neutral-extent')
            rows = read('merged.jsonl')
            self.assertEqual([r['damage_categories'] for r in rows], cases)
            self.assertEqual(rows[0]['description'], 'No visible structural damage is observed in the image.')
            self.assertEqual(rows[5]['description'], 'Concrete spalling. v07 crack.')
            # An absent required descriptor must fail, never silently omit damage.
            write('a.jsonl', [])
            call('merge.py', '--stage1', stage, '--expert-a', a, '--expert-b', selected, '--output', str(root / 'invalid.jsonl'), success=False)


if __name__ == '__main__':
    unittest.main()
