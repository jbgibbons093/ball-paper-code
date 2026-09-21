"""Keep empirical source publishable while blocking protected artifacts."""
import hashlib
import json
from pathlib import Path

import pytest
from validation.check_repository_privacy import _blocked_path


@pytest.mark.parametrize('name', [
    'empirical/prospective_classical.py', 'empirical/refine_intervals.py',
    'validation/temporal/refit_temporal.py',
])
def test_analysis_source_is_allowed(name):
    assert _blocked_path(name) is None


@pytest.mark.parametrize('name', [
    'empirical/patients.csv', 'empirical/patients.json',
    'empirical/derived/example.py', 'empirical/nested/helper.py',
    'empirical/credentials.py', 'raw/example.py', 'outputs/results.json',
    'validation/checkpoint.pt', 'clinical.csv.gz', 'Article.docx',
])
def test_patient_data_and_generated_artifacts_are_blocked(name):
    assert _blocked_path(name) is not None


def test_published_sources_match_finalized_analysis():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'validation/finalized_analysis_sources.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
