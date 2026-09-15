"""Check final report structure, source tables and local artifact links."""
import hashlib
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parents[1]/'results'

def main():
    path = OUT/'final_evaluation_report.md'
    report = path.read_text(encoding='utf-8')
    assert re.findall(r'^## (\d+)\.', report, flags=re.M) == [str(i) for i in range(1,12)]
    for name in ['final_model_comparison.md', 'b0_vs_halpst_statistics.md', 'inference_efficiency.md']:
        assert (OUT/name).read_text(encoding='utf-8').strip() in report, name
    assert 'Benchmark not yet run' not in report
    assert 'Statistical power is limited with n=5' in report
    assert 'synthetic and only for pipeline testing' in report
    assert 'final-epoch weights (last.pt)' in report
    assert 'Batch 1: B0 mean/median' in report and 'Batch 16: B0 mean/median' in report
    links = re.findall(r'\[[^\]]+\]\(([^)]+)\)', report)
    for link in links:
        if not link.startswith(('http:', 'https:', '#')):
            assert (OUT/link).exists(), link
    result = dict(status='passed', numbered_sections=11, embedded_source_tables=3, local_links_checked=len(links), report_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (OUT/'final_report_validation.json').write_text(json.dumps(result, indent=2))
    print('Final report validated: 11 sections, source tables, timing results, limitations and artifact links.')

if __name__ == '__main__':
    main()
