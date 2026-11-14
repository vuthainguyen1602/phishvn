"""Independently check saved subdomain audit outputs using the standard library."""
import csv
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'data/processed/subdomain'


def require(condition, message):
    if not condition:
        raise SystemExit('FAIL: ' + message)


def rows(name):
    with (OUT / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    manifest = json.loads((OUT / 'input_manifest.json').read_text())
    for name, meta in manifest['inputs'].items():
        data = gzip.decompress((OUT / 'inputs' / (name + '.csv.gz')).read_bytes())
        require(len(data) == meta['bytes'] and hashlib.sha256(data).hexdigest() == meta['sha256'],
                'frozen input changed: ' + name)
    script = Path(__file__).with_name('make_subdomain_assets.py')
    require(hashlib.sha256(script.read_bytes()).hexdigest() == summary['analysis_sha256'],
            'analysis changed since outputs were generated')
    hosts = rows('hostname_units.csv')
    first = rows('first_capture.csv')
    require(len(hosts) == len({r['host'] for r in hosts}) == summary['first_positive_hosts'],
            'unique hostname inventory')
    require(len(first) == len(hosts) and {r['host'] for r in first} == {r['host'] for r in hosts},
            'first-capture hostname inventory')
    require(sum(1 + int(r['later_rows']) for r in first) == summary['positive_rows'],
            'candidate capture count')
    for convention, key in [('ICANN', 'icann'), ('tenant', 'tenant'), ('tenant_www', 'tenant')]:
        groups = defaultdict(set)
        for row in hosts:
            host = row['host'].removeprefix('www.') if convention == 'tenant_www' else row['host']
            groups[row[key]].add(host)
        sizes = [len(value) for value in groups.values()]
        h, g = sum(sizes), len(sizes)
        expected = {'hosts': h, 'groups': g, 'multi_groups': sum(n >= 2 for n in sizes),
                    'four_groups': sizes.count(4), 'max_hosts': max(sizes),
                    'multi_pct': 100 * sum(n >= 2 for n in sizes) / g,
                    'inflation': h / g, 'one_seed_host_weighted_gain_pp': 100 * (1 - g / h),
                    'one_seed_group_mean_gain_pp': 100 * sum(1 - 1 / n for n in sizes) / g}
        for metric, value in expected.items():
            require(math.isclose(summary['units'][convention][metric], value, rel_tol=1e-12),
                    convention + ': ' + metric)
    for saved in summary['retry']:
        subset = [r for r in first if (r['public_a'] == 'True') == saved['first_public_a']]
        retried = sum(int(r['later_rows']) > 0 for r in subset)
        require(len(subset) == saved['hosts'] and retried == saved['retried'], 'retry counts')
        require(math.isclose(100 * retried / len(subset), saved['retry_pct']), 'retry percentage')
    require(all(int(r['later_rows']) == 0 for r in first if r['public_a'] == 'True'),
            'initially resolving hosts now have follow-up; reconsider manuscript claims')
    require(all(value.startswith('not evaluated:') for value in summary['rq_status'].values()),
            'registered question status')
    macro_text = (ROOT / 'papers/future_subdomain_evasion/sections/gen_summary.tex').read_text()
    macros = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]+)\}', macro_text))
    tenant = summary['units']['tenant']
    expected_macros = {
        'SubRows': f"{summary['rows']:,}",
        'SubPositiveRows': f"{summary['positive_rows']:,}",
        'SubHosts': f"{len(hosts):,}",
        'SubGroups': f"{tenant['groups']:,}",
        'SubMultiPct': f"{tenant['multi_pct']:.1f}",
        'SubInflation': f"{tenant['inflation']:.2f}",
        'SubCollapsedInflation': f"{summary['units']['tenant_www']['inflation']:.2f}",
        'SubChangedRows': f"{summary['stored_key_disagreement_positive_rows']:,}",
        'SubFour': f"{summary['four_tenant_groups']['groups']:,}",
        'SubFourCollapsed': str(summary['four_tenant_groups']['collapse_below_four']),
        'SubMissingUpstream': str(summary['hosts_absent_upstream_union']),
        'SubScopeGain': f"{tenant['one_seed_host_weighted_gain_pp']:.1f}",
    }
    require(macros == expected_macros, 'generated manuscript macros disagree with audit outputs')
    print('PASS: frozen inputs, analysis hash, hostname units, expansion quantities and retry support')


if __name__ == '__main__':
    main()
