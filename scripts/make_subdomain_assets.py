"""Offline exploratory audit of hostname multiplicity and outcome-dependent DNS sampling.

First run: --freeze copies input bytes; later runs read only those copies.
No survival inference, live DNS, or contemporary blocklist fetch is performed.
"""
from __future__ import annotations
import argparse
import ast
import gzip
import hashlib
import io
import ipaddress
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import tldextract
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'data/processed/subdomain'
PAPER = ROOT / 'papers/future_subdomain_evasion'
INPUTS = {
    'infra': 'data/raw/host_infra/host_infra.csv',
    'brands': 'data/raw/urlscan_brands/detections.csv',
    'general': 'data/raw/vn_phishing_live/detections.csv',
    'community': 'data/raw/chongluadao_live/detections.csv',
    'openphish': 'data/raw/openphish/feed.csv',
}
ICANN = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=False)
PRIVATE = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalize(host):
    return str(host).strip().lower().rstrip('.')


def boundary(host, extractor):
    result = extractor(host)
    return result.top_domain_under_public_suffix or host


def tenant_key(host, hosted):
    # PSL private rules plus the study's frozen explicit hosted list.
    key = boundary(host, PRIVATE)
    for suffix in sorted(hosted, key=len, reverse=True):
        if host.endswith('.' + suffix):
            candidate = '.'.join(host.split('.')[-len(suffix.split('.'))-1:])
            if len(candidate.split('.')) > len(key.split('.')):
                key = candidate
            break
    return key


def public_a(value):
    for token in str(value).split(';'):
        try:
            addr = ipaddress.ip_address(token.strip())
            if addr.version == 4 and addr.is_global:
                return True
        except ValueError:
            pass
    return False


def metrics(hosts, key):
    n = hosts.groupby(key).host.nunique()
    return {'hosts': int(hosts.host.nunique()), 'groups': int(len(n)),
            'multi_groups': int((n >= 2).sum()), 'multi_pct': float(100*(n >= 2).mean()),
            'inflation': float(n.sum()/len(n)), 'four_groups': int((n == 4).sum()),
            'max_hosts': int(n.max()),
            'one_seed_host_weighted_gain_pp': float(100*(1-len(n)/n.sum())),
            'one_seed_group_mean_gain_pp': float(100*(1-1/n).mean())}


def tex_escape(text):
    return str(text).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def table(name, caption, headers, rows):
    lines = [r'\begin{table}[htbp]\centering\small', r'\caption{'+caption+'}',
             r'\label{tab:'+name+'}', r'\begin{tabular}{'+'l'+'r'*(len(headers)-1)+'}',
             r'\toprule', ' & '.join(headers)+r' \\', r'\midrule']
    lines += [' & '.join(map(str, row))+r' \\' for row in rows]
    lines += [r'\bottomrule\end{tabular}', r'\end{table}']
    (PAPER/'sections'/('tab_'+name+'.tex')).write_text('\n'.join(lines)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = OUT/'inputs'
    manifest_path = OUT/'input_manifest.json'
    if args.freeze:
        if manifest_path.exists():
            raise SystemExit('Freeze already exists; preserve it and use a separate output for a new study.')
        frozen.mkdir(exist_ok=True)
        hosted_path = ROOT/'scripts/audit_capture_labels.py'
        module = ast.parse(hosted_path.read_text())
        hosted = next(ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'HOSTED_SUFFIXES' for t in n.targets))
        suffix_bytes = Path(tldextract.__file__).with_name('.tld_set_snapshot').read_bytes()
        (frozen/'suffix_snapshot.dat.gz').write_bytes(gzip.compress(suffix_bytes, mtime=0))
        manifest = {'parent_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    'status': 'exploratory implementation audit; not a preregistered code freeze',
                    'hosted_suffixes': list(hosted), 'hosted_module_sha256': sha(hosted_path.read_bytes()),
                    'suffix_snapshot_sha256': sha(suffix_bytes),
                    'tldextract_version': tldextract.__version__, 'inputs': {}}
        for name, rel in INPUTS.items():
            data = (ROOT/rel).read_bytes()
            (frozen/(name+'.csv.gz')).write_bytes(gzip.compress(data, mtime=0))
            manifest['inputs'][name] = {'source': rel, 'sha256': sha(data), 'bytes': len(data)}
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    manifest = json.loads(manifest_path.read_text())
    suffix_bytes = Path(tldextract.__file__).with_name('.tld_set_snapshot').read_bytes()
    if sha(suffix_bytes) != manifest['suffix_snapshot_sha256']:
        raise SystemExit('Installed PSL snapshot differs from frozen parser; restore recorded tldextract version.')
    datasets = {}
    for name, meta in manifest['inputs'].items():
        data = gzip.decompress((frozen/(name+'.csv.gz')).read_bytes())
        assert sha(data) == meta['sha256'], name
        datasets[name] = pd.read_csv(io.BytesIO(data), keep_default_na=False)
    d = datasets['infra'].copy()
    d['host'] = d.domain.map(normalize)
    d['icann'] = d.host.map(lambda h: boundary(h, ICANN))
    d['tenant'] = d.host.map(lambda h: tenant_key(h, manifest['hosted_suffixes']))
    d['hosted'] = d.host.map(lambda h: any(h == s or h.endswith('.'+s) for s in manifest['hosted_suffixes']))
    d['public_a'] = d.a_records.map(public_a)
    d['captured'] = pd.to_datetime(d.captured_at, format='mixed', utc=True, errors='coerce')
    d['detected'] = pd.to_datetime(d.first_detected, format='mixed', utc=True, errors='coerce')
    d['lag_h'] = (d.captured-d.detected).dt.total_seconds()/3600
    d['attempt'] = pd.to_numeric(d.attempt)
    ph = d[d.label == 'phish'].copy()
    first = ph.sort_values(['captured','attempt']).drop_duplicates('host').copy()
    first['later_rows'] = first.host.map(ph.groupby('host').size())-1
    hosts = first[['host','icann','tenant','source','hosted']].copy()
    collapsed = hosts.copy()
    collapsed['host'] = collapsed.host.str.replace(r'^www\.', '', regex=True)
    summary = {'rows': len(d), 'positive_rows': len(ph), 'capture_min': d.captured_at.min(),
               'capture_max': d.captured_at.max(), 'first_positive_hosts': len(first),
               'stored_key_disagreement_rows': int((d.registered_domain != d.tenant).sum()),
               'stored_key_disagreement_positive_rows': int((ph.registered_domain != ph.tenant).sum()),
               'invalid_or_negative_lag_rows': int((ph.lag_h.isna() | (ph.lag_h<0)).sum()),
               'units': {}, 'sources': [], 'retry': [], 'lag_bins': []}
    for name, frame, key in [('ICANN',hosts,'icann'),('tenant',hosts,'tenant'),
                              ('tenant_www',collapsed,'tenant')]:
        summary['units'][name] = metrics(frame,key)
    for source, sub in hosts.groupby('source'):
        summary['sources'].append({'source': source, **metrics(sub,'tenant'),
                                   'first_public_a_pct': float(first[first.source==source].public_a.mean()*100)})
    for resolved, sub in first.groupby('public_a'):
        summary['retry'].append({'first_public_a': bool(resolved), 'hosts':len(sub),
                                 'retried':int((sub.later_rows>0).sum()),
                                 'retry_pct':float(100*(sub.later_rows>0).mean())})
    multiplicity = hosts.groupby('tenant').host.nunique()
    first['multi'] = first.tenant.map(multiplicity)>=2
    for lo,hi in [(0,8),(8,32),(32,64),(64,128),(128,512),(512,float('inf'))]:
        row={'bin':f'{lo}--{hi:g}', 'all_rows':int(ph.lag_h.between(lo,hi,inclusive='left').sum())}
        sub=first[first.lag_h.between(lo,hi,inclusive='left')]
        row.update(first_hosts=len(sub), first_public_a_pct=float(100*sub.public_a.mean()) if len(sub) else None,
                   later_rows=row['all_rows']-len(sub))
        summary['lag_bins'].append(row)
    support=[]
    for lo,hi in [(32,64),(64,128)]:
        sub=first[first.lag_h.between(lo,hi,inclusive='left')]
        for (multi, hosted), group in sub.groupby(['multi','hosted']):
            support.append({'bin':f'{lo}--{hi}', 'multi':bool(multi), 'hosted':bool(hosted),
                            'hosts':len(group), 'groups':group.tenant.nunique(),
                            'public_a_hosts':int(group.public_a.sum())})
    summary['registered_window_support']=support
    # Auditing four-host groups is descriptive; source provisioning cannot be inferred.
    four_keys = multiplicity[multiplicity == 4].index
    four = hosts[hosts.tenant.isin(four_keys)]
    summary['four_tenant_groups'] = {'groups':len(four_keys),
        'with_www':int(four.groupby('tenant').host.apply(lambda x:x.str.startswith('www.').any()).sum()),
        'collapse_below_four':int((collapsed[collapsed.tenant.isin(four_keys)].groupby('tenant').host.nunique()<4).sum())}
    # Presence in upstream files is a lexical provenance check, not independent validation.
    upstream=set()
    for name in ['brands','general','community']:
        upstream |= set(datasets[name].domain.map(normalize))
    summary['hosts_absent_upstream_union'] = len(set(hosts.host)-upstream)
    op=datasets['openphish']
    summary['blocklist_reference']={'rows':len(op),'first':op.scraped_at.min(),'last':op.scraped_at.max(),
                                    'phishtank_available':False,'rq3_status':'not evaluated'}
    summary['rq_status']={'RQ1':'not evaluated: outcome-dependent revisit policy',
                          'RQ2':'not evaluated: same selection and no suspension observations',
                          'RQ3':'not evaluated: no contemporaneous paired blocklist snapshot or traffic weights'}
    summary['analysis_sha256']=sha(Path(__file__).read_bytes())
    summary['python_version']=sys.version.split()[0]
    summary['pandas_version']=pd.__version__
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    hosts.to_csv(OUT/'hostname_units.csv',index=False)
    first[['host','tenant','source','hosted','multi','lag_h','public_a','later_rows']].to_csv(OUT/'first_capture.csv',index=False)
    for key in ['sources','retry','lag_bins','registered_window_support']:
        pd.DataFrame(summary[key]).to_csv(OUT/(key+'.csv'),index=False)
    table('units','Multiplicity depends on the grouping and hostname convention.',
          ['Key','Hosts','Groups','Multi (\\%)','$H/G$'],
          [[tex_escape(k),f"{v['hosts']:,}",f"{v['groups']:,}",f"{v['multi_pct']:.1f}",f"{v['inflation']:.2f}"] for k,v in summary['units'].items()])
    table('sources','Source attribution follows the first retained capture; rows are not independent feed samples.',
          ['Source','Hosts','Groups','Multi (\\%)','Public A (\\%)'],
          [[tex_escape(x['source']),f"{x['hosts']:,}",f"{x['groups']:,}",f"{x['multi_pct']:.1f}",f"{x['first_public_a_pct']:.1f}"] for x in summary['sources']])
    table('retry','Later observations are selected by the first observed DNS outcome.',
          ['First public A','Hosts','Retried','Retried (\\%)'],
          [['Yes' if x['first_public_a'] else 'No',f"{x['hosts']:,}",f"{x['retried']:,}",f"{x['retry_pct']:.1f}"] for x in summary['retry']])
    table('lag','Observation support by capture lag. First observations form different arrival cohorts; later rows are retries.',
          ['Lag (h)','All rows','First hosts','Retry rows'],
          [[x['bin'].replace('inf',r'$\infty$'),f"{x['all_rows']:,}",f"{x['first_hosts']:,}",f"{x['later_rows']:,}"] for x in summary['lag_bins']])
    table('scope','Algebraic added hostname scope if exactly one hostname per group is seeded. Not measured protection.',
          ['Key','Host-weighted (pp)','Group mean (pp)'],
          [[tex_escape(k),f"{v['one_seed_host_weighted_gain_pp']:.1f}",f"{v['one_seed_group_mean_gain_pp']:.1f}"] for k,v in summary['units'].items()])
    macros={'SubRows':f'{len(d):,}','SubPositiveRows':f'{len(ph):,}', 'SubHosts':f'{len(hosts):,}',
            'SubGroups':f"{summary['units']['tenant']['groups']:,}",
            'SubMultiPct':f"{summary['units']['tenant']['multi_pct']:.1f}",
            'SubInflation':f"{summary['units']['tenant']['inflation']:.2f}",
            'SubCollapsedInflation':f"{summary['units']['tenant_www']['inflation']:.2f}",
            'SubChangedRows':f"{summary['stored_key_disagreement_positive_rows']:,}",
            'SubFour':f'{len(four_keys):,}', 'SubFourCollapsed':str(summary['four_tenant_groups']['collapse_below_four']),
            'SubMissingUpstream':str(summary['hosts_absent_upstream_union']),
            'SubScopeGain':f"{summary['units']['tenant']['one_seed_host_weighted_gain_pp']:.1f}"}
    (PAPER/'sections/gen_summary.tex').write_text('% Generated; do not edit.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(6.5,3.3))
    for frame,key,label in [(hosts,'icann','ICANN grouping'),(hosts,'tenant','Tenant grouping'),(collapsed,'tenant','Tenant, www collapsed')]:
        sizes=frame.groupby(key).host.nunique().sort_values().to_numpy()
        ax.step(sizes,np.arange(1,len(sizes)+1)/len(sizes),where='post',label=label)
    ax.set(xscale='log',xlabel='Distinct hostnames per group (log scale)',ylabel='Cumulative fraction of groups',ylim=(0,1.03))
    ax.legend(loc='lower right',fontsize=8);fig.tight_layout();fig.savefig(PAPER/'figures/multiplicity.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,3.3));l=summary['lag_bins'];x=np.arange(len(l))
    ax.bar(x,[r['first_hosts'] for r in l],label='First observed hosts',color='#31688e')
    ax.bar(x,[r['later_rows'] for r in l],bottom=[r['first_hosts'] for r in l],label='Later rows',color='#d89033')
    ax.set_xticks(x,[r['bin'] for r in l]);ax.set(xlabel='Hours since recorded detection',ylabel='Capture records');ax.legend();fig.tight_layout();fig.savefig(PAPER/'figures/observation_support.pdf');plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
