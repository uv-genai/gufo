#!/usr/bin/env python3
"""pp2048/tg128 depth series. Usage: depth-series.py <label> [max_depth]"""
import json, time, urllib.request, sys

URL = 'http://127.0.0.1:18200/v1/chat/completions'
if len(sys.argv) > 3:
    URL = sys.argv[3]
LABEL = sys.argv[1] if len(sys.argv) > 1 else 'run'
MAXD = int(sys.argv[2]) if len(sys.argv) > 2 else 131072

def fill(n, kind, tag):
    if kind == 'repetitive':
        unit = 'The quick brown fox jumps over the lazy dog. '
    else:
        unit = (f'Item {tag:04d}: the committee reviewed quarterly results and '
                f'discussed the roadmap for the remaining fiscal year. ')
    return (unit * ((n * 4) // len(unit) + 1))[:n * 4]

def ask(prompt, gen, timeout=600):
    body = {'model': 'Qwen3.8 Flash Next',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': gen, 'temperature': 0,
            'chat_template_kwargs': {'enable_thinking': False}}
    t0 = time.perf_counter()
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    u, g = d['usage'], d['usage'].get('gufo', {})
    return {'wall': time.perf_counter()-t0, 'prompt': u.get('prompt_tokens',0),
            'gen': u.get('completion_tokens',0),
            'pp': u.get('prompt_tokens_per_second') or 0,
            'tg': u.get('completion_tokens_per_second') or 0,
            'ttft': g.get('ttft_ms') or 0,
            'draft': u.get('draft_tokens') or 0,
            'acc': u.get('draft_tokens_accepted') or 0}

print(f'# {LABEL} depth series  pp2048/tg128  greedy MTP')
print(f'# URL {URL}')
print(f'{"depth":>8} {"kind":>11} {"prompt":>7} {"pp":>8} {"tg":>7} {"ttft":>9} {"draft":>7} {"wall":>6}')
rows = []
for depth in (0, 4096, 8192, 16384, 32768, 65536, 131072):
    if depth > MAXD:
        break
    for kind in ('mixed', 'repetitive'):
        try:
            r = ask(fill(depth + 2048, kind, depth), 128)
        except Exception as e:
            print(f'{depth:>8} {kind:>11} FAIL {str(e)[:50]}')
            continue
        acc = r['acc']/r['draft'] if r['draft'] else 0
        print(f'{depth:>8} {kind:>11} {r["prompt"]:>7} {r["pp"]:>8.1f} {r["tg"]:>7.1f} '
              f'{r["ttft"]:>9.1f} {r["draft"]:>3}/{r["acc"]:<3} {r["wall"]:>6.2f}  acc={acc:.0%}')
        rows.append((depth, kind, r))
print('done', len(rows), 'points')
