#!/usr/bin/env python3
"""Concurrent decode throughput. Usage: conc-bench.py <label> <url> <users...>"""
import json, time, urllib.request, concurrent.futures, sys

LABEL = sys.argv[1]
URL = sys.argv[2]
USERS = [int(x) for x in sys.argv[3:]] or [1, 2, 4]

def fill(n, tag):
    unit = (f'Note {tag}: the committee reviewed the quarterly results and '
            f'discussed the hiring roadmap for the remaining fiscal year. ')
    return (unit * ((n * 4) // len(unit) + 1))[:n * 4]

def ask(prompt, gen=128, timeout=180):
    body = {'model': 'Qwen3.8 Flash Next',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': gen, 'temperature': 0,
            'chat_template_kwargs': {'enable_thinking': False}}
    t0 = time.perf_counter()
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        u = d['usage']
        return {'ok': True, 'wall': time.perf_counter()-t0,
                'pp': u.get('prompt_tokens_per_second') or 0,
                'tg': u.get('completion_tokens_per_second') or 0,
                'prompt': u.get('prompt_tokens', 0),
                'gen': u.get('completion_tokens', 0),
                'draft': u.get('draft_tokens') or 0,
                'acc': u.get('draft_tokens_accepted') or 0}
    except Exception as e:
        return {'ok': False, 'wall': time.perf_counter()-t0, 'err': str(e)[:60]}

print(f'# {LABEL} concurrent  prompt~2K tg128  URL {URL}')
print(f'{"users":>6} {"sum_tg":>8} {"wall":>7} {"ok":>5} {"sum_gen":>8} {"per-user tg":>20}')
for users in USERS:
    # one warmup not needed; fire all together
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=users) as ex:
        futs = [ex.submit(ask, fill(2048, i+1), 128) for i in range(users)]
        rs = [f.result() for f in futs]
    wall = time.perf_counter() - t0
    ok = [r for r in rs if r['ok']]
    sum_tg = sum(r['tg'] for r in ok)
    sum_gen = sum(r['gen'] for r in ok)
    per = ', '.join(f"{r['tg']:.1f}" for r in ok)
    print(f'{users:>6} {sum_tg:>8.1f} {wall:>7.2f} {len(ok):>2}/{users} {sum_gen:>8}  [{per}]')
    for r in rs:
        if not r['ok']:
            print(f'       FAIL {r["err"]}')
print('done')
