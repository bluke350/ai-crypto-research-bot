import os, json, sqlite3, shutil, time

ROOT = os.path.abspath('.')
PROMOTED_OLD = os.path.join('models','promoted_artifacts_rl_manual_XBT.pth')
CKPT = os.path.join('experiments','artifacts','rl_manual_XBT.pth')
PROMOTIONS = os.path.join('experiments','promotions.json')
DB = os.path.join('experiments','registry.db')
ARTIFACT_DIR = os.path.join('experiments','artifacts')

# find run_id for the RL checkpoint in registry
con = sqlite3.connect(DB)
cur = con.cursor()
# Try exact match first, then fallback to LIKE to tolerate slash direction and absolute vs relative paths
row = None
try:
    cur.execute("SELECT run_id FROM artifacts WHERE path=? LIMIT 1", (CKPT,))
    row = cur.fetchone()
except Exception:
    row = None
if not row:
    # fallback match by filename
    basename = os.path.basename(CKPT)
    cur.execute("SELECT run_id, path FROM artifacts WHERE path LIKE ? ORDER BY rowid DESC LIMIT 1", (f"%{basename}%",))
    r = cur.fetchone()
    if r:
        row = (r[0],)
if row:
    rl_run_id = row[0]
else:
    rl_run_id = None

if not rl_run_id:
    print('Could not find run_id for checkpoint', CKPT)
    # show nearby artifacts to help debugging
    print('Recent artifact rows (last 10):')
    for r in cur.execute("SELECT rowid, run_id, path FROM artifacts ORDER BY rowid DESC LIMIT 10"):
        print(r)
    con.close()
    raise SystemExit(1)

basename = os.path.basename(PROMOTED_OLD)
new_name = f"promoted_{rl_run_id}_{os.path.basename(CKPT)}"
PROMOTED_NEW = os.path.join('models', new_name)

# move/rename file
if os.path.exists(PROMOTED_OLD):
    if os.path.abspath(PROMOTED_OLD) != os.path.abspath(PROMOTED_NEW):
        shutil.move(PROMOTED_OLD, PROMOTED_NEW)
        print('Moved', PROMOTED_OLD, '->', PROMOTED_NEW)
    else:
        print('Promoted file already correct path')
else:
    print('Old promoted file not found, copying checkpoint to new promoted path')
    os.makedirs(os.path.dirname(PROMOTED_NEW), exist_ok=True)
    shutil.copy2(CKPT, PROMOTED_NEW)

# update promotions.json
if os.path.exists(PROMOTIONS):
    with open(PROMOTIONS, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
else:
    data = []

# find the entry we previously added (match promoted_artifacts_rl_manual_XBT.pth)
found = False
for entry in data:
    if entry.get('promoted_path', '').endswith('promoted_artifacts_rl_manual_XBT.pth'):
        entry['promoted_path'] = os.path.abspath(PROMOTED_NEW)
        entry['run_id'] = rl_run_id
        entry['artifact_dir'] = os.path.abspath(ARTIFACT_DIR)
        # enrich metrics
        try:
            sz = os.path.getsize(CKPT)
        except Exception:
            sz = None
        entry['metrics'] = entry.get('metrics', {})
        entry['metrics'].update({'episodes': 2, 'total_steps': 500, 'checkpoint_size_bytes': sz, 'promoted_at_auto': time.strftime('%Y-%m-%dT%H:%M:%S')})
        found = True
        break

if not found:
    # append new entry
    try:
        sz = os.path.getsize(CKPT)
    except Exception:
        sz = None
    data.append({
        'promoted_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'run_id': rl_run_id,
        'artifact_dir': os.path.abspath(ARTIFACT_DIR),
        'promoted_path': os.path.abspath(PROMOTED_NEW),
        'best_params': {},
        'metrics': {'episodes': 2, 'total_steps': 500, 'checkpoint_size_bytes': sz}
    })

with open(PROMOTIONS, 'w', encoding='utf-8') as fh:
    json.dump(data, fh, indent=2)
print('Updated promotions.json')

# update registry DB artifacts row for promoted file
cur = con.cursor()
cur.execute("UPDATE artifacts SET path=? , run_id=? WHERE path LIKE ?", (PROMOTED_NEW, rl_run_id, '%promoted_artifacts_rl_manual_XBT.pth'))
con.commit()
print('Updated registry artifacts table rows:', cur.rowcount)

# write BEST_CANDIDATE.json next to checkpoint (artifact dir)
best = {
    'run_id': rl_run_id,
    'best_score': None,
    'best_params': {},
    'metrics': {'episodes': 2, 'total_steps': 500, 'checkpoint_size_bytes': os.path.getsize(CKPT) if os.path.exists(CKPT) else None, 'source_checkpoint': CKPT},
}
best_path = os.path.join(ARTIFACT_DIR, 'BEST_CANDIDATE.json')
with open(best_path, 'w', encoding='utf-8') as fh:
    json.dump(best, fh, indent=2)
print('Wrote', best_path)

con.close()
print('Done')
