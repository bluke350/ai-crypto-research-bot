import sqlite3
con=sqlite3.connect('experiments/registry.db')
cur=con.cursor()
print('Latest artifacts rows related to rl_manual_XBT.pth:')
for r in cur.execute("SELECT id, run_id, path, kind FROM artifacts WHERE path LIKE '%rl_manual_XBT.pth%' ORDER BY rowid DESC"):
    print(r)
con.close()
