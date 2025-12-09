import sqlite3
con=sqlite3.connect('experiments/registry.db')
cur=con.cursor()
for r in cur.execute("SELECT rowid, run_id, path FROM artifacts WHERE path LIKE '%rl_manual_XBT%'"):
    print(r)
con.close()
