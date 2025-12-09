import sqlite3
import json
con=sqlite3.connect('experiments/registry.db')
cur=con.cursor()
print('Tables:')
for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    print(row[0])

def print_table_info(tbl):
    print(f"\nSchema for {tbl}:")
    for r in cur.execute(f'PRAGMA table_info({tbl})'):
        print(r)

for tbl in [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]:
    print_table_info(tbl)

print('\nSample rows (last 10) per table:')
for tbl in [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]:
    try:
        print(f"\n-- {tbl} --")
        for r in cur.execute(f"SELECT * FROM {tbl} ORDER BY rowid DESC LIMIT 10"):
            print(r)
    except Exception as e:
        print('failed to query', tbl, e)

con.close()
