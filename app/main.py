import sqlite3
conn = sqlite3.connect('tmdb.db')
print(conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall())
print(conn.execute("SELECT COUNT(*) FROM movies").fetchone())
print(conn.execute("SELECT COUNT(*) FROM credits").fetchone())
conn.close()