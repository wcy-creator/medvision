import sqlite3, time, os

class MemorySystem:
    def __init__(self, db_path="/opt/medvision/data/memory.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS episodic (
            id INTEGER PRIMARY KEY, content TEXT, tags TEXT,
            importance REAL, timestamp REAL, expires_at REAL)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS semantic (
            id INTEGER PRIMARY KEY, content TEXT UNIQUE, tags TEXT,
            importance REAL, timestamp REAL)""")
        self.db.commit()

    def store_episodic(self, content, tags="", importance=0.5):
        ts = time.time()
        self.db.execute("INSERT INTO episodic (content, tags, importance, timestamp, expires_at) VALUES (?, ?, ?, ?, ?)",
            (content, tags, importance, ts, ts + 14*86400))
        self.db.commit()

    def store_semantic(self, content, tags="", importance=0.8):
        ts = time.time()
        try:
            self.db.execute("INSERT INTO semantic (content, tags, importance, timestamp) VALUES (?, ?, ?, ?)",
                (content, tags, importance, ts))
            self.db.commit()
        except sqlite3.IntegrityError:
            self.db.execute("UPDATE semantic SET importance=?, timestamp=? WHERE content=?",
                (importance, ts, content))
            self.db.commit()

    def recall(self, limit=4):
        ep = self.db.execute("SELECT content FROM episodic WHERE expires_at > ? ORDER BY timestamp DESC LIMIT ?",
            (time.time(), limit)).fetchall()
        sm = self.db.execute("SELECT content FROM semantic ORDER BY importance DESC LIMIT ?", (limit,)).fetchall()
        return {"episodic": [r[0] for r in ep], "semantic": [r[0] for r in sm]}

    def prune(self):
        self.db.execute("DELETE FROM episodic WHERE expires_at < ?", (time.time(),))
        self.db.commit()
