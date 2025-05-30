import psycopg2
from datetime import datetime

def log_detection(result):
    conn = psycopg2.connect(dbname="defectdb", user="admin", password="admin", host="localhost")
    cur = conn.cursor()
    cur.execute("INSERT INTO detections (timestamp, class, score) VALUES (%s, %s, %s)",
                (datetime.now(), result['class'], result['score']))
    conn.commit()
    cur.close()
    conn.close()
