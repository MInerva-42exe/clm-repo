import os
from redis import Redis
from rq import Worker, Queue
from dotenv import load_dotenv

# Import the function that does the slow work from your main app
from app import call_generative_model

load_dotenv()

REDIS_URL = os.environ.get('REDIS_URL')
if not REDIS_URL:
    raise RuntimeError("REDIS_URL is not configured in the environment.")

listen = ['default']
conn = Redis.from_url(REDIS_URL)

if __name__ == '__main__':
    queues = [Queue(name, connection=conn) for name in listen]
    worker = Worker(queues, connection=conn)
    print("Background worker started...")
    worker.work()
