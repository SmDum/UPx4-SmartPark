from fastapi import FastAPI, WebSocket
from contextlib import asynccontextmanager
import asyncio
import redis.asyncio as redis
import uvicorn

"""
allow pasting

const ws = new WebSocket("ws://localhost:8000/ws");
ws.onmessage = e => console.log("WS recebeu:", e.data);
ws.onopen = () => console.log("WS aberto");

Teste do Dev tools
"""



REDIS_URL = "redis://:N9PEdtmSCkjUkJrCG9uYKnlgzMuUnCDs@redis-15479.crce196.sa-east-1-2.ec2.redns.redis-cloud.com:15479/0"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    print("Redis conectado")
    yield
    await app.state.redis.close()
    

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    redis_client = app.state.redis
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("vagas:update")

    async def reader():
        async for message in pubsub.listen():
            if message["type"] == "message":
                await ws.send_text(message["data"])

    reader_task = asyncio.create_task(reader())

    try:
        while True:
            data = await ws.receive_text()
            await redis_client.publish("vagas:update", data)
    except Exception:
        pass
    finally:
        reader_task.cancel()
        await pubsub.close()
        await ws.close()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)