import asyncio
import websockets
import json
import hmac
import hashlib
from datetime import datetime

api_data_path = '/Users/samuelbodansky/deribit_api_data.json'
with open(api_data_path, 'r') as f:
    api_data = json.load(f)

clientId = api_data['clientId']
clientSecret = api_data['clientSecret']
timestamp = round(datetime.now().timestamp() * 1000)
nonce = "abcd"
data = ""
signature = hmac.new(
    bytes(clientSecret, "latin-1"),
    msg=bytes('{}\n{}\n{}'.format(timestamp, nonce, data), "latin-1"),
    digestmod=hashlib.sha256
).hexdigest().lower()

msg = {
    "jsonrpc": "2.0",
    "id": 8748,
    "method": "public/auth",
    "params": {
        "grant_type": "client_signature",
        "client_id": clientId,
        "timestamp": timestamp,
        "signature": signature,
        "nonce": nonce,
        "data": data
    }
}

print(msg)

async def call_api(msg):
    async with websockets.connect('wss://test.deribit.com/ws/api/v2') as websocket:
        await websocket.send(msg)
        while websocket.open:
            response = await websocket.recv()
            print(response)

asyncio.get_event_loop().run_until_complete(call_api(json.dumps(msg)))

https://test.deribit.com/api/v2/private/buy?ins trument_name=BTC-PERPETUAL&amount=1&type=limit&label=test_trade

