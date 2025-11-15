import requests
from datetime import datetime, timezone

WEBHOOK_URL = "https://latt3.app.n8n.cloud/webhook-test/drone-detection"

def send_detection(confidence: float, location: str) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "confidence 💀": confidence,
        "location": location,
    }
    try:
        resp = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {resp.text}")
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    # Example detection payload
    send_detection(0.95, "backyard")