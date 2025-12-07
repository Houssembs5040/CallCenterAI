"""Generate traffic for testing monitoring"""
import requests
import time
import random

API_URL = "http://localhost:6000/predict"

test_tickets = [
    "My laptop screen is broken and needs repair",
    "I forgot my password and cannot login",
    "Need to purchase new software licenses",
    "The database storage is full",
    "My email account is locked",
    "Computer won't turn on",
    "Need administrative access to install software",
    "Request for new hardware equipment"
]

def send_request():
    """Send a single request"""
    ticket = random.choice(test_tickets)
    try:
        response = requests.post(
            API_URL,
            json={"text": ticket, "return_probas": True},
            timeout=10
        )
        print(f"✅ {response.status_code} - {ticket[:50]}...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Generating traffic...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            send_request()
            time.sleep(random.uniform(0.5, 2.0))
    except KeyboardInterrupt:
        print("\n✅ Stopped!")