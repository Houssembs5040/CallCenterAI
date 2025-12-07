import requests
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

BACKEND_URL = "http://localhost:6000/predict"

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Ticket Classification Chat</title>

<style>
    body {
        font-family: Arial, sans-serif;
        background: #f5f5f5;
        color: #333;
        margin: 0;
        height: 100vh;
        display: flex;
        flex-direction: column;
    }

    .header {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #4a90e2;
        padding: 15px 20px;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
    }

    .header img {
        width: 32px;
        height: 32px;
    }

    .chat-container {
        max-width: 800px;
        margin: 20px auto;
        display: flex;
        flex-direction: column;
        flex: 1;
        transition: opacity 0.5s;
        opacity: 0;
    }

    .chat-container.active {
        opacity: 1;
    }

    .chat-box {
        flex: 1;
        overflow-y: auto;
        padding: 20px;
        border-radius: 10px;
        background: #ffffff;
        display: flex;
        flex-direction: column;
        gap: 10px;
        border: 1px solid #ccc;
    }

    .input-area {
        display: flex;
        margin-top: 10px;
    }

    input[type=text] {
        flex: 1;
        padding: 10px;
        border-radius: 5px 0 0 5px;
        border: 1px solid #ccc;
        font-size: 16px;
    }

    button {
        padding: 0 20px;
        border: none;
        border-radius: 0 5px 5px 0;
        background: #4a90e2;
        color: white;
        cursor: pointer;
        font-size: 16px;
    }

    button:hover {
        background: #357ab8;
    }

    .message {
        max-width: 70%;
        padding: 10px 15px;
        border-radius: 15px;
        word-wrap: break-word;
    }

    .user {
        background: #4a90e2;
        color: white;
        align-self: flex-end;
        border-bottom-right-radius: 0;
    }

    .assistant {
        background: #e1e4e8;
        color: #333;
        align-self: flex-start;
        border-bottom-left-radius: 0;
    }

    .meta {
        font-size: 12px;
        color: #999;
        margin-top: 5px;
    }

    .welcome-screen {
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }

    .welcome-screen h1 {
        font-size: 2rem;
        margin-bottom: 20px;
        color: #4a90e2;
    }

    .welcome-screen input {
        max-width: 400px;
        width: 80%;
        padding: 15px;
        font-size: 16px;
        border-radius: 5px;
        border: 1px solid #ccc;
    }

    .welcome-screen button {
        margin-top: 10px;
        width: 100px;
        padding: 10px 0;
        border-radius: 5px;
    }

    .footer {
        background: #4a90e2;
        color: #fff;
        text-align: center;
        padding: 10px 0;
        font-size: 12px;
    }

    .chat-box::-webkit-scrollbar {
        width: 8px;
    }

    .chat-box::-webkit-scrollbar-thumb {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 4px;
    }

    .chat-box::-webkit-scrollbar-track {
        background-color: rgba(0, 0, 0, 0.05);
        border-radius: 4px;
    }
</style>
</head>

<body>

<div class="header">
    <img src="https://img.icons8.com/ios-filled/50/ticket.png" alt="Ticket Icon"/>
    Ticket Classification
</div>

<div class="welcome-screen" id="welcome-screen">
    <h1>Welcome to Ticket Classification</h1>
    <input type="text" id="welcome-input"
        placeholder="Type your message..."
        autocomplete="off"/>
    <button onclick="sendFirstMessage()">Send</button>
</div>

<div class="chat-container" id="chat-container">
    <div class="chat-box" id="chat-box"></div>

    <div class="input-area">
        <input type="text" id="user-input"
            placeholder="Type a message..."
            autocomplete="off"/>
        <button onclick="sendMessage()">Send</button>
    </div>
</div>

<div class="footer">
    Made by Houssem Bensalah & Khalaf Nakbi &copy; 2025
</div>

<script>
const chatBox = document.getElementById("chat-box");
const chatContainer = document.getElementById("chat-container");
const inputBox = document.getElementById("user-input");
const welcomeScreen = document.getElementById("welcome-screen");
const welcomeInput = document.getElementById("welcome-input");

function appendMessage(text, sender = "assistant") {
    const msg = document.createElement("div");
    msg.className = "message " + sender;
    msg.innerHTML = text;
    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage(textOverride = null) {
    const text = textOverride || inputBox.value.trim();
    if (!text) return;

    appendMessage(text, "user");
    if (!textOverride) inputBox.value = "";

    const typingMsg = document.createElement("div");
    typingMsg.className = "message assistant";
    typingMsg.innerHTML = "Typing...";
    chatBox.appendChild(typingMsg);
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        const data = await res.json();
        typingMsg.remove();

        if (data.error) {
            appendMessage("Error: " + data.error, "assistant");
            return;
        }

        let reply = "<b>Category:</b> " + data.category +
                    "<br><b>Confidence:</b> " +
                    (data.confidence * 100).toFixed(1) + "%";

        if (data.reasoning) {
            reply += "<br><b>Reasoning:</b> " + data.reasoning;
        }

        appendMessage(reply, "assistant");

    } catch (err) {
        typingMsg.remove();
        appendMessage("Error: " + err, "assistant");
    }
}

function sendFirstMessage() {
    const text = welcomeInput.value.trim();
    if (!text) return;

    welcomeScreen.style.display = "none";
    chatContainer.classList.add("active");
    inputBox.focus();
    sendMessage(text);
}

inputBox.addEventListener("keypress", function (e) {
    if (e.key === "Enter") sendMessage();
});

welcomeInput.addEventListener("keypress", function (e) {
    if (e.key === "Enter") sendFirstMessage();
});
</script>

</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.json.get("text", "")
    try:
        response = requests.post(BACKEND_URL, json={"text": user_input})
        response.raise_for_status()
        return jsonify(response.json())
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=3020, debug=True)
