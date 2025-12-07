const chatBox = document.getElementById("chat-box");
const inputBox = document.getElementById("user-input");

/* SCROLL TO BOTTOM SMOOTHLY */
function scrollChat() {
    setTimeout(() => {
        chatBox.scrollTop = chatBox.scrollHeight;
    }, 30);
}

/* ADD MESSAGE */
function appendMessage(text, sender = "assistant") {
    const div = document.createElement("div");
    div.className = "message " + sender;
    div.innerHTML = text.replace(/\n/g, "<br>");
    chatBox.appendChild(div);
    scrollChat();
}

/* TYPING... */
function appendTyping() {
    const div = document.createElement("div");
    div.id = "typing";
    div.className = "message assistant typing";
    div.innerHTML = "Typing...";
    chatBox.appendChild(div);
    scrollChat();
}

function removeTyping() {
    const t = document.getElementById("typing");
    if (t) t.remove();
}

/* MAIN SEND LOGIC */
async function sendMessage() {
    const text = inputBox.value.trim();
    if (!text) return;

    appendMessage(text, "user");
    inputBox.value = "";

    appendTyping();

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ text })
        });

        const data = await res.json();
        removeTyping();

        if (data.error) {
            appendMessage("❌ Error: " + data.error);
            return;
        }

        const reply = `
<b>Category:</b> ${data.category}
<b>Confidence:</b> ${(data.confidence * 100).toFixed(1)}%
<b>Model Used:</b> ${data.model_used || "Unknown"}
<b>Latency:</b> ${data.prediction_time_ms?.toFixed(1) || "?"} ms
<b>Why This Model?</b> ${data.reasoning || "N/A"}
        `;

        appendMessage(reply, "assistant");

    } catch (err) {
        removeTyping();
        appendMessage("❌ Network Error: " + err);
    }
}

inputBox.addEventListener("keypress", e => {
    if (e.key === "Enter") sendMessage();
});
