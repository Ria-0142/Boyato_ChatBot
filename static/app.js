document.addEventListener("DOMContentLoaded", () => {
    const sendButton = document.getElementById("run-code");
    const codeInput = document.getElementById("code-input");
    const chatbox = document.querySelector(".chatbox");

    sendButton.addEventListener("click", () => {
        const code = codeInput.value.trim(); // Get user input from text area
        if (code) {
            fetch("/execute", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ code }), // Send user input to server
            })
                .then((response) => response.json())
                .then((data) => {
                    const responseMessage = data.output;
                    const chatMessage = `
          <li class="chat outgoing">
            <span class="material-symbols-outlined">account_circle</span>
            <p>${code}</p>
          </li>
          <li class="chat incomming">
            <span class="material-symbols-outlined">smart_toy</span>
            <p>${responseMessage}</p>
          </li>
        `;
                    chatbox.innerHTML += chatMessage; // Update chatbox with response
                })
                .catch((error) => console.error("Error:", error));
        }
    });
});
