// script.js
document
  .getElementById("generate-btn")
  .addEventListener("click", async function () {
    const bitLength = document.getElementById("bit-length").value;
    const resultDiv = document.getElementById("result");

    if (!bitLength || bitLength <= 0) {
      resultDiv.textContent = "Please enter a valid bit length.";
      return;
    }

    resultDiv.textContent = "Generating...";

    try {
      const response = await fetch("http://0.0.0.0:8000/generate-seed", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ bit_length: parseInt(bitLength) }),
      });

      if (response.ok) {
        const data = await response.json();
        resultDiv.textContent = `Generated Seed: ${data.seed}`;
      } else {
        const errorData = await response.json();
        resultDiv.textContent = `Error: ${errorData.detail}`;
      }
    } catch (error) {
      resultDiv.textContent = `Network Error: ${error.message}`;
    }
  });
