function requestExplanation() {
    const originalBias = {
        "Disparate Impact": 0.8,  // Replace with actual values
        "Statistical Parity Difference": -0.1
    };
    const combinedBias = {
        "Disparate Impact": 0.9,  // Replace with actual values
        "Statistical Parity Difference": -0.05
    };

    fetch('/explain_bias', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ original_bias: originalBias, combined_bias: combinedBias })
    })
    .then(response => response.json())
    .then(data => {
        if (data.explanation) {
            document.getElementById('biasExplanation').innerText = data.explanation;
        } else {
            document.getElementById('biasExplanation').innerText = "Failed to generate explanation.";
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('biasExplanation').innerText = "An error occurred.";
    });
}
