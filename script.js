document.getElementById('file').addEventListener('change', function(event) {
    const preview = document.getElementById('image-preview');
    preview.innerHTML = ''; // Clear previous content

    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.style.maxWidth = '300px'; // Limit image size for better layout
            img.style.maxHeight = '300px';
            preview.appendChild(img);
        };
        reader.readAsDataURL(file); // Read file as a data URL
    } else {
        preview.innerHTML = '<p>No image selected</p>';
    }
});

document.getElementById('prediction-form').addEventListener('submit', async function (e) {
    e.preventDefault();  // Prevent the default form submission behavior

    const fileInput = document.getElementById('file');
    const predictionDisplay = document.getElementById('prediction');

    // Clear previous prediction result
    predictionDisplay.textContent = '';

    // Validate that a file has been selected
    if (!fileInput.files[0]) {
        alert("Please select a file to upload.");
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);  // Append the uploaded file to the FormData object

    console.log('Submitting file...', fileInput.files[0].name);  // Log the file name for debugging

    try {
        // Send the file to the server
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        // Check if the response status is OK (200)
        if (!response.ok) {
            throw new Error(`Prediction failed with status ${response.status}: ${response.statusText}`);
        }

        // Parse the response as JSON
        const data = await response.json();

        console.log('Prediction data:', data);  // Log the parsed prediction result

        // Display the prediction result or handle errors in the response
        if (data.prediction) {
            predictionDisplay.textContent = `Prediction: ${data.prediction}`;
        } else if (data.error) {
            predictionDisplay.textContent = `Error: ${data.error}`;
        } else {
            predictionDisplay.textContent = "Error: Unexpected response format.";
        }
    } catch (error) {
        // Handle network or unexpected errors
        console.error('Error during prediction:', error);
        predictionDisplay.textContent = `Error: ${error.message}`;
    }
});
