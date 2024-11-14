document.addEventListener("DOMContentLoaded", function() {
    let featureDescriptions = {};
    let productFeatures = {};

    // Categorical options for dropdowns
    const categoricalOptions = {
        "job": ["admin", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed"],
        "marital": ["single", "married", "divorced"],
        "education": ["primary", "secondary", "tertiary", "unknown"],
        "default": ["yes", "no"],
        "contact": ["unknown", "telephone", "cellular"],
        "poutcome": ["unknown", "other", "failure", "success"]
    };

    // Fetch feature descriptions and product features
    fetch('/feature_descriptions')
        .then(response => response.json())
        .then(data => { featureDescriptions = data; })
        .catch(error => console.error('Error fetching feature descriptions:', error));

    fetch('/product_features')
        .then(response => response.json())
        .then(data => { productFeatures = data; })
        .catch(error => console.error('Error fetching product features:', error));

    document.getElementById("product_choice").addEventListener("change", displayRelevantFeatures);

    function displayRelevantFeatures() {
        const selectedProduct = document.getElementById('product_choice').value;
        const featureDiv = document.getElementById('featureInputs');
        featureDiv.innerHTML = ''; // Clear existing fields

        if (!selectedProduct) return;

        // Get relevant features based on product choice
        const relevantFeatures = productFeatures[selectedProduct === "1" ? "cd_account" :
            selectedProduct === "2" ? "loan" :
            selectedProduct === "3" ? "securities" : "term_deposit"];

        relevantFeatures.forEach(feature => {
            if (featureDescriptions[feature]) {
                const label = document.createElement('label');
                label.innerHTML = `${feature}: ${featureDescriptions[feature]}<br>`;

                // If the feature has predefined options, create a dropdown
                if (categoricalOptions[feature]) {
                    const select = document.createElement('select');
                    select.id = feature;
                    select.name = feature;
                    select.required = true;

                    // Add options to the dropdown
                    categoricalOptions[feature].forEach(option => {
                        const optionElement = document.createElement('option');
                        optionElement.value = option;
                        optionElement.textContent = option.charAt(0).toUpperCase() + option.slice(1);  // Capitalize first letter
                        select.appendChild(optionElement);
                    });

                    featureDiv.appendChild(label);
                    featureDiv.appendChild(select);
                } else {
                    // If no predefined options, create a text input
                    const input = document.createElement('input');
                    input.type = "text";
                    input.id = feature;
                    input.name = feature;
                    input.required = true;

                    featureDiv.appendChild(label);
                    featureDiv.appendChild(input);
                }

                featureDiv.appendChild(document.createElement('br'));
            }
        });
    }

    function submitForm() {
        const formData = new FormData(document.getElementById('recommendationForm'));
        const data = Object.fromEntries(formData.entries());

        const payload = {
            product_choice: data.product_choice,
            user_data: data
        };

        fetch('/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(response => response.json())
        .then(result => {
            document.getElementById('result').textContent = `Recommendation: ${result.recommendation}`;
        })
        .catch(error => console.error('Error:', error));
    }

    window.submitForm = submitForm;  // Make submitForm accessible globally
});
