from google.cloud import aiplatform

# Assuming you've authenticated your notebook to access GCP resources
client = aiplatform.Client(project="llm-test-435804")

# Endpoint name for your deployed Gemini model obtained from AI Platform
endpoint = client.get_endpoint(endpoint="projects/llm-test-435804/locations/us-central1/endpoints/your_endpoint_name")

# Use the endpoint to make predictions
prompt = "Tell me a joke."
response = endpoint.predict(instances=[prompt])

# Access the predicted text from the response
predicted_text = response.predictions[0]
print(predicted_text)