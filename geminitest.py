from google import genai

client = genai.Client(api_key="AIzaSyAVYTHaz3qfWB72Qf2ZSj9lbIc6yeZ9nZA")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain how AI works",
)

print(response.text)