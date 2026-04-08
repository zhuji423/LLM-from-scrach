
from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents="写一段爱情小说"
)
print(response.text)

# from google import genai
# from google.genai import types
# from PIL import Image

# client = genai.Client()

# prompt = ("Create a picture of a nano banana dish in a fancy restaurant with a Gemini theme")
# response = client.models.generate_content(
#     model="gemini-3.1-flash-image-preview",
#     contents=[prompt],
# )

# for part in response.parts:
#     if part.text is not None:
#         print(part.text)
#     elif part.inline_data is not None:
#         image = part.as_image()
#         image.save("generated_image.png")