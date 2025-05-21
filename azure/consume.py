import requests
import base64

endpoint = ""
api_key = ""

with open("image4.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

response = requests.post(endpoint, json={"image": base64_image}, headers=headers)
response.raise_for_status()

output_image = base64.b64decode(response.json()["image"])
with open("output.jpg", "wb") as out_file:
    out_file.write(output_image)

print("Inference complete. Saved output.jpg")
