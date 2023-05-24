import requests
import base64
import io
from PIL import Image


URL_LOCAL = "http://127.0.0.1:7860"
URL_REMOTE = "https://xdecoder-seem.hf.space"


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")


# Example usage
image_path = "C:/Git/imlex/crt-project/data/imlex-bottle/raw/0001.jpg"
base64_string = image_to_base64(image_path)

r1 = requests.get(URL_LOCAL)
print(r1)

post_data = "data:image/jpg;base64," + base64_string

# response = requests.post(URL_LOCAL + "/run/predict", json={
#     "data": [
#         post_data,
#         ["Text"],
#         "bottle",
#         None,
#     ]}).json()

response = requests.post("https://xdecoder-seem.hf.space/run/predict", json={
    "data": [
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
        ["Stroke"],
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
        "hello world",
        {"name": "audio.wav", "data": "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="},
        None,
    ]}).json()

data = response["data"]

image_data = base64.b64decode(data)
image = Image.open(io.BytesIO(image_data))
image.save("./data/imlex-bottle/seem/0001.png")  # Save the image as a file
image.show()  # Display the image
