import requests

# Path to your input image
input_path = "test.jpg"
output_path = "enhanced.png"

# Open and send the image
with open(input_path, "rb") as img_file:
    files = {"image": img_file}
    response = requests.post("http://localhost:5000/api/enhance", files=files)

# Check result
if response.status_code == 200:
    with open(output_path, "wb") as out_file:
        out_file.write(response.content)
    print(f"✅ Enhanced image saved as: {output_path}")
else:
    print(f"❌ Error {response.status_code}: {response.text}")
