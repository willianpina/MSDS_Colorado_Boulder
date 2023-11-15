from PIL import Image

def read_image(image_path):
    return Image.open(image_path)