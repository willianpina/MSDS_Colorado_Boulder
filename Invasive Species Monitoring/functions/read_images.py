from PIL import Image

def read_images(image_paths):
    if isinstance(image_paths, list):
        return [Image.open(path) for path in image_paths]
    else:
        return Image.open(image_paths)