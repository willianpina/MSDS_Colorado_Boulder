from matplotlib import pyplot as plt

def display_images(images, title):
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(images[i])
        plt.title(title[i])
        plt.axis('off')
    plt.show()
