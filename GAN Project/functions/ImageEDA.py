import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from skimage import feature
import imageio
from tensorflow.keras.models import Model

class ImageEDA:
    def __init__(self, monet_path, photo_path, img_size, model=None, layer_name=None):
        self.monet_path = monet_path
        self.photo_path = photo_path
        self.img_size = img_size
        self.model = model
        self.layer_name = layer_name

    def load_images_from_folder(self, folder_path, num_images):
        images = []
        loaded_images = 0
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") and loaded_images < num_images:
                img = imageio.imread(os.path.join(folder_path, filename))
                if img is not None:
                    images.append(img)
                    loaded_images += 1
        return images


    def perform_EDA(self):
        self.visual_inspection()
        self.color_analysis()
        self.texture_pattern_analysis()
        self.image_complexity()
        self.image_dimensions()
        self.color_intensity_analysis()
        self.pca_feature_visualization()
        self.fourier_transform_analysis()
        self.class_distributions()

        if self.model and self.layer_name:
            # Supondo que uma imagem de amostra está disponível em self.photo_path
            sample_image = self.load_images_from_folder(self.photo_path, 1)[0]
            self.convolutional_filters(self.model, self.layer_name, sample_image)
            self.correlation_analysis(sample_image)
            self.additional_custom_analysis(sample_image)

    # 1. Visual Inspection
    def visual_inspection(self):
        monet_samples = self.load_images_from_folder(self.monet_path, 5)
        photo_samples = self.load_images_from_folder(self.photo_path, 5)

        plt.figure(figsize=(10, 5))
        for i in range(5):
            plt.subplot(2, 5, i+1)
            plt.imshow(monet_samples[i])
            plt.title(f'Monet {i + 1}')
            plt.axis('off')  # Esconde os eixos
            
            plt.subplot(2, 5, i+6)
            plt.imshow(photo_samples[i])
            plt.title(f'Photo {i + 1}')
            plt.axis('off')  # Esconde os eixos
        plt.show()


    # 2. Color Analysis
    def color_analysis(self):
        # For simplicity, let's just plot the color histogram for one image from each class
        monet_samples = self.load_images_from_folder(self.monet_path, 1)
        photo_samples = self.load_images_from_folder(self.photo_path, 1)
        
        monet_img = monet_samples[0]
        photo_img = photo_samples[0]
        
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([monet_img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.title('Monet Color Histogram')
        plt.show()
        
        for i, col in enumerate(color):
            histr = cv2.calcHist([photo_img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.title('Photo Color Histogram')
        plt.show()

    # 3. Texture and Pattern Analysis
    def texture_pattern_analysis(self):
        # Using Gabor filter as an example for texture analysis
        monet_samples = self.load_images_from_folder(self.monet_path, 1)
        monet_img_gray = cv2.cvtColor(monet_samples[0], cv2.COLOR_BGR2GRAY)
        gabor_filter = cv2.getGaborKernel((21, 21), 5, np.pi/4, 10, 1, 0, ktype=cv2.CV_32F)
        gabor_img = cv2.filter2D(monet_img_gray, cv2.CV_8UC3, gabor_filter)
        plt.imshow(gabor_img, cmap='gray')
        plt.title('Gabor Filter Output for Monet')
        plt.show()

    # 4. Image Complexity (for example using edge detection)
    def image_complexity(self):
        monet_samples = self.load_images_from_folder(self.monet_path, 1)
        monet_img_gray = cv2.cvtColor(monet_samples[0], cv2.COLOR_BGR2GRAY)
        edges = feature.canny(monet_img_gray, sigma=1)
        plt.imshow(edges, cmap='gray')
        plt.title('Edge Detection for Monet')
        plt.show()

    # 5. Image Dimensions
    def image_dimensions(self):
        monet_samples = self.load_images_from_folder(self.monet_path, 5)
        dimensions = [(img.shape[0], img.shape[1]) for img in monet_samples]
        print("Dimensions of Monet samples: ", dimensions)

    # 6. Color Intensity Analysis
    def color_intensity_analysis(self):
        monet_samples = self.load_images_from_folder(self.monet_path, 1)
        mean_intensity = np.mean(monet_samples[0], axis=(0, 1))
        print("Mean color intensity of a Monet sample: ", mean_intensity)

    # 7. PCA for Feature Visualization
    def pca_feature_visualization(self):
        monet_samples = self.load_images_from_folder(self.monet_path, 5)
        monet_samples_flattened = [img.flatten() for img in monet_samples]
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(monet_samples_flattened)
        plt.scatter(pca_result[:, 0], pca_result[:, 1])
        plt.title('PCA Scatter Plot')
        plt.show()

    # 8. Fourier Transform for Frequency Domain Analysis
    def fourier_transform_analysis(self):
        monet_samples = self.load_images_from_folder(self.monet_path, 1)
        monet_img_gray = cv2.cvtColor(monet_samples[0], cv2.COLOR_BGR2GRAY)
        f_transform = np.fft.fft2(monet_img_gray)
        f_transform_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_transform_shift))
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Fourier Transform - Magnitude Spectrum')
        plt.show()

    # 9. Convolutional Filters for Feature Maps
    def convolutional_filters(model, layer_name, image):
        # Given a trained model, layer name and an input image,
        # this function will plot the feature maps (filters) of that layer.
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(image)
        
        n_filters = intermediate_output.shape[-1]
        size = intermediate_output.shape[1]
        
        # Plotting the feature maps
        for i in range(n_filters):
            plt.subplot(n_filters // 4, 4, i + 1)
            plt.imshow(intermediate_output[0, :, :, i], cmap='viridis')
            plt.axis('off')
        
        plt.show()

    # 10. Class Distributions (for Monet paintings and photos)
    def class_distributions(self):
        num_monet_samples = len(os.listdir(self.monet_path))
        num_photo_samples = len(os.listdir(self.photo_path))
        sns.barplot(x=['Monet', 'Photo'], y=[num_monet_samples, num_photo_samples])
        plt.title('Class Distribution')
        plt.show()

    # 11. Correlation Analysis
    def correlation_analysis(image):
        # Convert image to grayscale if it is RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute the correlation matrix
        corr_matrix = np.corrcoef(image, rowvar=False)
        
        # Plot the heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.tight_layout()  # Adicionado para evitar a sobreposição
        plt.show()


    # 12. Additional Custom Analysis
    def additional_custom_analysis(image):
        # Here, you can perform any additional custom analyses you might find useful.
        # For example, one could compute and visualize the image histogram.
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist)
        plt.title('Image Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()
