from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np
import os
from scipy.linalg import sqrtm

class ImageEvaluator:
    def __init__(self, reference_images_dir, transformed_images_dir, target_size=(224, 224), batch_size=32):
        self.reference_images_dir = reference_images_dir
        self.transformed_images_dir = transformed_images_dir
        self.target_size = target_size
        self.batch_size = batch_size


    def load_images_from_directory(self, directory):
        image_files = [f for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')]
        for i in range(0, len(image_files), self.batch_size):
            batch_files = image_files[i:i + self.batch_size]
            images = []
            for file in batch_files:
                img_path = os.path.join(directory, file)
                img = image.load_img(img_path, target_size=self.target_size)
                img = image.img_to_array(img)
                img = preprocess_input(img)
                images.append(img)
            yield np.array(images)

    def get_vgg19_features(self, image_generator):
        model = VGG19(include_top=False, pooling='avg', input_shape=(self.target_size[0], self.target_size[1], 3))
        features_model = Model(inputs=model.input, outputs=model.get_layer('block5_pool').output)
        all_features = []
        for images in image_generator:
            features = features_model.predict(images)
            all_features.append(features)
        return np.vstack(all_features).reshape(-1, features.shape[-1])
    
    def get_style_image_features(self, style_image_path):
        img = image.load_img(style_image_path, target_size=self.target_size)
        img = image.img_to_array(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        model = VGG19(include_top=False, pooling='avg', input_shape=(self.target_size[0], self.target_size[1], 3))
        features_model = Model(inputs=model.input, outputs=model.get_layer('block5_pool').output)
        features = features_model.predict(img)

        return features.reshape(-1, features.shape[-1])

    def calculate_fid(self, real_features, generated_features):
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
        ssdiff = np.sum((mu1 - mu2)**2.0)
        covmean = sqrtm(sigma1.dot(sigma2))
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def calculate_memorization_distance(self, generated_features, real_features, epsilon=0.1):
        distances = []
        for gf in generated_features:
            cos_distances = 1 - np.dot(real_features, gf) / (np.linalg.norm(gf) * np.linalg.norm(real_features, axis=1))
            min_distance = np.min(cos_distances)
            distances.append(min_distance if min_distance < epsilon else 1.0)
        return np.mean(distances)

    def calculate_mifid(self, fid, memorization_distance):
        return fid * (1 / memorization_distance)

    def evaluate_images(self, style_image_path):
        # Carregar e extrair características da imagem de estilo de referência
        style_features = self.get_style_image_features(style_image_path)

        # Carregar imagens transformadas e extrair características
        transformed_image_generator = self.load_images_from_directory(self.transformed_images_dir)
        transformed_features = self.get_vgg19_features(transformed_image_generator)

        # Calcular FID
        fid_value = self.calculate_fid(style_features, transformed_features)

        # Calcular distância de memorização e MiFID
        memorization_distance = self.calculate_memorization_distance(transformed_features, style_features)
        mifid_value = self.calculate_mifid(fid_value, memorization_distance)

        return fid_value, mifid_value


