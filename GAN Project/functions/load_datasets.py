import tensorflow as tf
import os

def load_datasets(monet_path, photo_path, test_ratio=None):
    # Verifica a extensão dos arquivos nos diretórios especificados
    monet_files = tf.io.gfile.glob(str(monet_path + '/*'))
    photo_files = tf.io.gfile.glob(str(photo_path + '/*'))

    # Verifica se os arquivos são TFRecord ou JPEG
    is_tfrec = monet_files[0].endswith('.tfrec') and photo_files[0].endswith('.tfrec')

    if is_tfrec:
        # Carregando os datasets a partir de arquivos TFRecord
        monet_dataset = tf.data.TFRecordDataset(monet_files)
        photo_dataset = tf.data.TFRecordDataset(photo_files)
    else:
        # Carregando os datasets a partir de arquivos JPEG
        monet_dataset = tf.data.Dataset.list_files(monet_files, shuffle=False)
        photo_dataset = tf.data.Dataset.list_files(photo_files, shuffle=False)

    # Define a proporção de divisão para treino e teste
    test_ratio = test_ratio if test_ratio is not None else 0.2
    train_ratio = 1 - test_ratio
    
    total_monet_files = len(monet_files)
    total_photo_files = len(photo_files)
    
    train_size_monet = int(train_ratio * total_monet_files)
    train_size_photo = int(train_ratio * total_photo_files)
    
    train_monets = monet_dataset.take(train_size_monet)
    test_monets = monet_dataset.skip(train_size_monet)
    
    train_photos = photo_dataset.take(train_size_photo)
    test_photos = photo_dataset.skip(train_size_photo)
    
    return train_monets, test_monets, train_photos, test_photos

# Uso:
# monet_path = '/caminho/para/monet'
# photo_path = '/caminho/para/photo'
# test_ratio = 0.25  # 25% para o conjunto de teste, por exemplo

# train_monets, test_monets, train_photos, test_photos = load_datasets(monet_path, photo_path, test_ratio)
