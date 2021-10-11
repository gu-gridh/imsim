from .features import Embedding
from .data import ImageDataset
from .nearest_neighbours import Nearest

import umap
import sklearn
import os
import json 
import tensorflow as tf

class ImageSimilarity:

    def __init__(self, images: ImageDataset, height: int, width: int) -> None:
        self.images = images
        self.height = height
        self.width = width
        self.paths = None
        self.projection = None

    @classmethod
    def from_pattern(cls, height, width, batch_size, pattern):
        
        # Form a glob pattern for all files and index data
        images = ImageDataset.from_pattern(pattern).prepare(height, width, batch_size)


        return cls(images, height, width)

    @classmethod
    def from_files(cls, height, width, batch_size, image_list):
        
        images = ImageDataset.from_files(image_list).prepare(height, width, batch_size)
        
        return cls(images, height, width)


    def embed(self, model_path, n_features, n_neighbors=60):
        
        # Create an embedding in feature space
        embedding, paths = Embedding.create_from_hub(model_path=model_path, width=self.width, height=self.height)\
                                    .transform(self.images)

        images_dict = {path: feature for path, feature in zip(paths, embedding)}

        self.paths = paths
        self.nearest = Nearest(dims=n_features).build(images_dict)
        self.projection = umap.UMAP(densmap=False, n_neighbors=n_neighbors, random_state=42).fit_transform(embedding)

        return self

    def rescale(self, range=[0,1]):

        rescaler = sklearn.preprocessing.MinMaxScaler(feature_range=range)
        
        if self.projection is None:
            pass
        else:
            self.projection = rescaler.fit_transform(self.projection)

        return self

    def save_to(self, destination_folder):

        metadata = []

        for path, feature in zip(self.paths, self.projection):
            
            # Get dimensions of original image
            img = tf.io.read_file(path)
            img = tf.io.decode_jpeg(img, channels=3)
            height, width, depth = img.get_shape()

            # Get position in 2D-plane
            x = float(feature[0])
            y = float(feature[1])

            metadata.append({
                'path': path.decode('utf-8'),
                'height': height,
                'width': width,
                'x': x,
                'y': y
            })


        with open(os.path.join(destination_folder, 'projection.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

        # with open(os.path.join(destination_folder, 'nearest.npz')) as f:
        #     pass

        return self