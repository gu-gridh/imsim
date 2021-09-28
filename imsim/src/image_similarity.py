from .features import Embedding
from .data import ImageDataset
from .nearest_neighbours import Nearest

import umap
import sklearn
import os
import json 

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


        self.nearest = Nearest(dims=n_features).build(images_dict)
        self.projection = umap.UMAP(densmap=False, n_neighbors=n_neighbors, random_state=42).fit_transform(embedding)
        self.paths = paths

        return self

    def rescale(self, range=[0,1]):

        rescaler = sklearn.preprocessing.MinMaxScaler(feature_range=range)
        
        if self.projection:
            self.projection = rescaler.transform(self.projection)

        return self

    def save_to(self, destination_folder):

        images_dict = {path.decode('utf8'): {'x': float(feature[0]), 'y': float(feature[1])} for path, feature in zip(self.paths, self.projection)}

        with open(os.path.join(destination_folder, 'projection.json'), 'w') as f:
            json.dump(images_dict, f, indent=4)

        # with open(os.path.join(destination_folder, 'nearest.npz')) as f:
        #     pass

        return self