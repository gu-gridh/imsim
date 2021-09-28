import annoy
from typing import * 

class Nearest:

    def __init__(self, dims = 1280, n_trees = 10000) -> None:
        
        self.tree = annoy.AnnoyIndex(dims, metric='angular')
        self.n_trees = n_trees
        self.name_to_index_dict = {}
        self.index_to_name_dict = {}


    def name_to_index(self, image_name):

        return self.name_to_index_dict[image_name]

    def index_to_name(self, idx):

        return self.index_to_name_dict[idx]


    def build(self, images: Dict):
        
        # Index all of the images
        for idx, (name, features) in enumerate(images.items()):
            
            # Add index to name mapping
            self.name_to_index_dict[name] = idx
            
            self.tree.add_item(idx, features)

        # Reverse mapping
        self.index_to_name_dict = {v: k for k, v in self.name_to_index_dict.items()}

        # Build a search tree
        self.tree.build(self.n_trees)

        return self

    def find(self, idx, n_nearest_neighbours = 30) -> Tuple:

        return self.tree.get_nns_by_item(idx, n_nearest_neighbours, include_distances=True)

    def find_by_name(self, name, n_nearest_neighbours = 30) -> Tuple:

        idxs, distance = self.find(self.name_to_index(name), n_nearest_neighbours)    

        return [self.index_to_name(idx) for idx in idxs], distance