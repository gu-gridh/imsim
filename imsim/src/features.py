#%%
import tensorflow as tf
import tensorflow_hub as hub
import umap
from tqdm import tqdm

#%%
class Embedding:

    def __init__(self, model) -> None:
        self.model = model

    @classmethod
    def create_from_hub(cls, 
        model_path="https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_m/feature_vector/2",
        width=480,
        height=480,
        depth=3
    ):

        model = tf.keras.Sequential([
            hub.KerasLayer(model_path, trainable=False)
        ])

        model.build([None, width, height, depth])

        return cls(model)

    def embed(self, x):

        try:
            y = self.model(x)

        except Exception as e:

            print(e)
            y = None

        return y

    def transform(self, images):
        
        features = []
        paths    = []

        for batch, ps in tqdm(images.data, total=images.data.cardinality().numpy()):
            embedding = self.embed(batch)

            # Add metadata
            paths.append(ps)
            features.append(embedding)

        # Flatten 
        features = tf.concat(features, axis=0).numpy()
        paths = tf.concat(paths, axis=0).numpy().tolist()

        return features, paths

# %%
class Reduction:

    def __init__(self) -> None:
        pass

    def reduce(self, features):

        embedding = umap.UMAP().fit_transform(features)
