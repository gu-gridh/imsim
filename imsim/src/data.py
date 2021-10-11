import tensorflow as tf
from typing import *

class ImageDataset:
    """A simple wrapper around the tf.data.Dataset class.
    """

    def __init__(self, dataset) -> None:
        self.data = dataset

    @classmethod
    def from_pattern(cls, file_pattern: str):

        return cls(tf.data.Dataset.list_files(file_pattern))

    @classmethod
    def from_files(cls, file_list: List[str]):

        return cls(tf.data.Dataset.from_tensor_slices(file_list))

    @staticmethod
    def decode(path, new_height: int, new_width: int) -> Tuple[tf.Tensor, str]:

        # load the raw data from the file as a string
        img = tf.io.read_file(path)

        # convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_jpeg(img, channels=3)

        # Resize to destination size
        x = tf.image.resize_with_pad(img, new_height, new_width)

        return x, path

    def prepare(self, height: int, width: int, batch_size: int):

        self.data = self.data.map(lambda x: ImageDataset.decode(x, height, width), 
                                  num_parallel_calls = tf.data.AUTOTUNE)\
                             .cache()\
                             .batch(batch_size)\
                             .prefetch(buffer_size=tf.data.AUTOTUNE)

        return self

# Usage:
# height = 240
# width  = 240
# batch_size = 128
# ds = ImageDataset.from_pattern("image_directory/*.jpg")
# ds = ds.map(lambda x: ImageDataset.decode(x, height, width), num_parallel_calls = tf.data.AUTOTUNE)\
#        .cache()\
#        .batch(batch_size)\
#        .prefetch(buffer_size=tf.data.AUTOTUNE)\