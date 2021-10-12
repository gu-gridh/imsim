import argparse
import os
import yaml
import glob

from imsim import utils

def run(model_url='https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4', 
        height=299, 
        width=299, 
        batch_size=128, 
        pattern=[], 
        destination_dir='', 
        n_features=2048, 
        n_neighbors=30):

    from imsim import ImageSimilarity

    # If only one pattern
    if isinstance(pattern, str):
    
      index = ImageSimilarity.from_pattern(height, width, batch_size, pattern)\
                            .embed(model_url, n_features, n_neighbors)\
                            .rescale()\
                            .save_to(destination_dir)

    # Pass all files if multiple patterns (or filenames)
    elif isinstance(pattern, list):

      files = []      
      for p in pattern:
        files.extend(glob.glob(p))

      index = ImageSimilarity.from_files(height, width, batch_size, files)\
                             .embed(model_url, n_features, n_neighbors)\
                             .rescale()\
                             .save_to(destination_dir)

def main():

  parser = argparse.ArgumentParser(description='Process a set of images into an image embedding and similarity index.')

  # Argument set
  parser.add_argument('-s', '--pattern',      type=str, nargs='+', help='The immediate source directory with files')
  parser.add_argument('-m', '--model-url',    type=str, metavar="model_url", help='A TensorflowHub model url.')
  parser.add_argument('-H', '--height',       type=int, help='Desired height in pixels')
  parser.add_argument('-W', '--width',        type=int, help='Desired width in pixels')
  parser.add_argument('-n', '--n-features',   type=int, metavar="n_features", help='Dimensionality of output embedding')
  parser.add_argument('-N', '--n-neighbors',  type=int, metavar="n_neighbors", help='Number of similar images to index')
  parser.add_argument('-d', '--destination',  type=str, help='Destination directory to store embedding and indexing')
  parser.add_argument('-b', '--batch-size',   type=int, metavar="batch_size", default=128, help='batch size during embedding')
  parser.add_argument('-c', '--config',       type=str, help='Configuration file')
  parser.add_argument('-l', '--logging',      type=int, help='Level of logging: All=0, Warnings=1, Errors=2, Fatal=3')

    # Parse the input
  args = parser.parse_args()

  params = dict()
  try: 

    # First check for configuration file
    if args.config:

      # Configuration file is default
      with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

      # Parse the configuration file
      params.update(utils.parse_config(config))
          
    # Overwrite any arguments from config
    params.update({k:v for k,v in utils.parse_args(args).items() if v is not None})

    print(params)

    # Set logging level
    utils.set_logging_level(params.pop('logging'))

    # Run application
    run(**params)

  except Exception as e:
    # TODO: Move to logging
    print(e)


if __name__ == '__main__':

    main()