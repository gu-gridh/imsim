import argparse
import os
import yaml
import glob

from imsim import ImageSimilarity

def parse_config(config):

  # Save params in dict
  params = dict(
    model_url = config['model']['url'], 
    height = config['model']['height'], 
    width = config['model']['width'], 
    batch_size = config['model']['batch size'], 
    pattern = config['import']['pattern'], 
    destination_dir = config['export']['destination'], 
    n_features = config['model']['number of features'], 
    n_neighbors = config['similarity']['number of similar images']
  )

  return params

def parse_args(args): 

  # Save params in dict
  params = dict(
    model_url = args.model_url, 
    height = args.height, 
    width = args.width, 
    batch_size = args.batch_size, 
    pattern = args.pattern, 
    destination_dir = args.destination, 
    n_features = args.n_features, 
    n_neighbors = args.n_neighbors
  )

  return params

def run(model_url, height, width, batch_size, pattern, destination_dir, n_features, n_neighbors):

    # If only one pattern
    if isinstance(pattern, str):
    
      index = ImageSimilarity.from_pattern(height, width, batch_size, pattern)\
                            .embed(model_url, n_features, n_neighbors)\
                            .save_to(destination_dir)

    # Pass all files if multiple patterns (or filenames)
    elif isinstance(pattern, list):

      files = []      
      for p in pattern:
        files.extend(glob.glob(p))

      index = ImageSimilarity.from_files(height, width, batch_size, files)\
                             .embed(model_url, n_features, n_neighbors)\
                             .save_to(destination_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process a set of images into an image embedding and similarity index.')

    # Argument set
    parser.add_argument('-s', '--pattern',      type=str, nargs='+', help='The immediate source directory with files')
    parser.add_argument('-m', '--model-url',    type=str, metavar="model_url", help='A TensorflowHub model url.')
    parser.add_argument('-H', '--height',       type=int, help='Desired height in pixels')
    parser.add_argument('-W', '--width',        type=int, help='Desired width in pixels')
    parser.add_argument('-n', '--n-features',   type=int, metavar="n_features", help='Dimensionality of output embedding')
    parser.add_argument('-d', '--destination',  type=str, help='Destination directory to store embedding and indexing')
    parser.add_argument('-b', '--batch-size',   type=int, metavar="batch_size", default=128, help='batch size during embedding')
    parser.add_argument('-c', '--config',       type=str, help='Configuration file')

    # Parse the input
    args = parser.parse_args()

    params = dict()

    # First check for configuration file
    if args.config:

      try:

        # Configuration file is default
        with open(args.config, 'r') as f:
          config = yaml.load(f, Loader=yaml.Loader)

        # Parse the configuration file
        params.update(parse_config(config))

      except Exception as e:
        print(e)

    try:
      
      # Overwrite any arguments from config
      params.update(parse_args(args))

    except Exception as e:
      print(e)

    # Run application
    run(**params)
