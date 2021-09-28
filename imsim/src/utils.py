import os
import logging

def set_logging_level(level: int):

    if isinstance(level, int) and level < 4:

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(level)

        logging.getLogger('tensorflow').setLevel(level)

    else:
        raise ValueError("Logging level must be integer between 0 and 3.")


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
    n_neighbors = config['similarity']['number of similar images'],
    logging = config['logging']['level']
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
    n_neighbors = args.n_neighbors,
    logging = args.logging
  )

  return params

