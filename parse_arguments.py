import argparse
import os

#type function for argparse to check directories are valid
def isdir(directory):
    if not os.path.isdir(directory):
        raise argparse.ArgumentTypeError("Could not find given data directory \"{}\". Please use a valid directory".format(directory))
    return directory

def ispath(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError("Could not find image at {}. Please use a valid image path.".format(path))
    return path

def make_train_parser():
    parser = argparse.ArgumentParser(description='Train a flower species identifier')
    parser.add_argument('data_dir', action='store', type=isdir)
    parser.add_argument('--arch', action='store', choices=['densenet', 'alexnet'], default='densenet')
    parser.add_argument('--hidden_units', action='store', type=int, default=512)
    parser.add_argument('--epochs', action='store', type=int, default=1)
    parser.add_argument('--learning_rate', action='store', type=float, default=0.001)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--save_dir', action='store', default='checkpoint.pth')

    args = vars(parser.parse_args())

    data_dir = args['data_dir']
    arch = args['arch']
    hidden_units = args['hidden_units']
    epochs = args['epochs']
    learning_rate = args['learning_rate']
    gpu = args['gpu']
    save_dir = args['save_dir']
    
    return data_dir, arch, hidden_units, epochs, learning_rate, gpu, save_dir

def make_predict_parser():
    parser = argparse.ArgumentParser(description='Predict a flower species identifier')
    parser.add_argument('img_path', action='store', type=ispath)
    parser.add_argument('--category_names', action='store', default=False)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--top_k', action='store', type=int, default=3)
    
    args = vars(parser.parse_args())
    
    img_path = args['img_path']
    cat_names = args['category_names']
    gpu = args['gpu']
    topk = args['top_k']
    
    return img_path, cat_names, gpu, topk