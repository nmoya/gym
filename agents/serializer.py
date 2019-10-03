import os
import rootpath
from keras.models import model_from_json

ROOT = rootpath.detect()
NETWORKS_FOLDER = "networks/"

def new_extension(file_path, new_extension):
    segments = file_path.split(".")
    name = "".join(segments[:-1])
    return "{}.{}".format(name, new_extension)

def build_fullpath(file_path):
    return os.path.join(ROOT, NETWORKS_FOLDER, file_path)

def save_model(file_path, model):
    model_json = model.to_json()
    with open(build_fullpath(file_path), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(build_fullpath(new_extension(file_path, "h5")))
    print("Files:\n{}\n{}".format(build_fullpath(file_path), build_fullpath(new_extension(file_path, "h5"))))

def load_model(file_path):
    model_file = open(build_fullpath(file_path), 'r')
    loaded_model_json = model_file.read()
    model_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(build_fullpath(new_extension(file_path, "h5")))
    return loaded_model
