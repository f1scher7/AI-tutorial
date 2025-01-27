import numpy as np
from datetime import datetime
from env_loader import SAVED_MODELS_PATH


def save_nn_model(file_name, model_info):
    timestamp =datetime.now().strftime("%Y.%m.%d-%H:%M:%S")

    file = f'{SAVED_MODELS_PATH}{file_name}_{timestamp}'
    np.save(file, model_info)

    print(f"{file_name} was saved to {file}")


def load_saved_nn_model(file_name):
    nn_model_info = np.load(f'{file_name}', allow_pickle=True).item()

    return nn_model_info