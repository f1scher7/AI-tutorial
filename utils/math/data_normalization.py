import numpy as np


# We're using Min-Max normalization:
# when we have small spread in training data (age(18, 80));
# when we want to have a range for training data;
def min_max_normalization_for_each_feature(data, norm_range=(0, 1)):
    norm_data = np.zeros_like(data, dtype=float)

    min_max_for_each_features = []
    min_feature_range, max_feature_range = norm_range

    for i in range(data.shape[2]):
        min_feature = np.min(data[:, :, i])
        max_feature = np.max(data[:, :, i])

        if max_feature - min_feature == 0:
            print("WARNING! Min-Max normalization causes division by zero")
            return data, None

        normalized_feature = (data[:, :, i] - min_feature) / (max_feature - min_feature)
        scaled_normalized_feature = normalized_feature * (max_feature_range - min_feature_range) + min_feature_range

        norm_data[:, :, i] = scaled_normalized_feature
        min_max_for_each_features.append((min_feature, max_feature))

    # Adding norm_range as a last element in the list
    min_max_for_each_features.append(norm_range)

    return norm_data, min_max_for_each_features


def min_max_normalization_for_each_feature_with_params(data, norm_params):
    norm_data = np.zeros_like(data, dtype=float)

    min_feature_range, max_feature_range = norm_params[-1]

    for i in range(data.shape[2]):
        min_feature, max_feature = norm_params[i]

        normalized_feature = (data[:, :, i] - min_feature) / (max_feature - min_feature)
        scaled_normalized_feature = normalized_feature * (max_feature_range - min_feature_range) + min_feature_range
        norm_data[:, :, i] = scaled_normalized_feature

    return norm_data


def min_max_denormalization_for_each_feature(norm_data, norm_data_params):
    denorm_data = np.zeros_like(norm_data, dtype=float)

    min_feature_range, max_feature_range = norm_data_params[-1]

    for i in range(norm_data.shape[2]):
        min_feature, max_feature = norm_data_params[i]

        unscaled_norm_feature = (norm_data[:, :, i] - min_feature_range) / (max_feature_range - min_feature_range)
        denorm_feature = unscaled_norm_feature * (max_feature - min_feature) + min_feature

        denorm_data[:, :, i] = denorm_feature

    return denorm_data



# We're using Log normalization:
# when we have large numbers in training data;
# when we have large spread in training data (price(1000$-1000000$));
def log_normalization(x):
    x += 1e-10 # We're adding a small value to avoid log(0)
    return np.log(x)
