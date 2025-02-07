import os
import numpy as np
from PIL import Image
from env_loader import DIGITS_DATASET, DIGITS_PROCESSED_DATASET_FILE


def load_and_process_dataset():
    data = []
    labels = []

    for label in range(10):
        folder_path = os.path.join(DIGITS_DATASET, str(label))

        img_inc = 0

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            try:
                img = Image.open(img_path).convert('L')
                # img.show()

                resized_img = img.resize((28, 28))
                # resized_img.show()

                # Binarization (thresholding)
                binary_img = resized_img.point(lambda p: 255 if p > 64 else 0)
                # binary_img.show()

                data.append(np.array(binary_img))
                labels.append(label)

                img_inc += 1
            except Exception as e:
                print(f'Error while loading {img_path}: {e}')

        print(f"'{label}' {img_inc} images were processed")

    data_reshaped = np.array(data, dtype=np.uint8).reshape(-1, 28, 28, 1)
    label_one_hot = np.eye(10, dtype=np.uint8)[labels]

    print(data_reshaped.shape)
    print(label_one_hot.shape)

    data_labels = {
        "data": data_reshaped,
        "labels_one_hot": label_one_hot
    }

    np.save(DIGITS_PROCESSED_DATASET_FILE, data_labels)


def load_processed_dataset():
    data = np.load(DIGITS_PROCESSED_DATASET_FILE, allow_pickle=True).item()
    return data["data"], data["labels_one_hot"]


# load_and_process_dataset()

# data_test, labels_test = load_processed_dataset()
#
# print(data_test[0])
# print(labels_test[0])


