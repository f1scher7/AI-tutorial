import os
import numpy as np
from PIL import Image
from env_loader import DIGITS_DATASET, DIGITS_PROCESSED_DATASET_FILE


def load_and_process_dataset():
    data = []
    labels = []

    for label in range(10):
        folder_path = os.path.join(DIGITS_DATASET, str(label))

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

                data.append(np.array(binary_img).flatten())
                labels.append(label)
            except Exception as e:
                print(f'Error while loading {img_path}: {e}')

        print(f"'{label}' images were processed")

    data_labels = {
        "data": np.array(data),
        "labels": np.array(labels)
    }

    np.save(DIGITS_PROCESSED_DATASET_FILE, data_labels)


def load_processed_dataset():
    data = np.load(DIGITS_PROCESSED_DATASET_FILE, allow_pickle=True).item()
    return data["data"], data["labels"]


data_test, labels_test = load_processed_dataset()


print(data_test[0])
print(labels_test[0])


