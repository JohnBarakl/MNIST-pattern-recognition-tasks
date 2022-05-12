import numpy as np
import matplotlib.pyplot as plt


def load_labels(filename: str):
    with open(filename, 'rb') as file:
        # Επαλήθευση του magic number του header.
        magic_number = int.from_bytes(file.read(4), byteorder='big')
        if magic_number != 0x00000801:
            raise ValueError('File header did not validate.')

        number_of_labels = int.from_bytes(file.read(4), byteorder='big')

        # Διαβάζω το σύνολο των ετικετών και το επιστρέφω.
        return list(file.read(number_of_labels))


def load_images(filename: str, target_indices: list):
    with open(filename, 'rb') as file:

        # Επαλήθευση του magic number του header.
        magic_number = int.from_bytes(file.read(4), byteorder='big')
        if magic_number != 0x00000803:
            raise ValueError('File header did not validate.')

        number_of_images = int.from_bytes(file.read(4), byteorder='big')
        img_rowsize = int.from_bytes(file.read(4), byteorder='big')
        img_colsize = int.from_bytes(file.read(4), byteorder='big')
        img_bytesize = img_rowsize * img_colsize  # Συνολικός αριθμός bytes κάθε εικόνας.

        images = []

        # Διάβασμα των εικόνων στις θέσεις που ζητήθηκε. Αν η παράμετρος θέσεων είναι None, διαβάζονται όλες.
        if len(target_indices) == 0:
            while number_of_images > 0:
                # Διαβάζω μία ολόκληρη εικόνα: Εφόσον είναι αποθηκευμένες row-wise, διαβάζω γραμμές * στήλες bytes.
                images.append(np.array(list(file.read(img_bytesize))))

                number_of_images -= 1
            return images
        else:
            # Για να λειτουργήσει ο ακόλουθος αλγόριθμος, χρειάζεται ταξινομημένη λίστα δεικτών.
            target_indices.sort()

            # Η φόρτωση ξεκινάει από την 1η εικόνα.
            current_index = 0
            for index in target_indices:
                # Αν "βρίσκομαι" στη ζητούμενη εικόνα τη διαβάζω.
                if current_index == index:
                    images.append(np.array(list(file.read(img_bytesize))))
                    current_index += 1
                # Αν η τρέχουσα θέση έχει ξεπεράσει το όριο, αγνοώ την παράμετρο και τερματίζω.
                elif current_index > number_of_images:
                    return images
                # Πρέπει να διαβάσω την εικόνα της ζητούμενης θέσης:
                #   Μεταβαίνω στη ζητούμενη θέση αφού αγνοήσω τις ενδιάμεσες εικόνες.
                else:
                    # Μεταβαίνω στη σωστή θέση.
                    file.seek((index - current_index) * img_bytesize, 1)

                    current_index = index

                    # Διαβάζω την εικόνα.
                    images.append(np.array(list(file.read(img_bytesize))))
                    current_index += 1

            return images


def load_subset(image_file: str, label_file: str, numbers_filter: list = None):
    labels = load_labels(label_file)

    # Κάνω επιλογή ψηφίων μόνο αν δίνεται λίστα με ψηφία προς επιλογή.
    if numbers_filter is not None:
        image_indices = []
        filtered_labels = []
        for i in range(len(labels)):
            if labels[i] in numbers_filter:
                filtered_labels.append(labels[i])
                image_indices.append(i)
        labels = filtered_labels
    else:
        image_indices = [i for i in range(len(labels))]

    images = load_images(image_file, image_indices)

    return images, labels


training_images_filename, training_labels_filename = 'datasets/train-images-idx3-ubyte', 'datasets/train-labels-idx1-ubyte'
testing_images_filename, testing_labels_filename = 'datasets/t10k-images-idx3-ubyte', 'datasets/t10k-labels-idx1-ubyte'


def get_M_N_Ltr_Lte():
    # Κατασκευάζω τα M, N, L_tr και L_te όπως ζητούνται.
    train_images, train_labels = load_subset(training_images_filename, training_labels_filename, [1, 3, 7, 9])
    M = np.array(train_images)

    test_images, test_labels = load_subset(testing_images_filename, testing_labels_filename, [1, 3, 7, 9])
    N = np.array(test_images)

    L_tr = np.array(train_labels)

    L_te = np.array(test_labels)

    return M, N, L_tr, L_te


# Εκτέλεση του ζητούμενου task.
if __name__ == "__main__":
    # Φόρτωση δεδομένων.
    data = get_M_N_Ltr_Lte()
    # Κρατάω μόνο τα training data.
    data = data[0], data[2]

    # "Εκτύπωση" 16 εικόνων για επαλήθευση ορθότητας διαδικασίας.
    fig, ax = plt.subplots(4, 4)
    for ii in range(4):
        for jj in range(4):
            i = ii * 4 + jj
            ax[ii, jj].imshow(data[0][i].reshape((28, 28)))
            ax[ii, jj].set_title(label='Image of %d' % data[1][i])
            ax[ii, jj].set_axis_off()
    plt.tight_layout()
    plt.show()

