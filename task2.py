import task1

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def reshape_and_extract_feature_vector(images_matrix: np.ndarray):
    m = []

    for img in images_matrix:
        # Μετασχηματίζω την εικόνα σε τετραγωνικό πίνακα με διαστάσεις 28 x 28.
        img = img.reshape((28, 28))

        # Η πρώτη διάσταση έχει τιμή το μέσο της τιμής των pixel των περιττών γραμμών.
        d1 = np.mean(img[1::2, :].reshape((-1,)))

        # Η δεύτερη διάσταση έχει τιμή το μέσο της τιμής των pixel των περιττών στηλών.
        d2 = np.mean(img[:, 0::2].reshape((-1,)))

        # Αποθηκεύω τον μετασχηματισμό της εικόνας.
        m.append((d1, d2))

    return np.array(m)


def visualize_scatter():
    # Επεξεργασία από Task 1.
    M, _, L_tr, _ = task1.get_M_N_Ltr_Lte()

    # Μετασχηματισμός δεδομένων.
    M_cap = reshape_and_extract_feature_vector(M)

    # Αντιστοίχιση από αρίθμηση κλάσεων με ακεραίους σε χρώματα.
    colors = L_tr
    cmap = ListedColormap(['r', 'r', 'g', 'r', 'r', 'r', 'b', 'r', 'm'])

    # -- Δημιουργία scatter plot --
    scatter = plt.scatter(M_cap[:, 0], M_cap[:, 1], c=colors, cmap=cmap, alpha=0.02)

    # Δημιουργία legend.
    legend = plt.legend(*scatter.legend_elements())
    for legend_handle in legend.legendHandles:
        legend_handle.set_alpha(1)

    plt.title(r"$\hat{\mathbf{M}}$ visualization")
    plt.show()

# Εκτέλεση του ζητούμενου task.
if __name__ == '__main__':
    visualize_scatter()
