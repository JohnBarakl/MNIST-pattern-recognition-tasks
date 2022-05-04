import task1

import numpy as np
import matplotlib.pyplot as plt


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

    # Αντιστοίχηση από αρίθμηση κλάσεων με ακεραίους σε χρώματα.
    colors = list(map(lambda l: {1: 'r', 3: 'g', 7: 'b', 9: 'm'}[l], L_tr))

    # Δημιουργία scatter plot.
    ones = M_cap[L_tr == 1]
    threes = M_cap[L_tr == 3]
    sevens = M_cap[L_tr == 7]
    nines = M_cap[L_tr == 9]
    plt.scatter(ones[0, 0], ones[0, 1], c='r', alpha=0.2, label='Class label 1')
    plt.scatter(threes[0, 0], threes[0, 1], c='g', alpha=0.2, label='Class label 3')
    plt.scatter(sevens[0, 0], sevens[0, 1], c='b', alpha=0.2, label='Class label 7')
    plt.scatter(nines[0, 0], nines[0, 1], c='m', alpha=0.2, label='Class label 9')
    plt.scatter(M_cap[:, 0], M_cap[:, 1], c=colors, alpha=0.02, marker=',')
    plt.legend()
    plt.show()


# Εκτέλεση του ζητούμενου task.
if __name__ == '__main__':
    visualize_scatter()
