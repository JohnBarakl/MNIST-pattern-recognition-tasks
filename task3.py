import matplotlib.pyplot as plt

import task1, task2

import numpy as np


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


def maximin(matrix: np.ndarray, num_of_clusters, distance=euclidean_distance):
    # Ψευδώνυμο του matrix για ευκολότερη χρήση.
    x = matrix

    # Κέντρα ομάδων.
    z = []

    # ------ Βήμα 1 ------ #
    # z_1 = x_1
    z.append(x[0])

    # ------ Βήμα 2 ------ #

    # Υπολογισμός των αποστάσεων όλων των δειγμάτων x_i από το z_1.
    d_from_z_1 = distance(z[0], x)

    # z_2 = το πιο απομακρυσμένο δείγμα από το z_1.
    # Αν υπάρχουν περισσότερα του ενός τέτοια σημεία, επιλέγεται αυθαίρετα το πρώτο.
    z.append(x[np.argmax(d_from_z_1)])

    # Τα βήματα 3 και 4 εκτελούνται μέχρι ικανοποίησης της συνθήκης τερματισμού (επίτευξη επιθυμητού αριθμού κέντρων).
    while len(z) < num_of_clusters:
        # ------ Βήμα 3 ------ #
        # Υπολογισμός των αποστάσεων όλων των δειγμάτων x_i από το z_i.
        distances = np.array([distance(x, i) for i in z])

        # Κρατάω την ελάχιστη απόσταση Δ_i για κάθε δείγμα x_i.
        delta = np.min(distances, axis=0)

        # Από τις κρατημένες Δ_i, παίρνω τη μέγιστη, Δ_max (στην πράξη το i του z_i για το οποίο "έδωσε" τη Δ_max).
        delta_max_i = np.argmax(delta)

        # ------ Βήμα 4 ------ #
        # Αν η συνθήκη τερματισμού ισχύει, τότε ο αλγόριθμος τερματίζεται: Έλεγχος από while loop.
        # Αλλιώς, έχουμε νέο κέντρο z' = x_i όπου Δ_i = Δ_max.
        # Αν υπάρχουν περισσότερα του ενός τέτοια σημεία, επιλέγω αυθαίρετα το πρώτο.
        z.append(x[delta_max_i])

    return z


def k_means(matrix: np.ndarray, num_of_clusters, distance=euclidean_distance):
    # Ψευδώνυμο του matrix για ευκολότερη χρήση.
    x = matrix

    # Αρχική τιμή του z, θα ανατραπεί στη 1η επανάληψη.
    z = np.array([[0, 0] for i in range(num_of_clusters)])

    # ------ Βήμα 1 ------ #
    # Επιλέγονται τα αρχικά κέντρα από τα αποτελέσματα του αλγορίθμου MaxiMin.
    z_new = np.array(maximin(matrix, num_of_clusters, distance))

    # ------ Βήμα 4 ------ #
    # Σύγκριση z με z_new:
    #   Άν ταυτίζονται: Τέλος εκτέλεσης.
    #   Διαφορετικά: Συνεχίζω στο βήμα 2.
    while np.not_equal(z, z_new).all():
        z = z_new

        # ------------------------ Βήμα 2: ------------------------- #
        # ------ Αντιστοίχιση σημείων στις ομάδες που ανήκουν ------ #

        # Αρχικοποίηση συνόλων σημείων κάθε ομάδας.
        S = [[] for i in range(num_of_clusters)]

        # Υπολογισμός των αποστάσεων όλων των δειγμάτων x_i από το z_i.
        distances = np.array([distance(x, i) for i in z])

        # Κρατάω τον δείκτη ομάδας το κέντρο της οποίας δίνει ελάχιστη απόσταση Δ_i, για κάθε δείγμα x_i.
        delta_i = np.argmin(distances, axis=0)

        # Αντιστοίχιση κάθε σημείου στην ομάδα που ανήκει.
        for xi, zi in enumerate(delta_i):
            S[zi].append(x[xi])

        # ------------- Βήμα 3: ------------- #
        # ------ Ενημέρωση των κέντρων ------ #
        z_new = np.array([np.average(S[i], axis=0) for i in range(num_of_clusters)])

    return z_new, S


def visualize_clustered_M_cap(clustering_result):
    # Δημιουργία scatter plot της κάθε ομάδας.
    plt.scatter(np.array(clustering_result[1][0])[:, 0], np.array(clustering_result[1][0])[:, 1], c='m', alpha=0.2,
                marker='o')
    plt.scatter(np.array(clustering_result[1][1])[:, 0], np.array(clustering_result[1][1])[:, 1], c='g', alpha=0.2,
                marker='o')
    plt.scatter(np.array(clustering_result[1][2])[:, 0], np.array(clustering_result[1][2])[:, 1], c='b', alpha=0.2,
                marker='o')
    plt.scatter(np.array(clustering_result[1][3])[:, 0], np.array(clustering_result[1][3])[:, 1], c='r', alpha=0.2,
                marker='o')
    plt.show()


def calculate_cluster_purity(image_matrix: np.ndarray, label_matrix: np.ndarray, clustering_results: list):
    # "Κρατάω" μόνο τα σύνολα των ομάδων.
    sets = clustering_results[1]

    # Μετατροπή σε λίστα για εύκολη αναζήτηση (στη συνέχεια) δείκτη εικόνας με τη μέθοδο index()
    # για αντιστοίχιση σε ετικέτα
    image_list = image_matrix.tolist()

    # Μεταβλητή μερικού αθροίσματος.
    s = 0

    # Υπολογισμός πληρότητας βάσει του τύπου: (1/N)* ∑_{m∈M} max_{d∈D}|m∩d|, όπου N ο αριθμός παραδειγμάτων ομαδοποίησης
    # (εδώ αριθμός εικόνων), M οι ομάδες (clusters) του αποτελέσματος και D οι κλάσεις στις οποίες στην πραγματικότητα
    # ανήκουν τα παραδείγματα ομαδοποίησης.
    for cluster in sets:
        class_examples = [0 for i in range(10)]
        for example in cluster:
            class_examples[label_matrix[image_list.index(example.tolist())]] += 1
        s += class_examples[np.argmax(class_examples)]

    return s / len(image_matrix)


# Εκτέλεση του ζητούμενου task.
if __name__ == '__main__':
    # Επεξεργασία Task 1.
    M, _, L_tr, _ = task1.get_M_N_Ltr_Lte()

    # Επεξεργασία Task 2.
    M_cap = task2.reshape_and_extract_feature_vector(M)

    # Clustering με K-means με αρχικοποίηση σημείων από maximin.
    results = k_means(M_cap, 4)

    # Κλήση για οπτικοποίηση.
    visualize_clustered_M_cap(results)

    # Υπολογισμός και εκτύπωσης Purity των αποτελεσμάτων ομαδοποίησης.
    print('Purity=', calculate_cluster_purity(M_cap, L_tr, results))
