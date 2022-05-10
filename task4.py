import task1, task2, task3
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Εφαρμογή της μεθόδου PCA βάσει των τύπων και θεωριών από [ΑΝΑΓΝΩΡΙΣΗ ΠΡΟΤΥΠΩΝ ΚΑΙ ΜΗΧΑΝΙΚΗ ΜΑΘΗΣΗ, 2η έκδοση,
# Christopher M. Bishop], σελίδες 621-623.
def pca(data_matrix: np.ndarray, V: int):
    """
    Εφαρμογή της μεθόδου PCA.
    :param data_matrix: Πίνακας N x D με δεδομένα όπου N είναι ο αριθμός των δειγμάτων και D ο αριθμός διαστάσεων κάθε
                        δείγματος (οι γραμμές περιέχουν D-διάστατα διανύσματα που αποτελούν εικόνες και οι στήλες
                        περιέχουν τα features (στην περίπτωση αυτή τα pixel των εικόνων) των δεδομένων).
    :param V: Ο αριθμός διαστάσεων για τις οποίες πρέπει να βρεθούν ιδιοδιανύσματα.
    :return: Πίνακας με τα V διανύσματα βάσης, αποτελώντας τη βάση προς μετασχηματισμό των δεδομένων μικρότερης διάστασης.
    """
    X = data_matrix  # Ψευδώνυμο του πίνακα.

    m = X.mean(axis=0)  # Εύρεση του μέσου όρου (διανύσματος) του πίνακα.

    N = len(X)  # Αριθμός στοιχείων του πίνακα.

    # Υπολογισμός του πίνακα συνδιακύμανσης
    S = (1 / N) * np.matmul((X - m).T, (X - m))

    # Υπολογισμός των ιδιοδιανυσμάτων του πίνακα συνδιακύμανσης που δίνουν τις κατευθύνσεις των V κυρίων συνιστωσών του
    # πίνακα δεδομένων.
    # Ο πίνακας S είναι συμμετρικός με πραγματικές τιμές, επομένως χρησιμοποιώ την eigh για υπολογισμός των V μεγαλύτερων
    # ιδιοδιανυσμάτων.
    eigenvectors = eigh(S, subset_by_index=[len(S) - V, len(S) - 1])[1]

    # Επιστροφή του πίνακα με τα V μεγαλύτερα ιδιοδιανύσματα, όπου ο πίνακας αυτός περιέχει διανύσματα στήλες v(i) (ιδιοδιανύσματα)
    # και είναι της μορφής [v(0) v(1) ... v(V-1)] όπου το v(0) είναι το ιδιοδιάνυσμα που αντιστοιχεί στη μέγιστη ιδιοτιμή,
    # το v(1) στην αμέσως επόμενη μεγαλύτερη κ.ο.κ.
    return np.array(list(reversed(eigenvectors.T))).T


# Εφαρμογή μείωσης διαστάσεων με τη χρήση PCA.
def dimensionality_reduction_with_pca(data_matrix: np.ndarray, V: int):
    """
    Εφαρμογή μείωσης διαστάσεων με τη χρήση PCA.
    :param data_matrix: Πίνακας N x D με δεδομένα όπου N είναι ο αριθμός των δειγμάτων και D ο αριθμός διαστάσεων κάθε
                        δείγματος (οι γραμμές περιέχουν D-διάστατα διανύσματα που αποτελούν εικόνες και οι στήλες
                        περιέχουν τα features (στην περίπτωση αυτή τα pixel των εικόνων) των δεδομένων).
    :param V: Ο νέος αριθμός διαστάσεων.
    :return: Ο μετασχηματισμένος πίνακας N x V με νέες διαστάσεις.
    """

    # Υπολογισμός βάσης προς προβολή.
    base = pca(data_matrix, V)

    # Υπολογισμός προβολής δειγμάτων του X (data_matrix) προς τη νέα βάση εκτελώντας μείωση διαστατικότητας.
    return np.matmul(base.T, data_matrix.T).T


def visualize_plain_scatter(M, L_tr):
    # Αντιστοίχιση από αρίθμηση κλάσεων με ακεραίους σε χρώματα.
    colors = list(map(lambda l: {1: 0, 3: 1, 7: 2, 9: 3}[l], L_tr))
    cmap = ListedColormap(['r', 'g', 'b', 'm'])

    # -- Δημιουργία scatter plot --
    scatter = plt.scatter(M[:, 0], M[:, 1], c=colors, cmap=cmap, alpha=0.02)

    # Δημιουργία legend.
    leg = plt.legend(*scatter.legend_elements())
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    plt.show()


# Εκτέλεση του ζητούμενου task.
if __name__ == '__main__':
    # Επεξεργασία από Task 1.
    M, _, L_tr, _ = task1.get_M_N_Ltr_Lte()

    # Λίστα με clustering purity κάθε ανάλυσης με PCA.
    purities = []

    # ----- Για V = 2 ----- #
    V = 2

    # Μετασχηματισμός δεδομένων.
    M_cap = dimensionality_reduction_with_pca(M, V)

    # Δημιουργία scatter plot για μείωση διαστατικότητας με V = 2.
    plt.figure(0)
    visualize_plain_scatter(M_cap, L_tr)

    # Clustering με K-Means μετέπειτα από Maximin (υλοποιείται από task3 οπότε απλά χρησιμοποιείται από εκεί):
    results = task3.k_means(M_cap, 4)

    # Κλήση για οπτικοποίηση.
    plt.figure(1)
    task3.visualize_clustered_M_cap(results)

    purity = task3.clustering_purity(M_cap, L_tr, results)
    purities.append(purity)

    # Υπολογισμός και εκτύπωσης Purity των αποτελεσμάτων ομαδοποίησης.
    print('Purity with V = %d: '%V, purity)

    # ----- Για V = 25 ----- #
    V = 25

    # Μετασχηματισμός δεδομένων.
    M_cap = dimensionality_reduction_with_pca(M, V)

    # Clustering με K-Means μετέπειτα από Maximin (υλοποιείται από task3 οπότε απλά χρησιμοποιείται από εκεί):
    results = task3.k_means(M_cap, V)

    purity = task3.clustering_purity(M_cap, L_tr, results)
    purities.append(purity)

    # Υπολογισμός και εκτύπωσης Purity των αποτελεσμάτων ομαδοποίησης.
    print('Purity with V = %d: '%V, purity)

    # ----- Για V = 50 ----- #
    V = 50

    # Μετασχηματισμός δεδομένων.
    M_cap = dimensionality_reduction_with_pca(M, V)

    # Clustering με K-Means μετέπειτα από Maximin (υλοποιείται από task3 οπότε απλά χρησιμοποιείται από εκεί):
    results = task3.k_means(M_cap, V)

    purity = task3.clustering_purity(M_cap, L_tr, results)
    purities.append(purity)

    # Υπολογισμός και εκτύπωσης Purity των αποτελεσμάτων ομαδοποίησης.
    print('Purity with V = %d: '%V, purity)
    # ----- Για V = 100 ----- #
    V = 100

    # Μετασχηματισμός δεδομένων.
    M_cap = dimensionality_reduction_with_pca(M, V)

    # Clustering με K-Means μετέπειτα από Maximin (υλοποιείται από task3 οπότε απλά χρησιμοποιείται από εκεί):
    results = task3.k_means(M_cap, V)

    purity = task3.clustering_purity(M_cap, L_tr, results)
    purities.append(purity)

    # Υπολογισμός και εκτύπωσης Purity των αποτελεσμάτων ομαδοποίησης.
    print('Purity with V = %d: '%V, purity)

    print()
    print('Result: Vmax = ', [2, 25, 50, 100][np.argmax(purities)])


