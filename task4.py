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

    # Υπολογισμός του πίνακα συνδιακύμανσης.
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
def transform(data_matrix: np.ndarray, V: int = None, base=None):
    """
    Εφαρμογή μείωσης διαστάσεων με τη χρήση PCA.
    :param data_matrix: Πίνακας N x D με δεδομένα όπου N είναι ο αριθμός των δειγμάτων και D ο αριθμός διαστάσεων κάθε
                        δείγματος (οι γραμμές περιέχουν D-διάστατα διανύσματα που αποτελούν εικόνες και οι στήλες
                        περιέχουν τα features (στην περίπτωση αυτή τα pixel των εικόνων) των δεδομένων).
    :param V: Ο νέος αριθμός διαστάσεων. Τιμή None σηματοδοτεί τη χρήση δοθέντος διανύσματος βάσης base (αν λείπει και
    το base, τότε θεωρείται V = 2).
    :param base: Η βάση μετασχηματισμού (που προήλθε από τη συνάρτηση pca ή κάποια με αντίστοιχη έξοδο). Τιμή None
    σηματοδοτεί τη δημιουργία νέου διανύσματος βάσης από μετασχηματισμό PCA.
    :return: Ο μετασχηματισμένος πίνακας N x V με νέες διαστάσεις.
    """

    # Άν έχει δοθεί νέα βάση, τότε πρέπει να χρησιμοποιήσω αυτή. Διαφορετικά υπολογίζω νέα βάσει δοθέν V.
    if base is None:
        # "Aν λείπει και το base, τότε θεωρείται V = 2"
        if V is None:
            V = 2

        # Υπολογισμός βάσης προς προβολή.
        base = pca(data_matrix, V)

    # Υπολογισμός προβολής δειγμάτων του X (data_matrix) προς τη νέα βάση εκτελώντας μείωση διαστατικότητας.
    return np.matmul(base.T, data_matrix.T).T


def visualize_plain_scatter(M, L_tr, title=""):
    # Αντιστοίχιση από αρίθμηση κλάσεων με ακεραίους σε χρώματα.
    colors = L_tr
    cmap = ListedColormap(['r', 'r', 'g', 'r', 'r', 'r', 'b', 'r', 'm'])

    # -- Δημιουργία scatter plot --
    scatter = plt.scatter(M[:, 0], M[:, 1], c=colors, cmap=cmap, alpha=0.02)

    # Δημιουργία legend.
    legend = plt.legend(*scatter.legend_elements())
    for legend_handle in legend.legendHandles:
        legend_handle.set_alpha(1)

    plt.title(title)
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
    M_approx = transform(M, V=V)

    # Δημιουργία scatter plot για μείωση διαστατικότητας με V = 2.
    plt.figure(0)
    visualize_plain_scatter(M_approx, L_tr, r"$\tilde{\mathbf{M}}$ visualization")

    # Clustering με K-Means μετέπειτα από Maximin (υλοποιείται από task3 οπότε απλά χρησιμοποιείται από εκεί):
    results = task3.k_means(M_approx, 4)

    # Κλήση για οπτικοποίηση.
    plt.figure(1)
    task3.visualize_clustering(results, r"$\tilde{\mathbf{M}}$ clustering visualization")

    purity = task3.clustering_purity(M_approx, L_tr, results)
    purities.append(purity)

    # Υπολογισμός και εκτύπωσης Purity των αποτελεσμάτων ομαδοποίησης.
    print('Purity with V = %d: ' % V, purity)

    # ----- Για V = 25 ----- #
    V = 25

    # Μετασχηματισμός δεδομένων.
    M_approx = transform(M, V=V)

    # Clustering με K-Means μετέπειτα από Maximin (υλοποιείται από task3 οπότε απλά χρησιμοποιείται από εκεί):
    results = task3.k_means(M_approx, V)

    purity = task3.clustering_purity(M_approx, L_tr, results)
    purities.append(purity)

    # Υπολογισμός και εκτύπωσης Purity των αποτελεσμάτων ομαδοποίησης.
    print('Purity with V = %d: ' % V, purity)

    # ----- Για V = 50 ----- #
    V = 50

    # Μετασχηματισμός δεδομένων.
    M_approx = transform(M, V=V)

    # Clustering με K-Means μετέπειτα από Maximin (υλοποιείται από task3 οπότε απλά χρησιμοποιείται από εκεί):
    results = task3.k_means(M_approx, V)

    purity = task3.clustering_purity(M_approx, L_tr, results)
    purities.append(purity)

    # Υπολογισμός και εκτύπωσης Purity των αποτελεσμάτων ομαδοποίησης.
    print('Purity with V = %d: ' % V, purity)
    # ----- Για V = 100 ----- #
    V = 100

    # Μετασχηματισμός δεδομένων.
    M_approx = transform(M, V=V)

    # Clustering με K-Means μετέπειτα από Maximin (υλοποιείται από task3 οπότε απλά χρησιμοποιείται από εκεί):
    results = task3.k_means(M_approx, V)

    purity = task3.clustering_purity(M_approx, L_tr, results)
    purities.append(purity)

    # Υπολογισμός και εκτύπωσης Purity των αποτελεσμάτων ομαδοποίησης.
    print('Purity with V = %d: ' % V, purity)

    print()
    print('Result: Vmax = ', [2, 25, 50, 100][np.argmax(purities)])
