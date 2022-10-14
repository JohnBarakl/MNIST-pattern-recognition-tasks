import task1, task2, task3, task4

import numpy as np


def train_naive_bayes(data_matrix: np.ndarray, L_tr: np.ndarray):
    # Ψευδώνυμο του data_matrix για ευκολότερη διαχείριση και ταύτιση με τους τύπους θεωρίας.
    # Το x_ik που αναφέρεται παρακάτω αντιστοιχεί στο X[i][k].
    X = data_matrix

    # Υποθέτω πως οι κατανομές των κλάσεων C_jk των ψηφίων j = 1, 2, 7, 9 και pixel k = 1, 2, ..., 50 είναι κανονικές
    #   με μέση τιμή m_jk και διασπορά s_jk ^ 2.
    # Επομένως, αρκεί να βρούμε τα m και s για κάθε μία κατανομή ώστε να μπορέσει στην πορεία να υπολογιστούν οι πιθανότητες
    #   P(x_ik|C_k) = (1 / (2πσ_jk^2)^(1/2) ) * exp{-(1/2σ_jk^2)(x_ik-m_jk)^2} ([ΑΝΑΓΝΩΡΙΣΗ ΠΡΟΤΥΠΩΝ ΚΑΙ ΜΗΧΑΝΙΚΗ ΜΑΘΗΣΗ, 2η έκδοση,
    #   Christopher M. Bishop], σελίδα 97).
    # Τα m_jk και s_jk ^ 2 κάθε κλάσης C_k μπορούν να εκτιμηθούν βάσει μέγιστης πιθανοφάνειας από τους τύπους
    # (προσαρμοσμένοι σε διάσταση D = 1 από [ΑΝΑΓΝΩΡΙΣΗ ΠΡΟΤΥΠΩΝ ΚΑΙ ΜΗΧΑΝΙΚΗ ΜΑΘΗΣΗ, 2η έκδοση, Christopher M. Bishop],
    # σελίδες 112-113):
    #   m_jk = 1/N * Σ_{i=1}^{N} (x_ik), N σύνολο σημείων x,
    #   s_jk^2 = 1/(N-1) * Σ_{i=1}^{N} ((x_ik - m_jk)^2),  N σύνολο σημείων x.

    N_total = len(X)  # Συντομογραφία του πλήθους στοιχείων (συνολικά).
    D = X.shape[1]  # Συντομογραφία του πλήθους διαστάσεων του x_i.

    # Συντομογραφίες "δεικτών" κάθε κλάσης.
    ONE = 0
    THREE = 1
    SEVEN = 2
    NINE = 3

    # Διαχωρισμός του συνόλου μάθησης σε 4 λίστες ανάλογα με την κλάση στην οποία ανήκουν.
    num_class = [np.array([]) for i in range(4)]
    num_class[ONE] = X[L_tr == 1]
    num_class[THREE] = X[L_tr == 3]
    num_class[SEVEN] = X[L_tr == 7]
    num_class[NINE] = X[L_tr == 9]

    # Συντομογραφία του πλήθους στοιχείων (ανα κλάση).
    N = [len(num_class[i]) for i in range(4)]

    # Υπολογίζω τις apriori πιθανότητες των κλάσεων.
    apriori_P_c = [N[i] / N_total for i in range(4)]

    # -------------------------------------------- Εύρεση Κατανομών -------------------------------------------- #

    # Εκτίμηση μέσων τιμών.
    m = []
    for k in range(4):
        m.append([sum([num_class[k][i][j] for i in range(N[k])]) / N[k] for j in range(D)])  # Σχέση 2.122.

    # Εκτίμηση διασπορών.
    s_squared = []
    for k in range(4):
        s_squared.append([sum([(num_class[k][i][j] - m[k][j]) ** 2 for i in range(N[k])]) / (N[k] - 1) for j in
                          range(D)])  # Σχέση 2.125.

    def P(event, evidence_normal):
        """
        Υπολογίζεται η πιθανότητα P(event|norm), όπου event η μαρτυρία και norm η κανονική κατανομή με μέση τιμή
        μ = evidence_normal[0] και διασπορά σ^2 = evidence_normal[1].

        Θα χρησιμοποιηθεί για να υπολογίζεται η πιθανότητα P(x_i | C_j) όπου x_i είναι κάποιο χαρακτηριστικό κάποιου δείγματος
        του συνόλου δεδομένων εκπαίδευσης και C_j η κλάση στην οποία υπολογίζουμε την πιθανότητα να βρίσκεται το x.
        :param event: Το γεγονός για το οποίο υπολογίζουμε την πιθανότητα, είναι ένας πραγματικός αριθμός.
        :param evidence_normal: Ζεύγος μέσης τιμής και διασποράς κανονικής κατανομής για την οποία υπολογίζουμε την πιθανότητα
        να ανήκει το γεγονός.
        :return: Η πιθανότητα P(event|norm).
        """

        # Ψευδώνυμα για ευκολία.
        m, s2 = evidence_normal[:2]
        x = event

        return np.exp(-((x - m) ** 2) / (2 * s2)) / np.sqrt(2 * np.pi * s2)

    # -------------------------------------------- Υλοποίηση Naive Bayes -------------------------------------------- #

    # Ο απλοϊκός ταξινομητής bayes (Naive Bayes) ταξινομεί ένα άγνωστο δείγμα x στην κλάση C_m βάσει μέγιστης πιθανοφάνειας όπου:
    #   C_m = argmax_C_j{ Π_{i = 1}^{l} P(x_i | C_j) }, ([Αναγνώριση Προτύπων, S. Theodoridis, K. Koutroubas, 4η έκδοση],
    #   σελίδα 70).

    def gausian_naive_bayes_classification(x):
        """
        Προβλέπει την κλάση στην οποία ανήκει το σημείο x (δηλαδή προβλέπει το ψηφίο (1 ή 3 ή 7 ή 9)
        που απεικονίζεται σε μία εικόνα).
        :param x: Μία εικόνα ίδιας διάστασης με το σύνολο εικόνων του συνόλου εκπαίδευσης.
        :return: Την κλάση που (είναι πιο πιθανό) να ανήκει η εικόνα (1 ή 3 ή 7 ή 9).
        """

        # Αντιστοίχηση μετρητών σε κλάσεις (π.χ. ώστε ο δείκτης a[0] να αντιστοιχεί στο a της κλάσης 1).
        class_labels = (1, 3, 7, 9)

        probability_for_each_class = []
        for i in range(4):
            # Χρησιμοποιείται log για να αντιμετωπιστούν (πιθανώς) πολύ μικρές τιμές.
            # Η apriori πιθανότητα των κλάσεων θεωρείται ισοπίθανη (ανεξαρτήτως του πραγματικού πλήθους δοθέντων στοιχείων
            # ανα κλάση) και επομένως απαλείφεται (εφόσον θα προβούμε σε σύγκριση ένας ίσος όρος μπορεί να απαλειφθεί).
            pr = np.sum(np.log([P(x[j], (m[i][j], s_squared[i][j])) for j in range(len(x))]))

            probability_for_each_class.append(pr)

        return class_labels[np.argmax(probability_for_each_class)]

    return gausian_naive_bayes_classification


# Υλοποιεί τα ζητούμενα της εκφώνησης
if __name__ == '__main__':
    M, N, L_tr, L_te = task1.get_M_N_Ltr_Lte()

    # Μετασχηματισμός δεδομένων.
    transformation_base = task4.pca(M, 50)
    M_approx = task4.transform(M, base=transformation_base)
    N_approx = task4.transform(N, base=transformation_base)

    # Εκπαίδευση μοντέλου.
    clf = train_naive_bayes(M_approx, L_tr)

    # Υπολογισμός και εμφάνιση ακρίβειας.
    print("Test set classification accuracy:", sum([1 if clf(N_approx[i]) == L_te[i] else 0 for i in range(len(L_te))]) / len(L_te))
