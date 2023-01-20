import numpy as np

A = np.array([[1, 1, -1], [1, 3, 1], [-1, 1, 3]])
eigvals = np.linalg.eigvalsh(A)

def minor(arr,i,j):
    # ith row, jth column removed
    return arr[np.array(list(range(i))+list(range(i+1,arr.shape[0])))[:,np.newaxis],
               np.array(list(range(j))+list(range(j+1,arr.shape[1])))]

for eigval in eigvals:
    eigvec_mod_squares = np.array([])
    
    denominator = 1
    for eigval2 in eigvals:
        if eigval != eigval2:
            denominator *= (eigval - eigval2)

    for j in range(len(A)):
        M = minor(A, j, j)
        M_eigvals = np.linalg.eigvalsh(M)
        
        numerator = 1
        for m_eigval in M_eigvals:
            numerator *= (eigval - m_eigval)
        
        term = numerator/denominator
        eigvec_mod_squares = np.append(eigvec_mod_squares, term)
        # eigvec_mod_squares.append(term)
    
    print("Eigenvalue:", eigval)
    print("Eigenvector mod squares:", eigvec_mod_squares)
    print("=" * 20)