import numpy as np
from scipy.linalg import schur, eigvals

expensive_asserts = True


def sort_real_schur(Q, R, z, b, inplace=False):
    '''
    :param Q: np.array((N, N))
        orthogonal real Q such that AQ=QR
    :param R: np.array((N, N))
        quasi-triangular real R such that AQ=QR
    :param z: complex
        target z in the complex plane
        if z==float('inf'), order eigenvalues decreasingly by magnitude
    :param b: float
        determines the length of the ordering with respect to z to be produced
        * if b < 0 then -b blocks will be sorted,
        * if b > 0 then  b or b+1 eigenvalues will be sorted, depending on the sizes of the blocks,
        * if b = 0 then the whole Schur form will be sorted.
    :return: Q, R, ap
        * Q, R : orthogonal real Q and quasi-triangular real R such that AQ=QR with
          the diagonal blocks ordered with respect to the target z. The number of
          ordered blocks/eigenvalues is determined by the parameter b.
        * A vector ap warns for inaccuracy of the solution if an entry of ap exceeds one.
    '''
    eps = np.finfo(R.dtype).eps
    if not np.all(np.abs(np.tril(R, -2)) <= 100*eps):
        raise ValueError('R is not block-triangular')
    if not inplace:
        Q = Q.copy()
        R = R.copy()

    # detect subdiagonal nonzero entries
    r = np.where(np.abs(np.diag(R, -1)) > 100*eps)[0]
    # construct from them a vector s with the-top left positions of each block
    s = [i for i in range(R.shape[0] + 1) if not i in r + 1]
    p = [None]*(len(s) - 1)  # will hold the eigenvalues

    for k in range(1, len(s) - 1):  # debug
        assert R[s[k], s[k] - 1] <= 100*eps  # debug

    for k in range(len(s) - 1):  # ranging over all blocks
        sk = s[k]
        if s[k + 1] - sk == 2:  # if the block is 2x2
            Q, R = normalize(Q, R, slice(
                sk, s[k + 1]), inplace=True)  # normalize it
            # store the eigenvalues
            p[k] = R[sk, sk] + np.lib.scimath.sqrt(R[sk + 1, sk]*R[sk, sk + 1])
        else:  # (the one with the positive imaginary part is sufficient)
            assert s[k + 1] - sk == 1  # debug
            p[k] = R[s[k], s[k]]  # if the block is 1x1, only store the eigenvalue

    ap = []

    for k in swaplist(p, s, z, b):  # For k ranging over all neighbor-swaps
        assert k + 2 < len(s)  # debug
        # collect the coordinates of the blocks
        v = list(range(s[k], s[k + 1]))
        w = list(range(s[k + 1], s[k + 2]))
        assert v[0] != w[0]  # debug
        if len(v) == 2:
            assert v[0] < v[1]  # debug
        if len(w) == 2:
            assert w[0] < w[1]  # debug
        # debug: check that we are moving the larger eigenvalues to the left (expensive test)
        if __debug__ and expensive_asserts:
            if v[0] < w[0]:  # debug
                arr = [p[k], p[k + 1]]  # debug
                _, which = select(arr, z)  # debug
                assert which == 1  # debug
            else:  # debug
                arr = [p[k + 1], p[k]]  # debug
                _, which = select(arr, z)  # debug
                assert which == 1  # debug
        vw = v + w
        # compute norm of the matrix A from (6)
        nrA = np.linalg.norm(R[vw, :][:, vw], ord=np.inf)
        Q, R = swap(Q, R, v, w, inplace=True)  # swap the blocks
        p[k], p[k + 1] = p[k + 1], p[k]  # debug
        s[k + 1] = s[k] + s[k + 2] - s[k + 1]  # update positions of blocks
        v = list(range(s[k], s[k + 1]))  # update block-coordinates
        w = list(range(s[k + 1], s[k + 2]))
        if len(v) == 2:  # if the first block is 2 x 2
            Q, R = normalize(Q, R, v, inplace=True)  # normalize it
        if len(w) == 2:  # if the second block is 2 x 2
            Q, R = normalize(Q, R, w, inplace=True)  # normalize it
        # measure size of bottom-left block (see p.6, Sect. 2.3)
        ap.append(np.linalg.norm(R[w, :][:, v], ord=np.inf) / (10*eps*nrA))

    R = R - np.tril(R, -2)  # Zero the below-block entries
    for k in range(1, len(s)-1):  # to get a quasi-triangle again
        R[s[k], s[k]-1] = 0

    return Q, R, ap


def normalize(U, S, v, inplace=False):
    r'''Applies a Givens rotation such that the two-by-two diagonal block of S situated at diagonal positions v[0], v[1] is in standardized form.
    :param U:
    :param S:
    :param v:
    :return:
    '''
    Q = rot(S[v, :][:, v]
            )  # Determine the Givens rotation needed for standardization -
    if not inplace:
        S = S.copy()
        U = U.copy()
    # and apply it left and right to S, and right to U.
    S[:, v] = np.dot(S[:, v], Q)
    # Only rows and columns with indices in the vector v can be affected by this.
    S[v, :] = np.dot(Q.T, S[v, :])
    U[:, v] = np.dot(U[:, v], Q)
    return U, S


def rot(X):
    r'''Computes a Givens rotation needed in `normalize`
    :param X:
    :return:
    '''
    c = 1.0  # Start with the identity transformation, and if needed, change it into ...
    s = 0.0
    if X[0, 0] != X[1, 1]:
        tau = (X[0, 1] + X[1, 0]) / (X[0, 0] - X[1, 1])
        off = (tau**2 + 1)**0.5
        v = [tau - off, tau + off]
        w = np.argmin(np.abs(v))
        # ... the cosine and sine as given in Section 2.3.1
        c = 1.0/(1.0 + v[w]**2)**0.5
        s = v[w]*c
    Q = np.array([[c, -s], [s, c]], dtype=X.dtype)
    return Q


def swaplist(p, s, z, b):
    r'''Produces the list v of swaps of neighboring blocks needed to order the eigenvalues assembled in the vector v
     from closest to z to farthest away from z, taking into account the parameter b.
    :param p: list
        list of eigenvalues (only one copy for each complex-conjugate pair)
    :param s: list
        list of the the-top left positions of each block
    :param z: complex
        target z in the complex plane
    :param b: float
        determines the length of the ordering with respect to z to be produced
        * if b < 0 then -b blocks will be sorted,
        * if b > 0 then  b or b+1 eigenvalues will be sorted, depending on the sizes of the blocks,
        * if b = 0 then the whole Schur form will be sorted.
    :return:
    '''
    p_orig = p  # debug
    n = len(p)
    p = list(p)
    k = 0
    v = []
    srtd = 0  # Number of sorted eigenvalues.
    q = list(np.diff(s))  # Compute block sizes.
    q_orig = list(q)  # debug
    fini = False
    while not fini:
        _, j = select(p[k:n], z)  # Determine which block will go to position k
        p_j = p[k + j]  # debug
        p[k:n + 1] = [p[j + k]] + p[k:n]  # insert this block at position k,
        assert p[k] == p_j  # debug
        del p[j + k + 1]  # and remove it from where it was taken.
        if expensive_asserts and __debug__:
            assert np.all(sorted(p) == sorted(p_orig))  # debug
        q_j = q[k + j]  # debug
        q[k:n + 1] = [q[j + k]] + q[k:n]  # Similar for the block-sizes
        assert q[k] == q_j  # debug
        del q[j + k + 1]
        if expensive_asserts and __debug__:
            assert np.all(sorted(q) == sorted(q_orig))  # debug
        # Update the list of swaps for this block
        v = v + list(range(k, j + k))[::-1]
        srtd = srtd + q[k]  # Update the number of sorted eigenvalues
        k += 1
        fini = (k >= n - 1 or k == -b or srtd ==
                b or (srtd == b + 1 and b != 0))
    return v


def select(p, z):
    r'''Determined which element is next in the ordering.
    :param p:
    :param z:
    :return:
    '''
    if np.isinf(z):
        pos = np.argmax(np.abs(p))
        return -abs(p[pos]), pos
    else:
        # Move target to the upper half plane.
        y = np.real(z) + np.abs(np.imag(z))*1j
        delta = np.abs(np.array(p) - y)
        pos = np.argmin(delta)  # Find block closest to the target.
        return delta[pos], pos


def swap(U, S, v, w, inplace=False):
    r'''Swaps the two diagonal blocks at positions symbolized by the entries of v and w.
    :param U:
    :param S:
    :param v:
    :param w:
    :return:
    '''
    p, q = S[v, :][:, w].shape  # p and q are block sizes
    Ip = np.eye(p)
    Iq = np.eye(q)
    # Vectorize right-hand side for Kronecker product formulation of the Sylvester equations (7).
    r = np.concatenate([S[v, w[j]] for j in range(q)])
    # Kronecker product system matrix.
    K = np.kron(Iq, S[v, :][:, v]) - np.kron(S[w, :][:, w].T, Ip)
    # LU-decomposition of this matrix.
    L, H, P, Q = lu_complpiv(K, overwrite=True)
    e = np.min(np.abs(np.diag(H)))  # Scaling factor to prevent overflow.
    sigp = np.arange(p*q)
    # Implement permutation P of the LU-decomposition PAQ=LU ...
    for k in range(p*q - 1):
        sigp[[k, P[k]]] = sigp[[P[k], k]].copy()
    r = e*r[sigp]  # ... scale and permute the right-hand side.
    # and solve the two triangular systems.
    x = np.linalg.solve(H, np.linalg.solve(L, r))
    sigq = np.arange(p*q)
    # Implement permutation Q of the LU-decomposition PAQ=LU ...
    for k in range(p*q - 1):
        sigq[[k, Q[k]]] = sigq[[Q[k], k]].copy()
    x[sigq] = x.copy()  # ... and permute the solution.
    # De-vectorize the solution back to a block, or, quit Kronecker formulation.
    X = np.vstack([x[j*p:(j + 1)*p] for j in range(q)]).T
    # Householder QR-decomposition of X.
    Q, R = np.linalg.qr(np.vstack((-X, e*Iq)), mode='complete')
    vw = list(v) + list(w)
    if not inplace:
        S = S.copy()
        U = U.copy()
    # Perform the actual swap by left- and right-multiplication of S by Q,
    S[:, vw] = np.dot(S[:, vw], Q)
    S[vw, :] = np.dot(Q.T, S[vw, :])
    U[:, vw] = np.dot(U[:, vw], Q)  # and, right-multiplication of U by Q
    return U, S


def lu_complpiv(A, overwrite=False):
    '''Computes the LU-decomposition of A with complete pivoting, i. e. PAQ=LU, with permutations symbolized by vectors.
    :param A:
    :return:
    '''
    if not overwrite or (__debug__ and expensive_asserts):
        A_inp = A  # debug
        A = A.copy()
    n = A.shape[0]
    P = np.zeros(n - 1, dtype=int)
    Q = np.zeros(n - 1, dtype=int)
    # See Golub and Van Loan, p. 118 for comments on this LU-decomposition with complete pivoting.
    for k in range(n - 1):
        Ak = A[k:n, :][:, k:n]
        rw, cl = np.unravel_index(np.argmax(np.abs(Ak), axis=None), Ak.shape)
        rw += k
        cl += k
        A[[k, rw], :] = A[[rw, k], :].copy()
        A[:, [k, cl]] = A[:, [cl,  k]].copy()
        P[k] = rw
        Q[k] = cl
        if A[k, k] != 0:
            rs = slice(k + 1, n)
            A[rs, k] = A[rs, k] / A[k, k]
            A[rs, :][:, rs] = A[rs, :][:, rs] - \
                A[rs, k][:, np.newaxis]*A[k, rs]
    U = np.tril(A.T).T
    L = np.tril(A, -1) + np.eye(n)
    if __debug__ and expensive_asserts:
        perm_p = np.arange(n)  # debug
        for k in range(n - 1):  # debug
            perm_p[[k, P[k]]] = perm_p[[P[k], k]].copy()  # debug
        perm_q = np.arange(n)  # debug
        for k in range(n - 1):  # debug
            perm_q[[k, Q[k]]] = perm_q[[Q[k], k]].copy()  # debug
        assert np.allclose(A_inp[perm_p, :][:, perm_q], np.dot(L, U))  # debug
    return L, U, P, Q


def truncateKrylov(Q, H, k, m):
    Q = np.concatenate((Q[:, :k], Q[:, m:m+1]), axis=1)
    H = np.concatenate((H[:k, :k], H[m:m+1, :k]), axis=0)

    return Q, H


def testConverge(H, kk, ii, tol):
    '''
    test the convergence of the ith eigenvalue in Krylov-Schur iteration
    H is the hessenberg matrix after truncation
    The test rule is 
        | b_i | < max( ||H(1:k, 1:k)||_F * epsilon, tol * | \lambda_i | ) 

    Parameters:
        i          the ith position of vector b = H(k+1, 1:k)
        k          the size of truncated H is (k+1) x k
        epsilon    machine precesion
        tol        tolerance passed by user
        ||*||_F    Frobeneous norm
        \lambda_i  the ith eigenvalue of H(1:k, 1:k). 
                    1x1 block => \lambda_i = H(i, i)
                    2x2 block => \lambda_i = eig( H(i:i+1, i:i+1))
                    Here it is not possible for H(i-1:i, i-1:i) because
                    whenever we find index i converges and it
                    corresponds to a pair of complex eigenvalues,
                    then we also mark i+1 converges.
    Return:
        flag       -1 : not converge 
                    1 : real eigenvalue converges
                    2 : complex eigenvalue pair converges
    ''' 
    i = ii-1

    k = kk-1
    epsilon = 2e-16
    
    if i < k:
        delta = (H[i, i]-H[i+1, i+1])**2+4*H[i+1, i]*H[i, i+1]
    else:
        delta = 1

    if delta > 0:
        if np.abs(H[k+1, i]) < np.max([np.linalg.norm(H, 'fro')*epsilon , np.abs(H[i, i]*tol)]):
            flag = 1
        else:
            flag = -1
    else:
        gamma = complex(H[i, i] + H[i+1, i+1], np.sqrt(-delta))/2

        if np.abs(H[k+1, i]) < np.max([np.linalg.norm(H, 'fro') * epsilon, np.abs(gamma) * tol]):
            flag = 2
        else:
            flag = -1

    return flag


def sortSchur(A, kk):
    """
    perform Schur decompostion on A and put the needed k eigenvalues on the upper left block.
    Paramters:
        A        a square matrix
        k        number of eigenvalues that will be reordered
    
    Return:
        US       othormormal matrix
        TS       quasi upper triangular matrix
        isC      isC = 0, the kth eigenvalue is real. 
                isC = 1, the kth and (k+1)th eigenvalues are complex
    
    Note:
        This function assumes that eigenvalues are reordered by their 
        magnitudes. You can change this functions accordingly to obtain
        eigenvalues that you need.
    """
    k = kk-1
    T, U = schur(A, 'real')
    es = eigvals(T)
    esp, ix = -np.sort(-abs(es)), np.argsort(-abs(es))
    k1 = ix[k]
    k2 = ix[k+1]
    delta = (T[k1, k1]-T[k2, k2])**2+4*T[k2, k1]*T[k1, k2]
    if (k2-k1 == 1) and (delta < 0):
        isC = 1
    else:
        isC = 0

    US, TS, ap = sort_real_schur(U, T, float('inf'), k+1)

    return US, TS, isC


def Ax(A, x):
    return A.dot(x)


def expandKrylov(A, Q, H, skk, ekk):

    """
     expand Krylov subspace.
       The function contruct the sk+1, sk+2, ..., ek_th column of Q.
       A * Q(:, 1:ek) = Q(:, 1:ek+1) * H
    Parameters:
        sk       start index
        ek       end index
    Return:
        Q        the expanded orthornormal matrix with dimension [n x ek+1]
        H        dimension [ek+1 x ek], the upper [ek x ek] block is Hessenberg
    """
    sk = skk-1
    ek = ekk-1
    for k in range(sk+1, ek+1):
        v = Ax(A, Q[:, k:k+1])
        w = Q[:, :k+1].T.dot(v)
        v = v-Q[:, :k+1].dot(w)
        w2 = Q[:, :k+1].T.dot(v)
        v = v-Q[:, :k+1].dot(w2)
        w = w+w2
        nv = np.linalg.norm(v)
        if Q.shape[1] <= k+2:
            Q = np.concatenate((Q, np.zeros((Q.shape[0], 1))), 1)
        Q[:, k+1:k+2] = (v/nv)
        if H.shape[0] < k+2:
            H = np.concatenate((H, np.zeros((H.shape[0], 1))), 1)
            H = np.concatenate((H, np.zeros((1, H.shape[1]))), 0)
        H[:k+2, k:k+1] = np.concatenate((w, np.array([[nv]])), 0)
    return Q, H


def KrylovSchur(A, v1, n, k, m, maxIt, tol):
    '''
    conduct Krylov-Schur decomposition to find invariant subspace of matrix A.
    A * Q(:, 1:k+isC) = Q(:, 1:k+isC) * H(1:k+isC, 1:k+isC)
    Paramters:
        A        nXn Matrix
        v1       initial vector
        n        size of matrix A
        k        number of eigenvalues needed
        m        Krylov restart threshold (largest dimension of krylov subspace)
        maxIt    maximum iteration number
        tol      convergence tolerance
    
    Return:
        Q        orthonormal matrix with dimension [n x k+1] or [n x k+2]
        H        `Hessenberg' matrix with dimension [k+1 x k] or [k+2 x k+1]
        isC      isC = 0, the kth eigenvalue is real. 
                    isC = 1, the kth and (k+1)th eigenvalues are complex 
                    conjugate pair, so Q and H has one more dimension
        flag      flag = 0, converge.
                  flag = 1, not converge
        nc        number of converged eigenvalues 
        ni        number of iteration used (each whole expansion stage counts 1)
    '''
    
    Q = np.zeros((n, m+1),dtype = 'complex_')
    H = np.zeros((m+1, m),dtype = 'complex_')
    Q[:, 0] = v1/np.linalg.norm(v1)
    p = 0
    isC = 0

    Q, H = expandKrylov(A, Q, H, 0, k)
    i = 0
    while (i < maxIt) & (p <= k):
        i = i+1
        Q, H = expandKrylov(A, Q, H, k+isC, m)
        U, T, isC = sortSchur(H[p:m, p:m], k-p)
        H[p:m, p:m] = T
        if p-1 >= 0:
            H[0:p-1, p:m] = H[0:p-1, p:m].dot(U)

        Q[:, p:m] = Q[:, p:m].dot(U)
        H[m, p:m] = H[m, m-1]*(U[-1, :])
        Q, H = truncateKrylov(Q, H, k+isC, m)
        check = True
        while check:
            result = testConverge(H, k+isC, p, tol)
            if (result == 1) or (result == 2):
                p = p+result
                if p > k:
                    check = False
            else:
                check = False

    ni = i
    if p > k:
        flag = 0
        nc = k+isC
    else:
        flag = 1
        nc = p-1
    return Q , H, isC, flag, nc, ni

    
