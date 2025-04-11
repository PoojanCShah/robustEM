import numpy as np
from scipy.stats import multivariate_normal


def EM(X, c, eps=1e-4, maxIter=100):
    """
    Perform the EM algorithm on a dataset X for Gaussian Mixture Models.

    Parameters:
    X : numpy.ndarray of shape (n, d) - input data
    c : int - number of clusters
    eps : float - convergence threshold
    maxIter : int - maximum number of iterations

    Returns:
    alpha : mixture weights
    mu : means of Gaussians
    sigma : covariances of Gaussians
    z : responsibilities (c x n matrix)
    """

    n, d = X.shape

    # Initialization
    np.random.seed(0)
    
    # Randomly select the starting clusters
    initIndices = np.random.choice(n, size=c)
    mu = X[initIndices]
    z = np.ones((c, n)) / c  # Responsibilities (c x n)
    alpha = np.ones(c) / c 
    sigma = np.zeros((c, d, d))

    # Initial covariance
    for k in range(c): 
        diff = X - mu[k]
        sigma[k] = (diff.T @ diff) / n 
    
    for _ in range(maxIter):
        z_old = z.copy()

        # E-step: compute responsibilities
        for k in range(c):
            z[k, :] = alpha[k] * multivariate_normal.pdf(X, mean=mu[k], cov=sigma[k])

        z_sum = z.sum(axis=0, keepdims=True)  # sum along the clusters dimension
        z = z / z_sum  # Normalize the responsibilities

        # M-step: update parameters
        n_c = z.sum(axis=1)  # Sum of responsibilities for each cluster

        alpha = n_c / n
        mu = (z @ X) / n_c[:, None]  # Weighted mean for each cluster

        for k in range(c):
            diff = X - mu[k]
            sigma[k] = (z[k, :] * diff.T) @ diff / n_c[k]  # Weighted covariance for each cluster

        # Convergence check
        if np.linalg.norm(z - z_old) < eps:
            break

    return alpha, mu, sigma, z


def robustEM(X, eps = 1e-4, maxIter = 100) : 
    """
    Perform the robust EM algorithm on a dataset X for Gaussian Mixture Models.

    Parameters:
    X : numpy.ndarray of shape (n, d) - input data
    eps : float - convergence threshold 

    Returns:
    alpha : mixture weights
    mu : means of Gaussians
    sigma : covariances of Gaussians
    z : responsibilities (c x n matrix)
    """

    (n,d) = X.shape 

    #Step 1 : Initial parameters. In the start, all points are cluster centers

    c = n 
    beta = 1 
    mu = X 
    alpha = np.ones(c) / c 
    gamma = 1e-4
    


    # Step 2 : Compute the Initial Covariance Matrix
    # We need to compute distance of each centre from every other center
    norms = np.sum(X**2,axis = 1)
    D = norms[:,np.newaxis] + norms[np.newaxis,:] - 2*X@X.T
    D = np.sort(D, axis = 1)
    d_min = np.min(D)
    
    sigma = np.zeros((c,d,d))

    for k in range(c) : 
        sigma[k] = D[k][int(np.sqrt(c))] * np.identity(d)

    # Step 3 : Compute the initial responsibilities
    z = np.zeros((c,n))

    for k in range(c):
            z[k, :] = alpha[k] * multivariate_normal.pdf(X, mean=mu[k], cov=sigma[k])

    z_sum = z.sum(axis=0, keepdims=True)  # sum along the clusters dimension
    z = z / z_sum  # Normalize the responsibilities

    for iteration in range(maxIter) : 

        # Step 4 : Update the means
        n_c = z.sum(axis=1)  # Sum of responsibilities for each cluster
        mu = (z @ X) / n_c[:, None]  # Weighted mean for each cluster

        # Step 5 : Update the weights
        alpha_old = alpha
        alpha_EM = n_c / n # Old EM update 
        E = np.sum(alpha_old*np.log(alpha_old))

        alpha_new = alpha_EM + beta*(np.log(alpha_old) - E)
        alpha = alpha_new

        # Step 6 : Update the beta parameter
        eta = min(1,0.5**(d//2-1))

        alpha1_EM = np.max(alpha_EM)
        alpha1_old = np.max(alpha_old)

        beta = (1/c) * (np.sum(np.exp(-eta * n * abs(alpha_new - alpha_old))))
        beta = min(beta , (1 - alpha1_EM) / (-alpha1_old * E + 1e-6))

        # Step 7 : Discard appropriate centers

        c_new  = c 
        alpha_new = []
        z_new = []
        mu_new = []
        
        for k in range(c) : 
            if (alpha[k] > 1/n) : 
                alpha_new.append(alpha[k])
                z_new.append(z[k])
                mu_new.append(mu[k])
            else : 
                c_new -= 1

        z_new = np.array(z_new)
        alpha_new = np.array(alpha_new)

        z = z_new / np.sum(z_new, axis = 0, keepdims= True)
        alpha = alpha_new / np.sum(alpha_new, axis = 0, keepdims= True)
        mu = np.array(mu_new)

        if (iteration > 60 and (c == c_new)) : 
            beta = 0


        c = c_new

        

        # Step 8 : Perform the routine updates

        sigma = np.zeros((c,d,d))

        for k in range(c):
            diff = X - mu[k]
            sigma[k] = (z[k, :] * diff.T) @ diff / n_c[k]

            # Update so that the matrices are non singular
            sigma[k] = (1 - gamma)*sigma[k] + gamma * d_min * np.identity(d)

        for k in range(c):
            z[k, :] = alpha[k] * multivariate_normal.pdf(X, mean=mu[k], cov=sigma[k])

        z_sum = z.sum(axis=0, keepdims=True)  # sum along the clusters dimension
        z = z / z_sum  # Normalize the responsibilities

        n_c = z.sum(axis=1)  # Sum of responsibilities for each cluster
        mu_new = (z @ X) / n_c[:, None]

        if (np.linalg.norm(mu - mu_new) < eps): 
            break

    return alpha, mu, sigma, z






    
