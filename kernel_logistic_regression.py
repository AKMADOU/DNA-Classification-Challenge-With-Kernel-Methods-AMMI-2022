from kernel_method_base import*
from kernel_ridge_regression import*


def sigmoid(x):
    # return 1 / (1 + np.exp(-x))
# tanh version helps avoid overflow problems
    return .5 * (1 + np.tanh(.5 * x))

class KernelLogisticRegression(KernelMethodBase):
    '''
    Kernel Logistic Regression
    '''
    def __init__(self, lambd=0.001, **kwargs):

        self.lambd = lambd
        super().__init__(**kwargs)
        
    
    def fit_K(self, K, y, method='gradient', lr=0.1, max_iter=500, tol=1e-12):
        '''
        Find the dual variables alpha
        '''
        if method == 'gradient':
            self.fit_alpha_gradient_(K, y, lr=lr, max_iter=max_iter, tol=tol)
        elif method == 'newton':
            self.fit_alpha_newton_(K, y, max_iter=max_iter, tol=tol)
            
        return self
        
    def fit_alpha_gradient_(self, K, y, lr=0.01, max_iter=500, tol=1e-6):
        '''
        Finds the alpha of logistic regression by gradient descent
        
        lr: learning rate
        max_iter: Max number of iterations
        tol: Tolerance wrt. optimal solution
        '''
        n = K.shape[0]
        # Initialize
        alpha = np.zeros(n)
        # Iterate until convergence or max iterations
        for n_iter in range(max_iter):
            alpha_old = alpha
            M = y*sigmoid(-y*K@alpha)
            gradient = -(1/n) *K@M +2*self.lambd*K@alpha
            alpha = alpha_old - lr * gradient
            # Break condition (achieved convergence)
            if np.sum((alpha-alpha_old)**2) < tol**2:
                break
        self.n_iter = n_iter
        self.alpha = alpha

    def fit_alpha_newton_(self, K, y, max_iter=500, tol=1e-6):
        '''
        Finds the alpha of logistic regression by the Newton-Raphson method
        and Iterated Least Squares
        '''
        n = K.shape[0]
        # IRLS
        KRR = KernelRidgeRegression(lambd=2*self.lambd)
        # Initialize
        alpha = np.zeros(n)
        # Iterate until convergence or max iterations
        for n_iter in range(max_iter):
            alpha_old = alpha
            m = K.dot(alpha_old)
            w = sigmoid(m) * sigmoid(-m)
            z = m + y / sigmoid(y * m)
            alpha = KRR.fit_K(K, z, sample_weights=w).alpha
            # Break condition (achieved convergence)
            if np.sum((alpha-alpha_old)**2) < tol**2:
                break
        self.n_iter = n_iter
        self.alpha = alpha
        
    def decision_function_K(self, K_x):
        # print('K', K_x.shape, 'alpha', self.alpha.shape)
        return sigmoid(K_x@self.alpha)
        
    def predict(self, X):
        return np.sign(2*self.decision_function(X)-1)