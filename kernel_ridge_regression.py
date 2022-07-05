import numpy as np
from kernel_method_base import*

lambd = .0001

class KernelRidgeRegression(KernelMethodBase):
    '''
    Kernel Ridge Regression
    '''
    def __init__(self, lambd=0.0001, **kwargs):
        self.lambd = lambd
        # Python 3: replace the following line by
        # super().__init__(**kwargs)
        super(KernelRidgeRegression, self).__init__(**kwargs)

    def fit_K(self, K, y, sample_weights=None):
        n = K.shape[0]
        assert (n == len(y))
        
        w_sqrt = np.ones_like(y) if sample_weights is None else sample_weights
        w_sqrt = np.sqrt(w_sqrt)

        # Rescale kernel matrix K to take weights into account
        A = K * w_sqrt * w_sqrt[:, None]
        # Add lambda to the diagonal of A (= add lambda*I):
        A += n * lambd * np.eye(n)
        # self.alpha = (K + n * lambda I)^-1 y
        self.alpha = w_sqrt * np.linalg.solve(A , y * w_sqrt)

        return self
    
    def decision_function_K(self, K_x):
        return self.alpha@K_x.T #K_x.dot(self.alpha)
    
    def predict(self, X):
        return self.decision_function(X)
    
    def predict_K(self, K_x):
        return self.decision_function_K(K_x)