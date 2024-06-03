# Made by: Tobias Schønau s224327
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from scipy.stats import mode
from copy import copy
from sklearn.ensemble import RandomForestClassifier




class MyLinearRegression:
    def __init__(self):
        self.beta = None

    def fit(self, X, y):
        # Add a column of ones to X for the intercept term
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        # Calculate the parameters beta
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        # Add a column of ones to X for the intercept term
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        # Make predictions
        y_pred = X @ self.beta
        return y_pred
    
class MyRidgeRegression:
    def __init__(self, lambda_=1.0):
        self.lambda_ = lambda_
        self.beta = None
    
    def fit(self, X, y):
        # Add a column of ones to X for the intercept term
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        # Calculate the parameters beta
        I = np.eye(X.shape[1])
        self.beta = np.linalg.inv(X.T @ X + self.lambda_ * I) @ X.T @ y
    
    def predict(self, X):
        # Add a column of ones to X for the intercept term
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        # Make predictions
        y_pred = X @ self.beta
        return y_pred

class MyLinearClassification(MyLinearRegression):
    def predict_proba(self, X):
        # Add a column of ones to X for the intercept term
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        # Make predictions
        y_pred = X @ self.beta
        return y_pred.clip(min=0, max=1)

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) > threshold
    
class MyLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.beta = None
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        # Add a column of ones for the bias term
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        num_samples, num_features = X.shape
        
        # Initialize beta
        self.beta = np.ones(num_features)
        
        # Gradient descent
        for _ in range(self.num_iterations):
            # Compute prediction probability
            p = self._sigmoid(np.dot(X, self.beta))
            
            # Compute gradients
            dw = np.dot(X.T, (y-p))
            
            # Update beta
            self.beta = self.beta + self.learning_rate * dw
    
    def predict_proba(self, X):
        # Add a column of ones for the bias term
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        linear_model = np.dot(X, self.beta)
        return self._sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return probabilities > threshold

class MyPerceptron:
    def __init__(self, learning_rate=0.5, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.W = None # Corresponds to vec{W}
        self.c = None # Corresponds to vec{c}
        self.w = None # Corresponds to bm{w}
        self.b = None # Corresponds to b
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        y = y.reshape(-1, 1) # Ensure y is a column vector
        num_samples, num_features = X.shape
        
        # Initialize weights and biases
        self.W = np.random.rand(num_features, 2) # Corresponds to vec{W}
        self.c = np.random.rand(2, 1) # Corresponds to vec{c}
        self.w = np.random.rand(2, 1) # Corresponds to bm{w}
        self.b = np.random.rand(1, 1) # Corresponds to b
        
        for _ in range(self.num_iterations):
            # Forward propagation
            layer1 = self._sigmoid(np.dot(X, self.W) + self.c.T)
            output = self._sigmoid(np.dot(layer1, self.w) + self.b)
            
            # Compute derivative
            derivative = 2*(y - output) * output * (1 - output)
            
            # Backpropagation
            d_w = np.dot(layer1.T, derivative)
            d_W = np.dot(X.T, (np.dot(derivative, self.w.T) * layer1 * (1 - layer1)))
            
            # Update the weights and biases
            self.W += self.learning_rate * d_W
            self.w += self.learning_rate * d_w
    
    def predict(self, X):
        layer1 = self._sigmoid(np.dot(X, self.W) + self.c.T)
        output = self._sigmoid(np.dot(layer1, self.w) + self.b)
        return output > 0.5

class MyKMeans:
    def __init__(self, n_clusters=3, random_state=None, max_iter=300):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
    
    def fit(self, X):
        # Set the random seed for reproducibility
        np.random.seed(self.random_state)
        
        # Initialize the cluster centers randomly from the data points
        rand_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.cluster_centers_ = X[rand_indices]
        
        for _ in range(self.max_iter):
            # Step 1: Assign each data point to the nearest center
            self.labels_ = np.argmin(cdist(X, self.cluster_centers_), axis=1)
            
            # Step 2: Compute new center as the mean of the data points assigned to each cluster
            new_centers = np.array([X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])
            
            # Step 3: If the centers do not change, then the algorithm has converged
            if np.all(self.cluster_centers_ == new_centers):
                break
            
            # Update the centers
            self.cluster_centers_ = new_centers
        
        return self
    
    def predict(self, X):
        return np.argmin(cdist(X, self.cluster_centers_), axis=1)

class MySpectralClustering:
    def __init__(self, n_clusters=2, gamma=1.0):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.eigenvalues_ = None
        self.eigenvectors_ = None
    
    def laplacian(self, W):
        # Degree matrix
        G = np.diag(np.sum(W, axis=1))
        
        # Laplacian matrix
        L = G - W
        return L
    
    def fit_predict(self, X):
        # Step 1: Create the similarity graph
        # Use gamma as the coefficient for the RBF kernel
        W = np.exp(-self.gamma * cdist(X, X, 'euclidean')**2)
        np.fill_diagonal(W, 0)
        
        # Step 2: Form the graph Laplacian
        L = self.laplacian(W)
        
        # Step 3: Compute the first k eigenvectors
        self.eigenvalues_, self.eigenvectors_ = np.linalg.eigh(L)
        Z = self.eigenvectors_[:, 1:self.n_clusters+1]
        
        # Step 4: Cluster the rows of the matrix of eigenvectors
        kmeans = MyKMeans(self.n_clusters, max_iter=100)
        kmeans.fit(Z)
        
        return kmeans.labels_

class FNN:
    def __init__(self, hidden_layer_sizes=[10], epochs=100, learning_rate=0.1):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.epochs = epochs
        self.learning_rate = learning_rate
    
    def initialize_weights(self, n_features):
        layer_sizes = [n_features] + self.hidden_layer_sizes + [1]
        weights = []
        biases = []
        sigmas = []
        
        for i in range(len(layer_sizes) - 1):
            std_dev = np.sqrt(2 / layer_sizes[i])
            weights.append(np.random.normal(0.0, std_dev, (layer_sizes[i], layer_sizes[i + 1])))
            biases.append(np.zeros((1, layer_sizes[i + 1])))
            sigmas.append(relu if i < len(self.hidden_layer_sizes) else sigmoid)
        
        return weights, biases, sigmas
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights, self.biases, self.sigmas = self.initialize_weights(n_features)
        
        for _ in range(self.epochs):
            # Forward pass
            layer_outputs = [X]
            
            for w, b, sigma in zip(self.weights, self.biases, self.sigmas):
                z = np.dot(layer_outputs[-1], w) + b
                layer_outputs.append(sigma(z))
            
            # Backward pass
            # Calculate the error for the output layer
            delta = layer_outputs[-1] - y.reshape(-1, 1)
            deltas = [delta * sigmoid_derivative(layer_outputs[-1])]
            
            # Calculate the errors for the hidden layers
            for l in range(len(self.weights) - 1, 0, -1):
                delta = np.dot(deltas[0], self.weights[l].T) * relu_derivative(layer_outputs[l])
                deltas.insert(0, delta)
            
            # Update the weights and biases for all layers
            for l in range(len(self.weights)):
                # Derivative of the loss with respect to weights
                weight_gradient = np.dot(layer_outputs[l].T, deltas[l]) / n_samples
                
                # Derivative of the loss with respect to biases
                bias_gradient = np.sum(deltas[l], axis=0, keepdims=True) / n_samples
                
                # Update weights and biases
                self.weights[l] -= self.learning_rate * weight_gradient
                self.biases[l] -= self.learning_rate * bias_gradient
    
    def predict(self, X):
        output = X
        
        for i in range(len(self.weights)):
            z = np.dot(output, self.weights[i]) + self.biases[i]
            activation_function = relu if i < len(self.weights) - 1 else sigmoid
            output = activation_function(z)
        
        return output

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(torch.relu(x))
        x = self.conv2(x)
        x = self.pool2(torch.relu(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x) # No activation, raw scores
        return x

class SimpleSVM:
    def __init__(self, kernel='linear', C=1.0, degree=3, gamma=None, learning_rate=0.01, n_iters=1000):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.alpha = None
        self.b = 0
        self.X_train = None
        self.y_train = None
    
    def _kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            return (np.dot(x1, x2) + 1) ** self.degree
        elif self.kernel == 'rbf':
            if self.gamma is None:
                self.gamma = 1 / x1.shape[0] # Default gamma value
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError("Unknown kernel")
    
    def fit(self, X, y):
        self.X_train = copy(X)
        self.y_train = copy(y)
        n_samples, _ = X.shape
        self.alpha = np.zeros(n_samples)
        
        for _ in range(self.n_iters):
            for i in range(n_samples):
                gradient = 1 - y[i] * sum(self.alpha[j] * y[j] * self._kernel_function(X[i], X[j]) for j in range(n_samples))
                self.alpha[i] += self.learning_rate * gradient
                self.alpha[i] = min(max(self.alpha[i], 0), self.C)
        
        self.b = 1 / X.shape[0] * sum(y[i] - sum(self.alpha[j] * y[j] * self._kernel_function(X[j], X[i]) for j in range(n_samples)) for i in range(n_samples))
    
    def predict(self, X):
        predictions = np.sign([np.sum(self.alpha * self.y_train * [self._kernel_function(x_i, x) for x_i in self.X_train]) + self.b for x in X])
        return predictions

class MyRandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        for _ in range(self.n_estimators):
            row_indices = np.random.choice(n_samples, n_samples, replace=True)
            n_selected_features = int(np.sqrt(n_features))
            feature_indices = np.random.choice(n_features, n_selected_features, replace=False)
            X_sample, y_sample = X[row_indices][:, feature_indices], y[row_indices]
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append((tree, feature_indices))
    
    def predict(self, X):
        predictions = np.array([tree.predict(X[:, features]) for tree, features in self.trees]).T
        return mode(predictions, axis=1).mode.ravel()
    
    def feature_importances(self):
        # Initialize an array to store feature importances
        importances = np.zeros(X_train.shape[1])
        
        # Sum up feature importances from each tree
        for tree, feature_indices in self.trees:
            tree_importances = tree.feature_importances_
            
            for i, idx in enumerate(feature_indices):
                importances[idx] += tree_importances[i]
        
        # Average the importances over all trees
        importances /= self.n_estimators
        
        return importances