import torch
import itertools
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, n, theta=0.1):
        self.n = n
        self.W = torch.zeros(n, n)
        self.theta = theta

    # Getters and setters
    def set_weights(self, W):
        self.W = W
        return self.W

    def get_weights(self):
        return self.W

    # training
    def train_once(self, x):
        self.W += torch.outer(x, x)
        self.W /= self.n
        return self.W

    def train(self, X):
        for x in X:
            self.W += torch.outer(x, x)
        self.W /= self.n
        return self.W
    
    # training with bias
    def train_bias(self, X, rho):
        for x in X:
            self.W += torch.outer(x-rho, x-rho)
        self.W /= self.n
        return self.W
    
    def train_bias_once(self, x, rho):
        self.W += torch.outer(x-rho, x-rho)
        self.W /= self.n
        return self.W

    # recall step
    def update_rule(self, x):
        return torch.sign(self.W @ x)
    
    def update_rule_bias(self, x):
        return 0.5 + 0.5 * torch.sign((self.W @ x.float()) - self.theta)

    # synchronous recall
    def predict_sync(self, x, max_iter=300):
        for _ in range(max_iter):
            y = self.update_rule(x)
            if torch.all(y == x):
                break
            x = y
        return y
    
    # Asynchronous recall
    def predict_asynchronous(self, x, original=None, max_iter=20000):
        for _ in range(max_iter):
            i = torch.randint(0, x.shape[0], (1,))
            y = self.update_rule(x)
            if torch.all(y == x):
                break
            x[i] = y[i]
            if original != None :
                if _ % 2000 == 0:
                    print(f"Iteration {_}")
                    self.plot_images([original, x], (32, 32), ["Original",
            
                                "Reconstructing at iteration " + str(_)])
        if original != None :
            self.plot_images([original, x], (32, 32), ["Original", "Reconstructed"])
        return x
    
    def predict_asynchronous_nrj(self, x, original=None, max_iter=20000):
        energy = [self.energy(x)]
        for _ in range(max_iter):
            i = torch.randint(0, x.shape[0], (1,))
            y = self.update_rule(x)
            energy.append(self.energy(y))
            if torch.all(y == x):
                break
            x[i] = y[i]
            if original != None :
                if _ % 2000 == 0:
                    print(f"Iteration {_}")
                    self.plot_images([original, x], (32, 32), ["Original",
            
                                "Reconstructing at iteration " + str(_)])
        if original != None :
            self.plot_images([original, x], (32, 32), ["Original", "Reconstructed"])
        return x, energy
    
    # Other useful methods    
    def energy(self, x):
        return - 0.5 * x @ self.W @ x

    
    def avg_activity(self, X):
        res = 0
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                res += X[i, j]
        return res/(X.shape[0]*X.shape[1])


    def attractors(self):
        permutations = torch.tensor(list(itertools.product([-1, 1], repeat=8)), dtype=torch.float32)
        attractors = []
        for x in permutations :
            attractors.append(self.predict_sync(x))
        attractors = torch.stack(attractors)
        attractors = torch.unique(attractors, dim=0)
        return attractors
    
    def plot_images(self, images, shape, title, filename=None):
        fig, axes = plt.subplots(1, len(images), figsize=(24, 24))
        for i, ax in enumerate(axes):
            ax.imshow(images[i].reshape(shape).T, cmap="gray")
            ax.set_title(title[i])
            ax.axis("off")
        if filename:
            plt.savefig(filename)
        plt.show()