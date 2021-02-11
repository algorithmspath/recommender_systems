import numpy as np

class recommender_system:
    def __init__(self, rating_matrix, valid_matrix, num_features=10):
        self.rating_matrix = rating_matrix
        self.valid_matrix = valid_matrix
        self.num_users, self.num_items = rating_matrix.shape
        self.num_features = num_features
        # initialize users_matrix, items_matrix to small random values
        self.users_matrix = np.random.rand(num_features, self.num_users) * 0.00001
        self.items_matrix = np.random.rand(num_features, self.num_items) * 0.00001
        self.mean_norm = rating_matrix.mean(axis=0).reshape((1,self.num_items))
        self.rating_matrix -= self.mean_norm # mean-normalize rating_matrix

    def train(self, num_iter=1000, print_cost=False):
        for i in range(0,num_iter):
            self.gradient_descent()
            if print_cost:
                print(self.compute_cost())

    def compute_cost(self):
        t1 = ((self.users_matrix.T @ self.items_matrix) + self.mean_norm) - self.rating_matrix
        t2 = self.valid_matrix / (self.num_users * self.num_items)
        return (t1 * t2).sum()

    def gradient_descent(self, learning_rate=100, lambda_reg=0.001):
        rating_matrix, valid_matrix = self.rating_matrix, self.valid_matrix
        users_matrix, items_matrix = self.users_matrix, self.items_matrix
        num_users, num_items = self.num_users, self.num_items
        users_gradient = (items_matrix @ ((items_matrix.T @ users_matrix - rating_matrix.T) * valid_matrix.T / (num_users * num_items))
         + lambda_reg * users_matrix)
        items_gradient = (users_matrix @ ((users_matrix.T @ items_matrix - rating_matrix) * valid_matrix / (num_users * num_items))
         + lambda_reg * items_matrix)
        self.users_matrix -= learning_rate * users_gradient
        self.items_matrix -= learning_rate * items_gradient

    def predict_rating(self, user, item):
        return self.users_matrix[:,user].T @ self.items_matrix[:,item] + self.mean_norm[0,item]

def test_rc(rating_matrix, valid_matrix, print_cost=False):
    rc = recommender_system(rating_matrix, valid_matrix, num_features=10)
    rc.train(num_iter=100, print_cost=print_cost)
    x = (rc.users_matrix.T @ rc.items_matrix) + rc.mean_norm
    # print(x)
    print(rating_matrix[0,100])
    print(rc.predict_rating(0,100))

def get_test_data():
    num_users, num_items = 100, 1000
    rating_matrix = 2 * np.ones((num_users, num_items))
    rating_matrix[:50,:] += 2
    rating_matrix[:50,50:] += 2
    valid_matrix = np.ones((num_users, num_items))
    valid_matrix[40:60,:] -= 1
    return rating_matrix, valid_matrix

def run_simulation():
    rating_matrix, valid_matrix = get_test_data()
    test_rc(rating_matrix, valid_matrix, print_cost=False)

def main():
    run_simulation()
    pass

if __name__ == '__main__':
    main()
