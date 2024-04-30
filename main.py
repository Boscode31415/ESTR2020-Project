import pandas as pd
import numpy as np
from numpy.random import multivariate_normal
from scipy.stats import wishart
import matplotlib.pyplot as plt
import numpy as np
import gc
import seaborn as sns

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preposses(R):
    dataset = pd.read_csv('ratings.csv')
    for _, row in dataset.iterrows():
        userId = int(row['userId'] - 1)
        movieId = int(row['movieId'] - 1)
        R[userId][movieId] = float(row['rating'])
    return

def inference(T, j ,k, min, max, U, V, alpha, norm):
    max_prob = -1.00
    max_value = 0
    for x in range(1, 11):
        prob = 0.0
        for t in range(T):
            mean = sigmoid(np.dot(U[t][j], np.transpose(V[t][k])))/ norm[t]
            mean = (mean - min[t]) / (max[t] - min[t])
            if mean < 0.0:
                mean = 0.0
            if mean > 1.0:
                mean = 1.0
            sd = alpha ** -0.5
            prob +=  np.exp(-0.5 * (((x - 1) / 9 - mean)/ sd) ** 2) / ((2 * np.pi) ** 0.5 * sd)
        if prob > max_prob:
            max_value = x
            max_prob = prob    
    return max_value / 2

if __name__ == '__main__':
    error_list = []
    N, M ,T, D = 610, 193609, 10, 50
    R = np.zeros((N + 1, M + 1))
    U_current = np.zeros((N, D))
    V_current = np.zeros((M, D))
    U_old = np.zeros((N, D))
    V_old = np.zeros((M, D))
    U = [np.zeros((N, D)) for _ in range(T)]
    V = [np.zeros((M, D)) for _ in range(T)]
    min = np.zeros(T)
    max = np.zeros(T)
    norm = np.zeros(T)
    mu_0 = np.zeros(D)
    beta_0 = 1
    nu_0 = D
    W_0 = np.identity(D)
    alpha = 2
    preposses(R)
    for i in range(T):
        #Sample user hyperparameters
        U_bar = np.mean(U_old, axis = 0)
        S_bar = np.zeros((D, D))
        for j in range(N):
            S_bar += np.dot(U_old[j] - U_bar, np.transpose(U_old[j] - U_bar))
        mu_0_star = (N * U_bar + beta_0 * mu_0) / (N + beta_0)
        beta_0_star = N + beta_0
        nu_0_star = nu_0 + N
        W_0_inv = np.linalg.inv(W_0)
        W_0_star_inv = W_0_inv + N * S_bar + N * beta_0 * np.dot(mu_0 - U_bar, np.transpose(mu_0 - U_bar)) / (N + beta_0)
        W_0_star = np.linalg.inv(W_0_star_inv)
        lambda_U = wishart.rvs(nu_0_star, W_0_star)
        mu_U = multivariate_normal(mu_0_star, np.linalg.inv(beta_0_star * lambda_U))

        #Sample user feature matrix
        np.copyto(U_old, U_current)
        for j in range(N):
            lambda_star = lambda_U
            mu_star = np.dot(lambda_U, mu_U)
            for k in range(M):
                if R[j][k] != 0:
                    lambda_star += alpha * np.dot(V_old[k], np.transpose(V_old[k]))
                    mu_star += alpha * R[j][k] * V_old[k]
            lambda_star_inv = np.linalg.inv(lambda_star)
            mu_star = np.dot(lambda_star_inv, mu_star)
            U_current[j] = multivariate_normal(mu_star, lambda_star_inv)
        np.copyto(U[i], U_current)

        #Sample movie hyperparameters
        V_bar = np.mean(V_old, axis = 0)
        S_bar = np.zeros((D, D))
        for j in range(M):
            S_bar += np.dot(V_old[j] - V_bar, np.transpose(V_old[j] - V_bar))
        beta_0_star = M + beta_0
        nu_0_star = nu_0 + M
        mu_0_star = (N * V_bar + beta_0 * mu_0) / (M + beta_0)
        W_0_star_inv = W_0_inv + M * S_bar + M * beta_0 * np.dot(mu_0 - V_bar, np.transpose(mu_0 - V_bar)) / (M + beta_0)
        W_0_star = np.linalg.inv(W_0_star_inv)
        lambda_V = wishart.rvs(nu_0_star, W_0_star)
        mu_V = multivariate_normal(mu_0_star, np.linalg.inv(beta_0_star * lambda_V))
        


        #Sample movie feature matrix
        np.copyto(V_old, V_current)
        for j in range(M):
            lambda_star = lambda_V
            mu_star = np.dot(lambda_V, mu_V)
            for k in range(N):
                if R[k][j] != 0:
                    lambda_star += alpha * np.dot(U_current[k], np.transpose(U_current[k]))
                    mu_star += alpha * R[k][j] * U_current[k]
            lambda_star_inv = np.linalg.inv(lambda_star)
            mu_star = np.dot(lambda_star_inv, mu_star)
            V_current[j] = multivariate_normal(mu_star, lambda_star_inv)
        np.copyto(V[i], V_current)

        tmp = sigmoid(np.dot(U_current, np.transpose(V_current)))
        norm[i] = np.linalg.norm(tmp)
        tmp = tmp/norm[i]
        if 0.1 + 0.1 * i < 2.5:
            min_factor = 0.1 + 0.15 * i
        else:
            min_factor = 2.5
        if 99.9 - i * 0.1 > 95:
            max_factor = 99.9 - i * 0.4
        else:
            max_factor = 95
        min[i], max[i] = np.percentile(tmp, min_factor), np.percentile(tmp, max_factor)
        #tmp = (tmp - min[i]) / (max[i] - min[i])
        #fig, ax = plt.subplots()
        #ax.hist(tmp.flatten(), bins='auto')
        #plt.show()
        del tmp
        gc.collect()

        #root mean square error
        rating_distribution = [int(0) for _ in range(11)]
        error, cnt = 0, 1
        error_distribution = [int(0) for _ in range(19)]
        for j in range(N):
            for k in range(M):
                if R[j][k] != 0.0:
                    infer_R = inference(i + 1, j, k, min, max, U, V, alpha, norm)
                    rating_distribution[int(infer_R * 2)] += 1
                    error_distribution[int((R[j][k] - infer_R)*2+9)] += 1
                    error += (R[j][k] - infer_R)**2
                    cnt += 1
        print((error/cnt)**0.5)
        error_list.append((error/cnt)**0.5)
        for j in range(11):
            print(int(rating_distribution[j]), end = ' ')
        print()
        for j in range(19):
            print(int(error_distribution[j]), end = ' ')
        print()

    fig, ax = plt.subplots()
    ax.plot(range(len(error_list)), error_list, color='blue')

    baseline_error_3, baseline_error_3_5, baseline_error_4, cnt = 0, 0, 0, 0
    for j in range(N):
        for k in range(M):
            if R[j][k] != 0:
                baseline_error_3 += (R[j][k] - 3)**2
                baseline_error_3_5 += (R[j][k] - 3.5)**2
                baseline_error_4 += (R[j][k] - 4)**2
                cnt+= 1
    baseline_error_3 = (baseline_error_3/cnt) ** 0.5
    baseline_error_3_5 = (baseline_error_3_5/cnt) ** 0.5
    baseline_error_4 = (baseline_error_4/cnt) ** 0.5

    ax.axhline(y=baseline_error_3, color='green', linestyle='--', label='Baseline Error: 3')
    ax.axhline(y=baseline_error_4, color='yellow', linestyle='--', label='Baseline Error: 4')
    ax.axhline(y=baseline_error_3_5, color='red', linestyle='--', label='Baseline Error: 3.5')
    ax.set_title('RMSE vs Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error')
    plt.show()
    
    rating_distribution = [int(0) for _ in range(11)]
    for j in range(N):
        for k in range(M):
                rating_distribution[int(2*inference(T, j, k, min, max, U, V, alpha, norm))] += 1
    sns.histplot(rating_distribution)
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.show()
    

    
                    
   
