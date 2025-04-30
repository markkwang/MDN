import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns

import scipy.stats
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.decomposition import PCA
from umap import UMAP

from tensorflow.keras.layers import Dense

tf.compat.v1.disable_v2_behavior()  # important

def sample_from_mixture_f(x, pred_weights, pred_means, pred_std, dist):
          """Draws samples from mixture model.

          Returns 2 d array with input X and sample from prediction of mixture model.
          """
          amount = len(x)
          samples = np.zeros((amount, 1))
          n_mix = len(pred_weights[0])
          to_choose_from = np.arange(n_mix)
          for j, (weights, means, std_devs) in enumerate(
                  zip(pred_weights, pred_means, pred_std)):
            index = np.random.choice(to_choose_from, p=weights)
            if dist.lower() == 'normal':
                samples[j, 0] = np.random.normal(means[index], std_devs[index], size=1)
            elif (dist.lower() == 'laplace' or dist.lower() == 'laplacian') == True:
                samples[j, 0] = np.random.laplace(means[index], std_devs[index], size=1)
            #samples[j, 0] = x[j]
            if j == amount - 1:
              break
          return samples

def listToString(s):
            # traverse in the string
            liste = []
            for ele in s:
                ele = str(ele)
                ele = 'X' + ele
                liste.append(ele)
            # return string
            return liste


class MDN:
    def __init__(self, n_mixtures = 1,
                 dist = 'laplace',
                 input_neurons = 1000,
                 hidden_neurons = [25],
                 optimizer = 'adam',
                 learning_rate = 0.001,
                 early_stopping = 10,
                 input_activation = 'relu',
                 hidden_activation = 'tanh'):
        
        tf.compat.v1.reset_default_graph()
        self.n_mixtures = n_mixtures
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.optimizer = optimizer
        self.input_activation = input_activation
        self.hidden_activation = hidden_activation
        self.dist = dist

        self._X = None
        self._Y = None
        self.x = None
        self.K = None
        self.layer_last = None
        self.mu = None
        self.var = None
        self.pi = None
        self.mixture_distribution = None
        self.distribution = None
        self.likelihood = None
        self.log_likelihood = None
        self.mean_loss = None
        self.global_step = None
        self.train_op = None
        self.init = None
        self.sess = None
        self.stopping_step = 0
        self.best_loss = None

    def fit(self, X, Y, epochs, batch_size):

        X = X.astype(np.float32)
        Y = Y.astype(np.float32)

        n = len(X)

        self._X = X.copy()
        self._Y = Y.copy()

        dataset = tf.compat.v1.data.Dataset \
    					.from_tensor_slices((X, Y)) \
    					.repeat(epochs).shuffle(len(X)).batch(batch_size)
        iter_ = tf.compat.v1.data.make_one_shot_iterator(dataset)

        self.x, y = iter_.get_next()

        self.K = self.n_mixtures

        if self.input_activation.lower() == 'relu':
            self.input_activation = tf.nn.relu
        elif self.input_activation.lower() == 'sigmoid':
            self.input_activation = tf.nn.sigmoid
        elif self.input_activation.lower() == 'tanh':
            self.input_activation = tf.nn.tanh
        elif self.input_activation.lower() == 'leaky_relu':
            self.input_activation = tf.nn.leaky_relu
        elif self.input_activation.lower() == 'softmax':
            self.input_activation = tf.nn.softmax
        else:
            raise ValueError("Activation function not supported. Use 'relu', 'sigmoid', 'tanh', 'leaky_relu' or 'softmax'.")

        if self.hidden_activation.lower() == 'relu':
            self.hidden_activation = tf.nn.relu
        elif self.hidden_activation.lower() == 'sigmoid':
            self.hidden_activation = tf.nn.sigmoid
        elif self.hidden_activation.lower() == 'tanh':
            self.hidden_activation = tf.nn.tanh
        elif self.hidden_activation.lower() == 'leaky_relu':
            self.hidden_activation = tf.nn.leaky_relu
        elif self.hidden_activation.lower() == 'softmax':
            self.hidden_activation = tf.nn.softmax
        else:
            raise ValueError("Activation function not supported. Use 'relu', 'sigmoid', 'tanh', 'leaky_relu' or 'softmax'.")
    
        n_layer = len(self.hidden_neurons)

        if n_layer == 0:
            self.layer_last = tf.compat.v1.layers.dense(self.x, units=self.input_neurons, activation=self.input_activation)
            self.mu = tf.compat.v1.layers.dense(self.layer_last, units=self.K, activation=None, name="mu")
            self.var = tf.exp(tf.compat.v1.layers.dense(self.layer_last, units=self.K, activation=None, name="sigma"))
            self.pi = tf.compat.v1.layers.dense(self.layer_last, units=self.K, activation=tf.nn.softmax, name="mixing")

        elif n_layer >= 1:
            layer = Dense(units=self.input_neurons, activation=self.input_activation)(self.x)
            for i in range(n_layer):
                n_neurons = self.hidden_neurons[i]
                layer = Dense(units=n_neurons, activation=self.hidden_activation)(layer)

            self.layer_last = layer
            self.mu = Dense(units=self.K, activation=None, name="mu")(self.layer_last)
            self.var = tf.exp(Dense(units=self.K, activation=None, name="sigma")(self.layer_last))
            self.pi = Dense(units=self.K, activation=tf.nn.softmax, name="mixing")(self.layer_last)
        
        else:
            pass

        self.mixture_distribution = tfp.distributions.Categorical(probs=self.pi)

        if self.dist.lower() == 'normal':
            self.distribution = tfp.distributions.Normal(loc=self.mu, scale=self.var)
        elif self.dist.lower() == 'laplace':
            self.distribution = tfp.distributions.Laplace(loc=self.mu, scale=self.var)
        else:
            self.distribution = tfp.distributions.Normal(loc=self.mu, scale=self.var)

        self.likelihood = tfp.distributions.MixtureSameFamily(mixture_distribution=self.mixture_distribution,
                                                             components_distribution=self.distribution)

        self.log_likelihood = -self.likelihood.log_prob(tf.transpose(y))
        self.mean_loss = tf.reduce_mean(self.log_likelihood)

        self.global_step = tf.Variable(0, trainable=False)

        if self.optimizer.lower() == 'adam':
            print("Using Adam optimizer")
            self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.mean_loss)
        elif self.optimizer.lower() == 'sgd':
            self.train_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.mean_loss)
        elif self.optimizer.lower() == 'rmsprop':
            self.train_op = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.mean_loss)
        else:
            raise ValueError("Optimizer not supported. Use 'adam', 'sgd' or 'rmsprop'.")

        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        best_loss = np.inf
        for i in range(epochs * (n // batch_size)):
            _, loss, mu, var, pi, x__ = self.sess.run([self.train_op, self.mean_loss, self.mu, self.var, self.pi, self.x])

            if loss < best_loss:
                self.stopping_step = 0
                self.best_loss = loss
                
                best_mu = mu
                best_var = var
                best_pi = pi
                best_mean_y = mu[:,0]
                best_x = x__
                best_loss = loss
                print("Epoch: {} Loss: {:3.3f}".format(i, loss))
            else:
                self.stopping_step += 1
            
            if self.stopping_step >= self.early_stopping:
                print("Early stopping is trigger at step: {} loss:{}".format(i,loss))
                return
            else:
                pass

            self._mean_y_train = mu[:,0]
            self._dist_mu_train = mu
            self._dist_var_train = var
            self._dist_pi_train = pi
            self._x_data_train = x__

    def predict_best(self, X_pred, q = 0.95, y_scaler = None):

        best_mean_y, best_mu, best_var, best_pi, best_x  =  self.sess.run([self.mu[:,0],
                      self.mu,
                      self.var,
                      self.pi,
                      self.x],
                      feed_dict={self.x: X_pred})

        self._mean_y_pred = best_mean_y
        self._dist_mu_pred = best_mu
        self._dist_var_pred = best_var
        self._dist_pi_pred = best_pi
        self._x_data_pred = best_x

        cluster_probs = self._dist_pi_pred
        best_cluster = np.argmax(cluster_probs, axis = 1)
        best_cluster_prob = np.max(cluster_probs, axis = 1)

        list_dist_mu = []
        for i in range(0, len(self._dist_mu_pred)):
            list_dist_mu.append(self._dist_mu_pred[i, np.argmax(cluster_probs, axis = 1)[i]])

        list_dist_var = []
        for i in range(0, len(self._dist_var_pred)):
            list_dist_var.append(self._dist_var_pred[i, np.argmax(cluster_probs, axis = 1)[i]])


        if y_scaler != None:
            y_pred_mean = y_scaler.inverse_transform(np.array(list_dist_mu).reshape(1, -1))
            y_pred_upper = y_scaler.inverse_transform((np.array(list_dist_mu) + (np.array(list_dist_var) * scipy.stats.t.ppf((1 + q) / 2., len(X_pred)-1))).reshape(1, -1))
            y_pred_lower = y_scaler.inverse_transform((np.array(list_dist_mu) - (np.array(list_dist_var) * scipy.stats.t.ppf((1 + q) / 2., len(X_pred)-1))).reshape(1, -1))

        else:
            y_pred_mean = np.array(list_dist_mu)
            y_pred_upper = np.array(list_dist_mu) + (np.array(list_dist_var) * scipy.stats.t.ppf((1 + q) / 2., len(X_pred)-1))
            y_pred_lower = np.array(list_dist_mu) - (np.array(list_dist_var) * scipy.stats.t.ppf((1 + q) / 2., len(X_pred)-1))


        all_preds = pd.DataFrame(y_pred_mean.reshape(-1,1))
        all_preds.columns = ['Y_PRED_MEAN']
        all_preds['Y_PRED_LOWER'] = y_pred_lower.reshape(-1,1)
        all_preds['Y_PRED_UPPER'] = y_pred_upper.reshape(-1,1)

        all_preds['BEST_CLUSTER'] = best_cluster
        all_preds['DIST_PI'] = best_cluster_prob

        all_preds['DIST_MU'] = np.array(list_dist_mu)
        all_preds['DIST_SIGMA'] = np.array(list_dist_var)

        if y_scaler != None:
            all_preds['DIST_MU_UNSCALED'] = y_scaler.inverse_transform(np.array(list_dist_mu))
            all_preds['DIST_SIGMA_UNSCALED'] = y_scaler.inverse_transform(np.array(list_dist_var))
        else:
            pass

        return all_preds#, samples

    def predict_mixed(self, X_pred, q = 0.95, y_scaler = None):

        all_preds = self.predict_dist(X_pred = X_pred, q=q, y_scaler = y_scaler)


        i = 0
        for elem in all_preds:
            if i == 0:
                y_pred_mean = elem['Y_PRED_MEAN'] * elem['DIST_PI']
                y_pred_lower = elem[['Y_PRED_LOWER']]
                y_pred_upper = elem[['Y_PRED_UPPER']]
            else:
                y_pred_mean = y_pred_mean + (elem['Y_PRED_MEAN'] * elem['DIST_PI'])
                y_pred_lower2 = pd.concat([y_pred_lower, elem[['Y_PRED_LOWER']]], axis = 1)
                y_pred_lower2 = np.nanmin(y_pred_lower2.values, axis=1)
                y_pred_lower['Y_PRED_LOWER'] = y_pred_lower2

                y_pred_upper2 = pd.concat([y_pred_upper, elem[['Y_PRED_UPPER']]], axis = 1)
                y_pred_upper2 = np.nanmax(y_pred_upper2.values, axis=1)
                y_pred_upper['Y_PRED_UPPER'] = y_pred_upper2
            i = i + 1

        all_preds = pd.concat([y_pred_mean, y_pred_lower, y_pred_upper], axis = 1)
        all_preds.columns = ['Y_PRED_MEAN', 'Y_PRED_LOWER', 'Y_PRED_UPPER']

        return all_preds#, samples

    def predict_with_overlaps(self, X_pred, q=0.95, y_scaler = None, pi_threshold = 0.215):

        all_preds = self.predict_dist(X_pred = X_pred, q=q, y_scaler = y_scaler)

        X_df = pd.DataFrame(X_pred)
        xcol_list = list(range(0, len(X_df.columns)))
        xcol_list = listToString(xcol_list)

        X_df.columns = xcol_list

        i = 0
        for elem in all_preds:
            if i == 0:
                y_mix = elem[['DIST_PI', 'Y_PRED_MEAN', 'Y_PRED_LOWER', 'Y_PRED_UPPER']]
                y_mix.columns = ['M' + str(i) + '_' + 'DIST_PI', 'M' + str(i) + '_' + 'Y_PRED_MEAN', 'M' + str(i) + '_' + 'Y_PRED_LOWER', 'M' + str(i) + '_' + 'Y_PRED_UPPER']
                full_data = pd.concat([X_df, y_mix], axis = 1)
                full_data['MIXTURE_' + str(i)] = np.where(full_data['M' + str(i) + '_' + 'DIST_PI'] >= pi_threshold, 'M' + str(i), '')
            else:
                y_mix = elem[['DIST_PI', 'Y_PRED_MEAN', 'Y_PRED_LOWER', 'Y_PRED_UPPER']]
                y_mix.columns = ['M' + str(i) + '_' + 'DIST_PI', 'M' + str(i) + '_' + 'Y_PRED_MEAN', 'M' + str(i) + '_' + 'Y_PRED_LOWER', 'M' + str(i) + '_' + 'Y_PRED_UPPER']
                full_data = pd.concat([full_data, y_mix], axis = 1)
                full_data['MIXTURE_' + str(i)] = np.where(full_data['M' + str(i) + '_' + 'DIST_PI'] >= pi_threshold, 'M' + str(i), '')
            i = i +1

        mixtures_cols = full_data.loc[:, full_data.columns.str.contains('MIXTURE')].columns.tolist()
        full_data['POSSIBLE_MIX'] = full_data[mixtures_cols].apply(lambda row: ','.join(row.values.astype(str)), axis=1)

        for i in range(0, self.n_mixtures+10):
            full_data['POSSIBLE_MIX'] = full_data['POSSIBLE_MIX'].str.replace(',,',',')

        full_data['POSSIBLE_MIX'] = np.where(full_data['POSSIBLE_MIX'].str[0:1] == ',', full_data['POSSIBLE_MIX'].str[1:], full_data['POSSIBLE_MIX'])
        full_data['POSSIBLE_MIX'] = np.where(full_data['POSSIBLE_MIX'].str[-1:] == ',', full_data['POSSIBLE_MIX'].str[:-1], full_data['POSSIBLE_MIX'])

        i = 0
        renamed_cols = X_df.columns.tolist() + ['Y_PRED_MEAN', 'Y_PRED_LOWER', 'Y_PRED_UPPER', 'DIST_PI', 'POSSIBLE_MIX', 'SOURCE']
        for col in mixtures_cols:
            all_cols = X_df.columns.tolist() + ['M' + str(i) + '_' + 'Y_PRED_MEAN', 'M' + str(i) + '_' + 'Y_PRED_LOWER', 'M' + str(i) + '_' + 'Y_PRED_UPPER', 'M' + str(i) + '_' + 'DIST_PI', 'POSSIBLE_MIX']
            if i == 0:
                to_merge_data = full_data[full_data[col] != ''][all_cols]
                to_merge_data['SOURCE'] = 'M' + str(i)
                to_merge_data.columns = renamed_cols
            else:
                to_merge_data2 = full_data[full_data[col] != ''][all_cols]
                to_merge_data2['SOURCE'] = 'M' + str(i)
                to_merge_data2.columns = renamed_cols
                to_merge_data = pd.concat([to_merge_data, to_merge_data2], axis = 0, ignore_index = True)
            i = i +1

        return to_merge_data


    def predict_dist(self, X_pred, q = 0.95, y_scaler = None):

        best_mean_y, best_mu, best_var, best_pi, best_x  =  self.sess.run([self.mu[:,0],
                      self.mu,
                      self.var,
                      self.pi,
                      self.x],
                      feed_dict={self.x: X_pred})

        self._mean_y_pred = best_mean_y
        self._dist_mu_pred = best_mu
        self._dist_var_pred = best_var
        self._dist_pi_pred = best_pi
        self._x_data_pred = best_x

        cluster_probs = self._dist_pi_pred
        best_cluster = np.argmax(cluster_probs, axis = 1)
        best_cluster_prob = np.max(cluster_probs, axis = 1)

        cluster_preds = []
        for i in range(0, self._dist_mu_pred.shape[1]):
            if y_scaler != None:
                y_pred_mean = y_scaler.inverse_transform(np.array(self._dist_mu_pred[:, i]))
                y_pred_upper = y_scaler.inverse_transform(np.array(self._dist_mu_pred[:, i]) + (np.array(self._dist_var_pred[:, i]) * scipy.stats.t.ppf((1 + q) / 2., len(X_pred)-1)))
                y_pred_lower = y_scaler.inverse_transform(np.array(self._dist_mu_pred[:, i]) - (np.array(self._dist_var_pred[:, i]) * scipy.stats.t.ppf((1 + q) / 2., len(X_pred)-1)))
                y_mu = np.array(self._dist_mu_pred[:, i])
                y_var = np.array(self._dist_var_pred[:, i])
            else:
                y_pred_mean = np.array(self._dist_mu_pred[:, i])
                y_pred_upper = np.array(self._dist_mu_pred[:, i]) + (np.array(self._dist_var_pred[:, i]) * scipy.stats.t.ppf((1 + q) / 2., len(X_pred)-1))
                y_pred_lower = np.array(self._dist_mu_pred[:, i]) - (np.array(self._dist_var_pred[:, i]) * scipy.stats.t.ppf((1 + q) / 2., len(X_pred)-1))
                y_mu = np.array(self._dist_mu_pred[:, i])
                y_var = np.array(self._dist_var_pred[:, i])
            #preds = np.concatenate((y_pred_mean, y_pred_lower, y_pred_upper, self._dist_pi_pred[:, i].reshape(-1,1)), axis = 1)
            preds = pd.concat([pd.DataFrame(y_pred_mean),
                              pd.DataFrame(y_pred_lower),
                              pd.DataFrame(y_pred_upper),
                              pd.DataFrame(self._dist_pi_pred[:, i]),
                              pd.DataFrame(y_mu),
                              pd.DataFrame(y_var),
                              ], axis = 1, ignore_index = True)
            #preds = pd.DataFrame(preds)
            preds.columns = ['Y_PRED_MEAN', 'Y_PRED_LOWER', 'Y_PRED_UPPER', 'DIST_PI', 'DIST_MU', 'DIST_VAR']
            preds['N_CLUSTER'] = i

            cluster_preds.append(preds)

        return cluster_preds#, samples

    def sample_from_mixture(self, X_pred, n_samples_batch = 1, y_scaler = None):

        all_preds = self.predict_dist(X_pred = X_pred, q = 0.95, y_scaler = y_scaler)

        i = 0
        for elem in all_preds:
            if i == 0:
                out_pi_test = elem[['DIST_PI']]
                out_mu_test = elem[['DIST_MU']]
                out_sigma_test = elem[['DIST_VAR']]
            else:
                out_pi_test2 = elem[['DIST_PI']]
                out_mu_test2 = elem[['DIST_MU']]
                out_sigma_test2 = elem[['DIST_VAR']]
                out_pi_test = pd.concat([out_pi_test, out_pi_test2], axis = 1)
                out_mu_test = pd.concat([out_mu_test, out_mu_test2], axis = 1)
                out_sigma_test = pd.concat([out_sigma_test, out_sigma_test2], axis = 1)
            i = i +1


        for i in range(0, n_samples_batch):
            if i == 0:
                samples = pd.DataFrame(sample_from_mixture_f(X_pred, np.array(out_pi_test), np.array(out_mu_test), np.array(out_sigma_test), self.dist))
                samples.columns = ['sample_' + str(i+1)]
                print(str(i+1) + '... /'+ str(n_samples_batch))
            else:
                samples2 = pd.DataFrame(sample_from_mixture_f(X_pred, np.array(out_pi_test), np.array(out_mu_test), np.array(out_sigma_test), self.dist))
                samples2.columns = ['sample_' + str(i+1)]
                samples = pd.concat([samples, samples2], axis = 1)
            if (i+1)%100 == 0:
                print(str(i+1) + '... /'+ str(n_samples_batch))

        return samples

    def plot_pred_vs_true(self, y_pred, y_true, y_scaler = None):
        if y_scaler != None:
            y_true = y_scaler.inverse_transform(y_true)
        else:
            pass

        all_preds = pd.DataFrame(y_true)
        all_preds.columns = ['Y_TRUE']
        all_preds['Y_PRED_MEAN'] = y_pred['Y_PRED_MEAN'] #y_pred['Y_PRED_MEAN']
        all_preds['Y_PRED_LOWER'] =  y_pred['Y_PRED_LOWER'] #y_pred['Y_PRED_LOWER']
        all_preds['Y_PRED_UPPER'] =  y_pred['Y_PRED_UPPER'] #y_pred['Y_PRED_UPPER']

        all_preds = all_preds.sort_values(by = ['Y_TRUE'])
        all_preds = all_preds.reset_index(drop = True)

        stat_r2 = r2_score(all_preds['Y_TRUE'], all_preds['Y_PRED_MEAN'])
        stat_rmse = math.sqrt(mean_squared_error(all_preds['Y_TRUE'], all_preds['Y_PRED_MEAN']))
        stat_mae = mean_absolute_error(all_preds['Y_TRUE'], all_preds['Y_PRED_MEAN'])
        plt.scatter(y_pred['Y_PRED_MEAN'], y_true)
        plt.title('Y_TRUE vs Y_PRED : LINEAR RELATION' + '\n' + 'R2:' + str(round(stat_r2, 4)) + '\n' + 'RMSE:' + str(round(stat_rmse, 2)) + '\n' + 'MAE:' + str(round(stat_mae, 2)))

    def plot_distribution_fit(self, n_samples_batch = 1, alpha = 0.2, y_scaler = None):

        X_pred = self._X.copy()

        y_sampled = self.sample_from_mixture(X_pred = X_pred, n_samples_batch = n_samples_batch, y_scaler = y_scaler)

        for col in y_sampled.columns:
            sns.kdeplot(y_sampled[col], shade=True, alpha = alpha)
        sns.kdeplot(self._y.ravel(), shade=True, linewidth = 2.5, label = 'True dist')

    def plot_all_distribution_fit(self, n_samples_batch = 1, alpha = 0.2, y_scaler = None):

        X_pred = self._X.copy()

        all_preds = self.predict_dist(X_pred, q = 0.95, y_scaler = None)

        i = 0
        for elem in all_preds:
            sns.kdeplot(elem['Y_PRED_MEAN'], shade=True, alpha = 0.15, label = 'fitted_mixture_' + str(i))
            i = i +1
        sns.kdeplot(self._y.ravel(), shade=True, label = 'True dist')

    def plot_samples_vs_true(self, X_pred, y_pred, alpha = 0.4, non_linear = False, y_scaler = None):


        y_sampled = self.sample_from_mixture(X_pred = X_pred, n_samples_batch = 1, y_scaler = y_scaler)


        if X_pred.shape[1] > 1:
            if non_linear == False:
                X_1d = PCA(n_components = 1).fit_transform(X_pred)
            else:
                X_1d = UMAP(n_components = 1).fit_transform(X_pred)
        else:
            X_1d = X_pred.copy()

        plt.scatter(X_1d, y_sampled, alpha = alpha, label = 'Generated sample')
        plt.scatter(X_1d, y_pred, alpha = alpha, label = 'True')
        plt.legend()
        plt.show()