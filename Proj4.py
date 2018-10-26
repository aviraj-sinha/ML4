#https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns




#creating an equal amount of data in each categoty in data fram iot traffic
b_thermostat = pd.read_csv('Ecobee_Thermostat/benign_traffic.csv')
limit = b_thermostat.shape[0]
b_thermostat = b_thermostat.iloc[:int(limit/3)]
g_thermostat = pd.read_csv('Ecobee_Thermostat/gafgyt_attacks/scan.csv')
g_thermostat = g_thermostat.iloc[:limit]
g_thermostat.shape
m_thermostat = pd.read_csv('Ecobee_Thermostat/mirai_attacks/scan.csv')
m_thermostat = m_thermostat.iloc[:limit]
m_thermostat.shape

b_camera = pd.read_csv('SimpleHomeCamera/benign_traffic.csv')
b_camera = b_camera.iloc[:int(limit/3)]
g_camera = pd.read_csv('SimpleHomeCamera/gafgyt_attacks/scan.csv')
g_camera = g_camera.iloc[:limit]
g_camera.shape
m_camera = pd.read_csv('SimpleHomeCamera/mirai_attacks/scan.csv')
m_camera = m_camera.iloc[:limit]
m_camera.shape

b_doorbell = pd.read_csv('Danmini_Doorbell/benign_traffic.csv')
b_doorbell = b_doorbell.iloc[:int(limit/3)]
g_doorbell = pd.read_csv('Danmini_Doorbell/gafgyt_attacks/scan.csv')
g_doorbell = g_doorbell.iloc[:limit]
g_doorbell.shape
m_doorbell = pd.read_csv('Danmini_Doorbell/mirai_attacks/scan.csv')
m_doorbell = m_doorbell.iloc[:limit]
m_doorbell.shape

b_camera["Class"] = "b"
m_camera["Class"] = "m_c"
g_camera["Class"] = "g_c"

b_doorbell["Class"] = "b"
m_doorbell["Class"] = "m_d"
g_doorbell["Class"] = "g_d"

b_thermostat["Class"] = "b"
m_thermostat["Class"] = "m_t"
g_thermostat["Class"] = "g_t"

iot_traffic = []
all_dfs = [m_camera, g_camera, b_camera,
             b_doorbell, m_doorbell, g_doorbell,
             b_thermostat, m_thermostat, g_thermostat]

iot_traffic = pd.concat(all_dfs).reset_index(drop=True)
iot_traffic.head()
iot_traffic.shape
list(iot_traffic.columns)





# sns.heatmap(iot_traffic.corr())
# plt.show()








#show the equal balance of classes
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
sns.countplot(x="Class", data=iot_traffic)
plt.show()

#formatting data
X = iot_traffic.drop(columns=["Class"])
y = iot_traffic["Class"]

X.shape
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

X = SelectKBest(f_classif, k=10).fit_transform(X, y)






n_samples, n_features = X.shape
n_classes = len(np.unique(y))

print("n_samples: {}".format(n_samples))
print("n_features: {}".format(n_features))
print("n_classes: {}".format(n_classes))
print(np.unique(y))




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)







#do pca for feature reduction
def plot_explained_variance(pca):
    import plotly
    from plotly.graph_objs import Bar, Line
    from plotly.graph_objs import Scatter, Layout
    from plotly.graph_objs.scatter import Marker
    from plotly.graph_objs.layout import XAxis, YAxis
    plotly.offline.init_notebook_mode() # run at the start of every notebook

    explained_var = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(explained_var)

    plotly.offline.iplot({
        "data": [Bar(y=explained_var, name='individual explained variance'),
                 Scatter(y=cum_var_exp, name='cumulative explained variance')
            ],
        "layout": Layout(xaxis=XAxis(title='Principal components'), yaxis=YAxis(title='Explained variance ratio'))
    })


#choose 5 principle components
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
df_normalized = pd.DataFrame(np_scaled)


pca = PCA(n_components=10)
X_pca = pca.fit(df_normalized)
plot_explained_variance(pca)

























# run logistic regression and pto explain why recall_score is best; we want recall of botnet infections to be high

from sklearn.metrics import recall_score, make_scorer
from sklearn.pipeline import Pipeline
pipe_lr = Pipeline([('scaling', StandardScaler()),
                    ('pca', PCA(n_components=5)),
                    ('clf', LogisticRegression(random_state=1))])


from sklearn.metrics import classification_report
y_hat = pipe_lr.fit(X_train,y_train).predict(X_test)
print(classification_report(y_test, y_hat))

my_scorer = make_scorer(score_func=recall_score, pos_label=1, greater_is_better=True, average='macro')
print('Recall Score of: ',recall_score(y_test,y_hat,average='macro'))
cm = metrics.confusion_matrix(y_test, y_hat)
print(cm)


sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = '0 benign; 1 galydart; 2 mira'
plt.title(all_sample_title, size = 15);
plt.show()




#galydary was hardest to classify; so far detection is alright but finding the device it is on needs work
# ideally a custom method with that would make misclassification of galydart as mirai not that bad but as bening very bad
# also being classified as a wrong device but still as a infected would be better than benign.












#the kfolds of the cross validation would be best to see the multiple results of the classification success of the machine learning algorithm on this limited dataset
#the stratification is not needed to even out variance from class imbalances. Also having more infected data would be realistic in helping increase the recall rate.
#since the ids would know these botnets exist they would be able to train ahead of time on a decent amount of data;
#furthermore ids would be able to similary signaures of unknown botnets but not be able prevent them ahead of time the way an ips (intrusion prevention system would)
# Since the detection is not as urgent as prevention it would also have multiple attempts to detect.


from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold



#select cross validation
cv = StratifiedKFold(n_splits=10)
# select evaluation criteria
my_scorer = make_scorer(recall_score, average="macro")
# run model training and cross validation
per_fold_eval_criteria = cross_val_score(estimator=pipe_lr, X=X, y=y, cv=cv, scoring=my_scorer)

plt.bar(range(len(per_fold_eval_criteria)),per_fold_eval_criteria)
plt.ylim([min(per_fold_eval_criteria)-0.01,max(per_fold_eval_criteria)])
plt.show()






from sklearn.model_selection import GridSearchCV

pipe_lr = Pipeline([('scaling', StandardScaler()),
('pca', PCA(n_components=5)),
('clf', LogisticRegression(random_state=1))])

param_grid = {
    'clf__C':np.logspace(0, 4, 10)
    }

gs = GridSearchCV(pipe_lr, param_grid,scoring=my_scorer, cv=5, verbose=0)

best_model = gs.fit(X_train, y_train)

# print('Best Penalty:', best_model.best_estimator_.get_params()['pca__n_components'])
print('Best C:', best_model.best_estimator_.get_params()['clf__C'])

best_model.predict(X_test)
recall_score(y_test, best_model.predict(X_test), average="macro")
# pipe_lr.get_params().keys()
y_test













# Example adapted from https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch12/ch12.ipynb
# Original Author: Sebastian Raschka

# This is the optional book we use in the course, excellent intuitions and straightforward programming examples
# please note, however, that this code has been manipulated to reflect our assumptions and notation.
from scipy.special import expit
import sys
import pandas as pd

# start with a simple base classifier, which can't be fit or predicted
# it only has internal classes to be used by classes that will subclass it
from sklearn.base import BaseEstimator, ClassifierMixin
class TwoLayerPerceptronBase(BaseEstimator):
    def __init__(self, cost="quad", layers =  2, n_hidden=30,
                 C=0.0, epochs=500, eta=0.001, random_state=None, alpha=0.0, decrease_const=0.0, shuffle=True,
                              minibatches=1):
        np.random.seed(random_state)
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

        self.n_hidden = n_hidden
        self.C = C
        self.epochs = epochs
        self.eta = eta
        self.layers = layers
        self.cost=cost


    @staticmethod
    def _encode_labels(y):
        """Encode labels into one-hot representation"""
        onehot = pd.get_dummies(y).values.T

        return onehot

    def _initialize_weights(self):
        """Initialize weights with small random numbers."""
         # common practice to start with zero bias

        w = []

        W1 = np.random.randn(self.n_hidden, self.n_features_ + 1)
        W1[:,1:] = W1[:,1:]/np.sqrt(self.n_features_+1) # don't saturate the neuron
        W1[:,:1] = 0
        w.append(W1)

        for i in range(self.layers-2):
            W = np.random.randn(self.n_hidden, self.n_hidden + 1)
            W[:,1:] = W[:,1:]/np.sqrt(self.n_hidden+1) # don't saturate the neuron
            W[:,:1] = 0 # common practice to start with zero bias
            w.append(W)

        W2 = np.random.randn(self.n_output_, self.n_hidden + 1)
        W2[:,1:] = W2[:,1:]/np.sqrt(self.n_hidden+1) # don't saturate the neuron
        W2[:,:1] = 0 # common practice to start with zero bias
        w.append(W2)

        return np.array(w)


    def _phi(self,z):
        """Use scipy.special.expit to avoid overflow"""
        # 1.0 / (1.0 + np.exp(-z))
        return expit(z)

    @staticmethod
    def _add_bias_unit(X, how='column'):
        """Add bias unit (column or row of 1s) to array at index 0"""
        if how == 'column':
            ones = np.ones((X.shape[0], 1))
            X_new = np.hstack((ones, X))
        elif how == 'row':
            ones = np.ones((1, X.shape[1]))
            X_new = np.vstack((ones, X))
        return X_new

    @staticmethod
    def _L2_reg(lambda_, W, layers):
        """Compute L2-regularization cost"""
        mean = 0
        for i in range(0, layers):
            mean += np.mean(W[i][:, 1:] ** 2)
        return (lambda_/2.0) * np.sqrt(mean)

    def _cost(self,A_top,Y_enc,W):
        '''Get the objective function value'''
        if self.cost=="entr":
            cost = -np.mean(np.nan_to_num((Y_enc*np.log(A_top)+(1-Y_enc)*np.log(1-A_top))))
        elif self.cost=="quad":
            cost = np.mean((Y_enc-A_top)**2)

        L2_term = self._L2_reg(self.C, W, self.layers)
        return cost + L2_term

    def _feedforward(self, X, W):
        """Compute feedforward step
        """
        A, Z = np.asarray([None]*(self.layers+1)), np.asarray([None]*self.layers)
        A[0] = self._add_bias_unit(X, how='column')
        A[0] = A[0].T
        Z[0] = W[0] @ A[0]
        A[1] = self._phi(Z[0])

        for i in range(1, self.layers):
            A[i] = self._add_bias_unit(A[i], how='row')
            Z[i] = W[i] @ A[i]
            A[i+1] = self._phi(Z[i])

        return A, Z

    def _get_gradient(self, A, Z, Y_enc, W):
        V = np.asarray([None]*(self.layers))

        grads = []

        # vectorized backpropagation
        if self.cost=="entr":
            V[self.layers-1] = (A[self.layers]-Y_enc)
        elif self.cost=="quad":
            V[self.layers-1] = -2*(Y_enc-A[self.layers])*A[self.layers]*(1-A[self.layers])  # last layer sensitivity

        grads.append( V[self.layers-1] @ A[self.layers-1].T) # no bias on final layer)

        for i in range(self.layers-1,0,-1):

            V[i-1] = A[i]*(1-A[i])*(W[i].T @ V[i]) # back prop the sensitivity
            V[i-1] = V[i-1][1:,:]
            grads.insert(0,  V[i-1] @ A[i-1].T) # dont back prop sensitivity of bias


        # regularize weights that are not bias terms
        for i in range(len(grads)):
            grads[i][:, 1:] += W[i][:, 1:] * self.C

        return np.array(grads)


    def predict(self, X):
        """Predict class labels"""
        A, Z = self._feedforward(X, self.W)
        y_pred = np.argmax(A[self.layers], axis=0)
        return y_pred


    def fit(self, X, y):
        print_progress=False
        """ Learn weights from training data. With mini-batch"""
        X_data, y_data = X.copy(), y.copy()
        Y_enc = self._encode_labels(y)

        # init weights and setup matrices
        self.n_features_ = X_data.shape[1]
        self.n_output_ = Y_enc.shape[0]
        self.W = self._initialize_weights()

        delta_W_prev = []
        for i in range(len(self.W)):
            delta_W_prev.append(np.zeros(self.W[i].shape))
        delta_W_prev = np.array(delta_W_prev)
        self.cost_ = []
        self.score_ = []
        self.grad_w = []
        # get starting acc

        self.score_.append(recall_score(mapped_attack(y_data),self.predict(X_data),average="macro"))
        for i in range(self.epochs):

            # adaptive learning rate
            self.eta /= (1 + self.decrease_const*i)

            if print_progress>0 and (i+1)%print_progress==0:
                sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx_shuffle = np.random.permutation(y_data.shape[0])
                X_data, Y_enc, y_data = X_data[idx_shuffle], Y_enc[:, idx_shuffle], y_data[idx_shuffle]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            mini_cost = []
            for idx in mini:

                # feedforward
                A, Z = self._feedforward(X_data[idx], self.W)

                cost = self._cost(A[self.layers],Y_enc[:, idx],self.W)
                mini_cost.append(cost) # this appends cost of mini-batch only

                # compute gradient via backpropagation
                grad = self._get_gradient(A=A, Z=Z, Y_enc=Y_enc[:, idx], W=self.W)

                # momentum calculations
                delta_W = self.eta * grad
                self.W -= (delta_W + (self.alpha * delta_W_prev))
                delta_W_prev = delta_W

            curr_grad=[]
            for i in range(len(self.W)):
                curr_grad.append(np.mean(np.abs(grad[i])))
            self.grad_w.append(curr_grad)

            self.cost_.append(mini_cost)
            self.score_.append(recall_score(mapped_attack(y_data),self.predict(X_data),average="macro"))

        return self




def mapped_attack(y_test):
    mapped_attacks = { "b":0, "g_c":1, "g_d":2, "g_t":3, "m_c":4, "m_d":5, "m_t":6}
    df = pd.DataFrame({"y_t": y_test})
    df['y_t'] = df['y_t'].map(mapped_attacks)
    df = df.fillna(0)
    return df['y_t'].values




#grid search

def score_func(y, y_pred, **kwargs):
    return recall_score(mapped_attack(y),y_pred, average="macro")



my_scorer = make_scorer(score_func)

param_grid = {
    'clf__C':[0.001, 0.01],
    'clf__cost':["entr","quad"],
    'clf__layers':[3]
    }

gs = GridSearchCV(pipe_nn, param_grid,scoring=my_scorer, cv=2, verbose=2)
X_train.shape
best_model = gs.fit(X_train, y_train.values)
# y_train.values.dtype('string')
# print('Best Penalty:', best_model.best_estimator_.get_params()['pca__n_components'])
print('Best C:', best_model.best_estimator_.get_params()['clf__C'])
print(pipe_nn.get_params().keys())
best_model.predict(X_test)
recall_score(mapped_attack(y_test), best_model.predict(X_test), average="macro")
y_test
pipe_lr.get_params().keys()
# print(classification_report(mapped_attack(y_test), y_hat))
# cm = metrics.confusion_matrix(mapped_attack(y_test), y_hat)
# print(cm)



#plot grid search results
gs_results = pd.DataFrame(gs.cv_results_)
gs_results.ftypes
gs_results.plot.line(x='param_clf__C', y='mean_test_score')
sns.barplot(x='param_clf__layers', y='mean_test_score',hue="param_clf__cost", data=gs_results)
a = sns.lineplot(x='param_clf__C', y='mean_test_score', data = gs_results)
gs_results.plot.line(x='param_clf__layers', y='mean_test_score')


y_test
pipe_lr.get_params().keys()





vals = { 'n_hidden':10,
         'C':1e-2, 'epochs':20, 'eta':0.001,
         'alpha':0.001, 'decrease_const':1e-6, 'minibatches':10,
         'shuffle':True,'random_state':1, 'layers':3, 'cost':"entr"}

pipe_nn = Pipeline([('scaling', StandardScaler()),
                    ('pca', PCA(n_components=5)),
                    ('clf', TwoLayerPerceptronBase(**vals))])

pipe_nn.fit(X_train, y_train.values)
y_hat = pipe_nn.predict(X_test)
# y_hat
type(y_test)
print('Test acc:',recall_score(mapped_attack(y_test),y_hat, average="macro"))


grads = np.asarray(pipe_nn.named_steps['clf'].grad_w)

ax = plt.subplot(1,1,1)

for i in range(len(grads[0])):
    plt.plot(np.abs(grads[:,i]), label='w'+str(len(grads[0])-i))
plt.legend()
plt.ylabel('Average gradient magnitude')
plt.xlabel('Iteration')
plt.show()
