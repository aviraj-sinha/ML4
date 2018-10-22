#https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


benign = pd.read_csv('benign_traffic.csv')
benign.shape[0]
gafgyt = pd.read_csv('gafgyt_attacks/scan.csv')
gafgyt = gafgyt.iloc[:benign.shape[0]]
gafgyt.shape
mirai = pd.read_csv('mirai_attacks/scan.csv')
mirai = mirai.iloc[:benign.shape[0]]
mirai.shape

benign.head()
benign["Class"] = "benign"
mirai["Class"] = "mirai"
gafgyt["Class"] = "gafgyt"

gafgyt.head()
all_dfs = [mirai, gafgyt,benign]
iot_traffic = pd.concat(all_dfs).reset_index(drop=True)
iot_traffic.head()
iot_traffic.shape
list(iot_traffic.columns)
# sns.heatmap(iot_traffic.corr())
# plt.show()

sns.countplot(x="Class", data=iot_traffic)
plt.show()

X = iot_traffic.drop(columns=["Class"])
y = iot_traffic["Class"]
X.head()
y.head()
X.shape
y.shape
print(np.unique(y))
n_samples, n_features = X.shape
n_classes = len(np.unique(y))

print("n_samples: {}".format(n_samples))
print("n_features: {}".format(n_features))
print("n_classes: {}".format(n_classes))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, make_scorer
lr = LogisticRegression()
lr.fit(X,y)
print(lr)
my_scorer = make_scorer(score_func=recall_score, pos_label=1, greater_is_better=True, average='macro')
yhat = lr.predict(X)
print('Recall Score of: ',recall_score(y,yhat,average='macro'))

cm = metrics.confusion_matrix(y, yhat)
print(cm)

from sklearn.metrics import classification_report

print(classification_report(y, yhat))


from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(f1_score(y,yhat,average='macro'))
plt.title(all_sample_title, size = 15);

from sklearn.model_selection import StratifiedKFold
clf = LogisticRegression()
#select cross validation
cv = StratifiedKFold(n_splits=10)
# select evaluation criteria
my_scorer = make_scorer(recall_score, average="macro")
# run model training and cross validation
per_fold_eval_criteria = cross_val_score(estimator=clf,
                                    X=X,
                                    y=y,
                                    cv=cv,
                                    scoring=my_scorer
                                   )

plt.bar(range(len(per_fold_eval_criteria)),per_fold_eval_criteria)
plt.ylim([min(per_fold_eval_criteria)-0.01,max(per_fold_eval_criteria)])




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)









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



from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
df_normalized = pd.DataFrame(np_scaled)


pca = PCA(n_components=20)
X_pca = pca.fit(df_normalized)
plot_explained_variance(pca)


from sklearn.pipeline import Pipeline


pipe_lr = Pipeline([('scaling', StandardScaler()),
                    ('pca', PCA(n_components=5)),
                    ('clf', LogisticRegression(random_state=1))])


pipe_lr.fit(X_train,y_train)
print(pipe_lr.score(X_test,y_test))
y_pred = pipe_lr.predict(X_test)












# Example adapted from https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch12/ch12.ipynb
# Original Author: Sebastian Raschka

# This is the optional book we use in the course, excellent intuitions and straightforward programming examples
# please note, however, that this code has been manipulated to reflect our assumptions and notation.
from scipy.special import expit
import sys
import pandas as pd

# start with a simple base classifier, which can't be fit or predicted
# it only has internal classes to be used by classes that will subclass it






class MultiLayerPerceptronBase(object):
    def __init__(self, layers=2, phi='sig', n_hidden=30, cost="entr",
                 C=0.01, epochs=50, eta=0.001, random_state=1):
        np.random.seed(random_state)
        self.n_hidden = n_hidden
        self.l2_C = C
        self.epochs = epochs
        self.eta = eta
        self.phi = phi
        self.cost = cost
        self.layers = layers

    @staticmethod
    def _encode_labels(y):
        """Encode labels into one-hot representation"""
        onehot = pd.get_dummies(y).values.T

        return onehot

    def _initialize_weights(self):
        """Initialize weights with small random numbers."""
        w = []

        W1_num_elems = (self.n_features_ + 1)*self.n_hidden
        W1 = np.random.uniform(-1.0, 1.0, size=W1_num_elems)
        W1 = W1.reshape(self.n_hidden, self.n_features_ + 1) # reshape to be W
        w.append(W1)

        for i in range(self.layers-2):
            W_num_elems = (self.n_hidden)*(self.n_hidden+1)
            W = np.random.uniform(-1.0, 1.0, size=W_num_elems)
            W = W.reshape(self.n_hidden, self.n_hidden + 1)
            w.append(W)

        W2_num_elems = (self.n_hidden + 1)*self.n_output_
        W2 = np.random.uniform(-1.0, 1.0, size=W2_num_elems)
        W2 = W2.reshape(self.n_output_, self.n_hidden + 1)
        w.append(W2)

        return w

    def _phi(self, z):
        if self.phi=='sig':
            return expit(z)
        return z

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
    def _L2_reg(layers, lambda_, W):
        """Compute L2-regularization cost"""
        # only compute for non-bias terms
        mean = 0
        for i in range(0, layers):
            mean += np.mean(W[0][:, 1:] ** 2)
        return (lambda_/2.0) * np.sqrt(mean)


    def _cost(self, A_top,Y_enc,W):
        '''Get the objective function value'''
        if self.cost=="quad":
            cost = np.mean((Y_enc-A_top)**2)
        else:
            cost = -np.mean(np.nan_to_num((Y_enc*np.log(A_top)+(1-Y_enc)*np.log(1-A_top))))

        L2_term = self._L2_reg(self.layers, self.l2_C, W)
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
        """ Compute gradient step using backpropagation.
        """
        V = np.asarray([None]*(self.layers))

        grads = []
        # vectorized backpropagation
        V[self.layers-1] = -2*(Y_enc-A[self.layers])*A[self.layers]*(1-A[self.layers])  # last layer sensitivity
        grads.append( V[self.layers-1] @ A[self.layers-1].T) # no bias on final layer)

        for i in range(self.layers-1,0,-1):

            V[i-1] = A[i]*(1-A[i])*(W[i].T @ V[i]) # back prop the sensitivity
            V[i-1] = V[i-1][1:,:]
            grads.insert(0,  V[i-1] @ A[i-1].T) # dont back prop sensitivity of bias


        # regularize weights that are not bias terms
        for i in range(len(grads)):
            grads[i][:, 1:] += W[i][:, 1:] * self.l2_C

        return grads

    def predict(self, X):
        """Predict class labels"""
        A, _ = self._feedforward(X, self.W)
        A_top = A[self.layers]
        y_pred = np.argmax(A_top, axis=0)
        return y_pred

    def fit(self, X, y, print_progress=False):
        """ Learn weights from training data."""

        X_data, y_data = X.copy(), y.copy()
        Y_enc = self._encode_labels(y)

        # init weights and setup matrices
        self.n_features_ = X_data.shape[1]
        self.n_output_ = Y_enc.shape[0]
        self.W = self._initialize_weights()

        self.cost_ = []
        for i in range(self.epochs):

            if print_progress>0 and (i+1)%print_progress==0:
                sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
                sys.stderr.flush()

            # feedforward all instances
            A, Z = self._feedforward(X_data,self.W)

            cost = self._cost(A[self.layers],Y_enc,self.W)
            self.cost_.append(cost)

            # compute gradient via backpropagation
            grads = self._get_gradient(A=A, Z=Z, Y_enc=Y_enc, W=self.W)

            for i in range(self.layers):
                self.W[i] -= self.eta * grads[i]


        return self


def mapped_attack(y_test):
    ytest = []
    for attack in y_test:
        if attack=="mirai":
            ytest.append(2)
        elif attack=="gafgyt":
            ytest.append(1)
        elif attack=="benign":
            ytest.append(0)
    return ytest


nn = MultiLayerPerceptronBase(phi='sig', layers = 6, cost='quad',n_hidden=30, C=0.1,  epochs=50,  eta=0.001,  random_state=None)
nn.fit(X_train, y_train, print_progress=10)
yhat = nn.predict(X_test)

pipe_nn = Pipeline([('scaling', StandardScaler()),
                    ('pca', PCA(n_components=5)),
                    ('clf', MultiLayerPerceptronBase(phi='sig', layers = 3, cost='quad',n_hidden=30, C=0.1,  epochs=50,  eta=0.001,  random_state=None))])


pipe_nn.fit(X_train,y_train)
# print(pipe_nn.score(X_test,y_test))
y_pred = pipe_nn.predict(X_test)
accuracy_score(mapped_attack(y_test), y_pred)
#classfication results acc,recal,percis, f1, and confusion_matrix
print('Test acc:',accuracy_score(mapped_attack(y_test),yhat))
print(classification_report(mapped_attack(y_test), yhat))
cm = metrics.confusion_matrix(mapped_attack(y_test), yhat)
print(cm)




# params = dict(n_hidden=50, phi='sig', cost='quad',
#               C=0.1, # tradeoff L2 regularizer
#               epochs=50, # iterations
#               eta=0.001,  # learning rate
#               random_state=1)
#
# nn = TwoLayerPerceptron(**params)
# nn.fit(X_train, y_train, print_progress=10)
# yhat = nn.predict(X_test)

#
# #classfication results acc,recal,percis, f1, and confusion_matrix
# print('Test acc:',accuracy_score(ytest,yhat)),
# print(classification_report(mapped_attack(y_test), yhat))
# cm = metrics.confusion_matrix(ytest, yhat)
# print(cm)
#classifying gafgyt as mirai which is not necessearily as bad as being classified as benign



# params = dict(n_hidden=50,
#               C=0.1, # tradeoff L2 regularizer
#               epochs=50, # iterations
#               eta=0.001,  # learning rate
#               random_state=1)

#cost= entr, quad        phi= sig, lin
