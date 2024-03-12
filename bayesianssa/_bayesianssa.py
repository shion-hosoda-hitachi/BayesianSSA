import os
import sys
import subprocess
from multiprocessing import Pool
from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sympy.matrices import Matrix, eye, zeros, ones, diag
import sympy as sy
from scipy.special import betaln


def calculate_kernel(matrix, for_debug=False):
    # calculate the analytical kernel of matrix
    matrix_nullspace = matrix.nullspace()
    if len(matrix_nullspace) == 0:
        return []
    kernel = sy.BlockMatrix(matrix_nullspace).as_explicit()
    if for_debug:
        kernel = sy.BlockMatrix([[kernel[:, :-1], 
                                  kernel[:, -2] - kernel[:, -1]]]).as_explicit()
    return kernel


def return_ndarray_sign(S, threshold=1.0e-10):
    I, J = S.shape
    SSign = np.zeros((I, J), dtype=object)
    for i in range(I):
        for j in range(J):
            SSignij = None
            if abs(S[i][j]) < threshold:
                SSignij = '0'
            elif S[i][j] > 0:
                SSignij = '+'
            elif S[i][j] < 0:
                SSignij = '-'
            SSign[i][j] = SSignij
    return SSign


def judge_indefinite(signs):
    sign = signs[0].copy()
    S_indefinite = np.any(signs != sign, axis=0)
    for i, Si_indefinite in enumerate(S_indefinite):
        for j, Sij_indefinite in enumerate(Si_indefinite):
            if Sij_indefinite:
                sign[i][j] = '±'
    return sign


def chop(S, threshold):
    S[abs(S) < threshold] = 0
    return S


class PointSSA(object):
    # conduct SSA with one r value.
    def __init__(self, J, M, c, d, indices, verbose=False, for_debug=False):
        self.J = J
        self.M = M
        self.c = c
        self.d = d
        self.indices = indices
        if len(c) != 0:
            self.K = c.shape[1]
        if len(d) != 0:
            self.L = d.shape[1]
        self.r = np.zeros((self.J, self.M))
        self.verbose = verbose
        self.for_debug = for_debug

    def _make_r_matrix(self, r_vec):
        # scipy.sparse enables rewriting this to "coo_matrix((r_vec, (index0, index1))).toarray()"
        for jm, index in enumerate(self.indices):
            self.r[index[0]][index[1]] = r_vec[jm]
        if self.verbose:
            print("Made rate matrix.")

    def _make_matrix_A(self):
        self.A = np.block([self.r, -self.c])
        # If A is not a square matrix, 
        # the conserved quantity bases have to be calculated
        if self.A.shape[0] != self.A.shape[1]:
            B = np.block([[-self.d.T, 
                           np.zeros((self.L, self.K))]])
            self.A = np.block([[self.A], [B]])
        if self.verbose:
            print("Made the matrix A.")
        
    def _make_sensitivity_matrix(self):
        self.S = np.linalg.inv(-self.A)
        self.S = chop(self.S, 1.0e-10)
        self.T = np.dot(self.c, self.S[-self.K:, :])
        self.T = chop(self.T, 1.0e-10)
        if self.verbose:
            print("Made the sensitivity matrix.")
        self.S_sign = return_ndarray_sign(self.S)
        self.T_sign = return_ndarray_sign(self.T)
        
    def run(self, r_vec):
        self._make_r_matrix(r_vec)
        self._make_matrix_A()
        self._make_sensitivity_matrix()
        
    def _combine_S_and_T(self, S, T):
        return np.block([[S[:self.M, :]], [T]])
        
    def get_STQ(self):
        self.ST = self._combine_S_and_T(self.S, self.T)
        self.Q = self._combine_S_and_T(self.S_sign, self.T_sign)
        return self.ST, self.Q
        
    def get_A(self):
        return self.A


class NumericalSSA(object):
    # conduct SSA with random several r values
    def __init__(self, nu, indices, 
                 r_dist=np.random.lognormal, r_dist_param=(0, 1), 
                 n_iter=10000, save_memory=True, verbose=False, 
                 for_debug=False):
        is_metabolite_flow = (nu > 0).any(1) & (nu < 0).any(1)
        is_all_metabolite_flow = is_metabolite_flow.all()
        if not is_all_metabolite_flow:
            non_flow_metabolites = nu.loc[~is_metabolite_flow].index.tolist()
            non_flow_metabolite_str = ', '.join(non_flow_metabolites)
            print(f'The following metabolites cannot flow: {non_flow_metabolite_str}')
            sys.exit(1)
        self.M = nu.shape[0]
        self.J = nu.shape[1]
        self.R = len(indices)
        self.indices = indices
        self.nu = nu
        self.r_dist = r_dist
        self.r_dist_param = r_dist_param
        self.n_iter = n_iter
        # store samples of sensitivities or not
        self.save_memory = save_memory
        self.verbose = verbose
        self.for_debug = for_debug

    def _generate_r_vec(self):
        r_vec = self.r_dist(*self.r_dist_param, size=self.R)
        return r_vec

    def _calculate_nu_kernel(self):
        # Since we need an analytical solution, we here use sympy.
        self.c = calculate_kernel(Matrix(self.nu), self.for_debug)
        if len(self.c) == 0:
            self.K = 0
        else:
            self.c = np.array(self.c.evalf(), dtype=float)
            self.K = self.c.shape[1]
            if self.verbose:
                print("Obtained the kernel of nu.")

    def _calculate_nu_t_kernel(self):
        self.d = calculate_kernel(Matrix(self.nu.T))
        if len(self.d) == 0:
            self.L = 0
        else:
            self.d = np.array(self.d.evalf(), dtype=float)
            self.L = self.d.shape[1]
            if self.verbose:
                print("Obtained the kernel of the transpose of nu.")

    def _store_names(self):
        self.reac_names = self.nu.columns.tolist()
        self.metab_names = self.nu.index.tolist()
        self.flux_names = ['Flux' + str(i) for i in range(self.K)]
        self.cons_quan_names = ['CQ' + str(i) for i in range(self.L)]

    def _sum_up_Q(self, this_Q):
        self.Q_sum['positive'] += (this_Q == '+').astype(int)
        # positive/negative response may yield zero
        self.Q_sum['positive'] += (this_Q == '0').astype(int)
        self.Q_sum['negative'] += (this_Q == '-').astype(int)
        self.Q_sum['negative'] += (this_Q == '0').astype(int)
        self.Q_sum['zero'] += (this_Q == '0').astype(int)
    
    def _Q_like_array_to_df(self, array):
        df = pd.DataFrame(array, 
                          index=self.metab_names+self.reac_names,
                          columns=self.reac_names+self.cons_quan_names)
        return df

    def _judge_Q_consensus(self):
        def judge_indefinite(sign_sum_mat_dict):
            sign_mat = np.full(sign_sum_mat_dict['positive'].shape, '±')
            sign_mat[sign_sum_mat_dict['positive'] == self.n_iter] = '+'
            sign_mat[sign_sum_mat_dict['negative'] == self.n_iter] = '-'
            sign_mat[sign_sum_mat_dict['zero'] == self.n_iter] = '0'
            return sign_mat
        self.Q_consensus = judge_indefinite(self.Q_sum)

    def _compute_confidence(self):
        confidence = {}
        confidence['positive'] = self.Q_sum['positive'] / self.n_iter
        confidence['negative'] = self.Q_sum['negative'] / self.n_iter
        confidence['positive'][self.Q_consensus == '0'] = 0.0
        confidence['negative'][self.Q_consensus == '0'] = 0.0
        self.confidence = confidence

    def run(self):
        self._calculate_nu_kernel()
        self._calculate_nu_t_kernel()
        self._store_names()
        self.Q_sum = {}
        self.Q_sum['positive'] = np.zeros((self.M + self.J, 
                                                 self.J + self.L))
        self.Q_sum['negative'] = np.zeros_like(
            self.Q_sum['positive'])
        self.Q_sum['zero'] = np.zeros_like(
            self.Q_sum['positive'])
        self.r_vecs = []
        self.STs = []
        self.Qs = []
        self.As = []
        point_ssa = PointSSA(self.J, self.M, self.c, self.d, self.indices, 
                             verbose=self.verbose, for_debug=self.for_debug)
        for t in tqdm(range(self.n_iter)):
            r_vec = self._generate_r_vec()
            self.r_vecs.append(r_vec)
            point_ssa.run(r_vec)
            this_ST, this_Q = point_ssa.get_STQ()
            this_A = point_ssa.get_A()
            self._sum_up_Q(this_Q)
            if not self.save_memory:
                this_Q = self._Q_like_array_to_df(this_Q)
                self.STs.append(this_ST)
                self.Qs.append(this_Q)
                self.As.append(this_A)
        self._judge_Q_consensus()
        self._compute_confidence()
        self.Q_consensus = self._Q_like_array_to_df(self.Q_consensus)
        self.confidence['positive'] = self._Q_like_array_to_df(
            self.confidence['positive'])
        self.confidence['negative'] = self._Q_like_array_to_df(
            self.confidence['negative'])


class BayesianSSA(NumericalSSA):
    def __init__(self, nu, indices, a=None, b=None,
                 r_dist=np.random.lognormal, r_dist_param=(0, 1), 
                 n_iter=10000, verbose=False, for_debug=False):
        super().__init__(nu, indices, 
                         r_dist=r_dist, r_dist_param=r_dist_param, 
                         n_iter=n_iter, save_memory=False, 
                         verbose=verbose, for_debug=for_debug)
        self.a = a
        self.b = b
        self.n = {}
        self.ln_g = np.ones(self.n_iter)

    def _calc_Qs_weighted_sum(self, sign_mark, rho=1):
        signs_arr = np.array(self.Qs)
        signs_arr = np.where(signs_arr == sign_mark, rho, 1-rho)
        return (signs_arr.transpose() @ self.r_dist_param).T

    def _empirical_run(self):
        columns = self.confidence['positive'].columns
        index = self.confidence['positive'].index
        self.confidence['positive'] = pd.DataFrame(
            self._calc_Qs_weighted_sum('+'), columns=columns, index=index)
        self.confidence['negative'] = pd.DataFrame(
            self._calc_Qs_weighted_sum('-'), columns=columns, index=index)

    def run(self):
        if self.r_dist == 'empirical':
            self._empirical_run()
        else:
            super().run()
            self.r_dist = 'empirical'
            self.r_dist_param = np.ones(self.n_iter) / self.n_iter

    def _return_correct_or_not(self, up_down, row_name, col_name):
        if up_down == 0:
            up_down = '-'
        elif up_down == 1:
            up_down = '+'
        correct_or_not = np.array([this_Q.loc[row_name, col_name]
                                   == up_down 
                                   for this_Q in self.Qs])
        return correct_or_not

    def _calc_beta_prod(self):
        ln_g = np.zeros(self.n_iter)
        # for each m, j
        for row_name, col_name in self.n.keys():
            n_mj_pos = self.n[(row_name, col_name)]['positive']
            n_mj_neg = self.n[(row_name, col_name)]['negative']
            # log likelihood of rv making positive/negative prediction
            ln_l_pos = None
            ln_l_neg = None
            # Beta(a^, b^)
            ln_l_pos = betaln(self.a + n_mj_pos, self.b + n_mj_neg)
            ln_l_neg = betaln(self.a + n_mj_neg, self.b + n_mj_pos)
            ln_g_mj = np.full(self.n_iter, ln_l_neg)
            # which s_hat is positive
            positive_or_not = self._return_correct_or_not(1, row_name, col_name)
            ln_g_mj[positive_or_not] = ln_l_pos
            ln_g += ln_g_mj
        return ln_g

    def update_distributions(self, ex_results):
        for up_down, (row_name, col_name) in zip(ex_results['up/down'], 
                                                 ex_results['target']):
            # first observation of (m, j) results
            if not (row_name, col_name) in self.n.keys():
                self.n[(row_name, col_name)] = {'positive': 0, 'negative': 0}
            if up_down == 1:
                self.n[(row_name, col_name)]['positive'] += 1
            elif up_down == 0:
                self.n[(row_name, col_name)]['negative'] += 1
        self.ln_g = self._calc_beta_prod()
        # normalization
        self.r_dist_param = np.exp(self.ln_g - logsumexp(self.ln_g))

    def calculate_predictive_prob(self, row_name, col_name, pos_or_neg):
        # this function first calculates the probability that positive is correct
        numer_mod = None
        denom_mod = None
        positive_or_not = self._return_correct_or_not(1, row_name, col_name)
        # in case that (m, j) of new data is not new
        if (row_name, col_name) in self.n.keys():
            # For positive prediction
            # True positive
            a_hat_pos = self.a + self.n[(row_name, col_name)]['positive']
            # False negative
            b_hat_pos = self.b + self.n[(row_name, col_name)]['negative']
            # For negative prediction
            # True negative
            a_hat_neg = self.a + self.n[(row_name, col_name)]['negative']
            # False positive
            b_hat_neg = self.b + self.n[(row_name, col_name)]['positive']
            # Beta(a^, b^) -> Beta(a^ + 1, b^) or Beta(a^, b^ + 1)
            numer_mod = np.full(self.n_iter, np.log(b_hat_neg / (a_hat_neg + b_hat_neg)))
            numer_mod[positive_or_not] = np.log(a_hat_pos / (a_hat_pos + b_hat_pos))
            denom_mod = 0
        else:
            numer_mod = np.full(self.n_iter, betaln(self.a, self.b + 1))
            numer_mod[positive_or_not] = betaln(self.a + 1, self.b)
            denom_mod = betaln(self.a, self.b)
        numer = logsumexp(self.ln_g + numer_mod)
        denom = logsumexp(self.ln_g + denom_mod)
        pos_pred_prob = np.exp(numer - denom)
        if pos_or_neg == 1:
            return pos_pred_prob
        if pos_or_neg == 0:
            return 1 - pos_pred_prob
        