import numpy as np
import sys
import matplotlib.pyplot as pp

np.random.seed(0)

class BoyanChain(object):
	
	def __init__(self, n):
		''' n is the size of the Boyan Chain. '''
		self.n = n
		self.state = n-1
		self._init_features()

	def _init_features(self):
		''' Initialize the state vector. Right now this only works for the 13-state Boyan chain.'''
		self.features = {}

		tmp = [1., 0., 0., 0.]
		state = self.n-1
		ix = 0
		while ix < 3:
			self.features[state] = np.asarray(tmp)
			state -= 1
			
			tmp[ix] -= 0.25
			tmp[ix+1] += 0.25
			if tmp[ix] == 0: 
				ix += 1
		self.features[0] = np.asarray(tmp)


	def _get_feature(self, state):
		return self.features[state]

	def reset(self):
		self.state = self.n-1

	def step(self):
		''' Return the next state. 
			Return None if episode is over (reached state 0).
			Otherwise return a reward and the feature for the next state.
		'''
		if self.state >= 2:
			if np.random.uniform() > 0.5:
				self.state, reward = self.state - 2, -3.
			else:
				self.state, reward = self.state - 1, -3.
		elif self.state == 1:
			self.state, reward = self.state - 1, -2.
		else:
			return (None, None)

		return (self._get_feature(self.state), reward)

class LSTD_lam(object):
	def __init__(self, dim, gamma, lam):
		''' Initialize the running averages will will keep track of.
			dim is the size of the feature vector.
		'''
		self.dim = dim
		self.t = 0
		self.gamma = gamma
		self.lam = lam
		self.A = np.zeros((dim, dim))
		self.b = np.zeros((dim,))

	def reset(self, s):
		# Reset the eligibility trace at the beginning of each episode.
		self.z = s

	def update(self, s, sp1, r):
		# Update the running averages.
		delta = s - self.gamma*sp1
		self.t += 1
		if self.t == 1:
			self.z = s
			self.A = np.outer(self.z, delta)
			self.b = self.z*r
		else:
			self.A = self.A + np.outer(self.z, delta) * (1. / (self.t-1))
			self.b = self.b + self.z*r * (1. / (self.t-1))
			self.A = self.A * (float(self.t-1) / float(self.t))
			self.b = self.b * (float(self.t-1) / float(self.t))
		self.z = self.lam*self.z + sp1

	def get_theta(self):
		# Return the current estimate.
		return np.linalg.solve(self.A + np.eye(self.dim)*0.01, self.b.reshape((self.dim, 1)))

class LSTD(object):
	def __init__(self, dim, gamma):
		''' Initialize the running averages will will keep track of.
			dim is the size of the feature vector.
		'''
		self.dim = dim
		self.t = 0
		self.gamma = gamma
		self.A = np.zeros((dim, dim))
		self.b = np.zeros((dim,))

	def update(self, s, sp1, r):
		# Update the running averages with the new values.
		delta = s - self.gamma*sp1
		self.t += 1
		if self.t == 1:
			self.A = np.outer(s, delta)
			self.b = s*r
		else:
			self.A = self.A + np.outer(s, delta) * (1. / (self.t-1))
			self.b = self.b + s*r * (1. / (self.t-1)) 
			self.A = self.A * (float(self.t-1) / float(self.t))
			self.b = self.b * (float(self.t-1) / float(self.t))

		
	def get_theta(self):
		# Return the current estimate.
		return np.linalg.solve(self.A + np.eye(self.dim)*0.01, self.b.reshape((self.dim, 1)))

class RLSTD(object):

	def __init__(self, dim, gamma, eps):
		''' Initialize the running averages will will keep track of.
			dim is the size of the feature vector.
		'''
		self.dim = dim
		self.t = 0
		self.gamma = gamma
		self.A_inv = np.eye(dim)*1./eps
		self.b = np.zeros((dim,))

	def update(self, s, sp1, r):
		# Update estimate of the inverse using the Sherman-Morrison formula.
		delta = s - self.gamma*sp1
		v = np.dot(self.A_inv.T, delta)
		self.A_inv = self.A_inv - np.outer(np.dot(self.A_inv, s), v)/(1 + np.dot(v, s))
		self.b = self.b + r*s
		
	def get_theta(self):
		# Return the current estimate.
		return np.dot(self.A_inv, self.b)

class TD(object):
	def __init__(self, dim, gamma, alpha):
		self.theta = np.zeros((dim,))
		self.gamma = gamma
		self.alpha = alpha

	def update(self, s, sp1, r):
		d = (r + self.gamma* np.dot(self.theta, sp1) - np.dot(self.theta, s))
		self.theta = self.theta + self.alpha*d*s

	def get_theta(self):
		return self.theta

def error(x, y):
	return np.sqrt(np.sum((x-y)**2))

N_EPS = 100
N_TRIALS = 100

FONT_SIZE=24
if __name__ == '__main__':
	
	boyan = BoyanChain(13)

	estimates = {'td':np.zeros((N_TRIALS, N_EPS)), 'lstd':np.zeros((N_TRIALS, N_EPS)), 'rlstd':np.zeros((N_TRIALS, N_EPS))}
	for t_ix in xrange(0, N_TRIALS):
		lstd = LSTD(4, 1.)
		rlstd = RLSTD(4, 1., 5)
		lstd_lam = LSTD_lam(4, 1., 1.)
		td = TD(4, 1., 0.05)

		true_theta = np.asarray([-24., -16., -8., 0.])
		for ep_ix in xrange(0, N_EPS):
			boyan.reset()

			prev_state, state = boyan._get_feature(boyan.state), None
			lstd_lam.reset(prev_state)
			# Execute an episode.
			while True:
				state, reward = boyan.step()
				if state is None:
					break
				# Update each algorithms estimates.
				rlstd.update(prev_state, state, reward)
				lstd.update(prev_state, state, reward)
				td.update(prev_state, state, reward)
				lstd_lam.update(prev_state, state, reward)
				prev_state = state
			# Get the current error.
			estimates['td'][t_ix, ep_ix] = error(true_theta, td.get_theta())
			estimates['lstd'][t_ix, ep_ix] = error(true_theta, lstd.get_theta().T)
			estimates['rlstd'][t_ix, ep_ix] = error(true_theta, rlstd.get_theta())

	line_td, = pp.plot(range(0, N_EPS), np.mean(estimates['td'], axis=0), 'g-', label='TD(0)')
	line_lstd, = pp.plot(range(0, N_EPS), np.mean(estimates['lstd'], axis=0), 'r-', label='LSTD')
	line_rlstd, = pp.plot(range(0, N_EPS), np.mean(estimates['rlstd'], axis=0), 'b-', label='RLSTD')
	pp.ylabel("RMSE")
	pp.xlabel("Episodes")
	pp.legend(handles=[line_td, line_lstd, line_rlstd], prop={'size':FONT_SIZE})
	pp.show()

	# Choose different step-sizes for TD(0).
	estimates = {}
	for alpha in [0.001, 0.01, 0.1, 1.]:
		td = TD(4, 1, alpha)
		estimates[alpha] = []

		for ep in xrange(0, N_EPS):
			print 'Episode: %d' % ep
			boyan.reset()

			prev_state, state = boyan._get_feature(boyan.state), None
			while True:
				state, reward = boyan.step()
				if state is None:
					break
				td.update(prev_state, state, reward)
				prev_state = state
			estimates[alpha].append(error(true_theta, td.get_theta()))


	line1, = pp.plot(range(0, N_EPS), estimates[0.001], 'g-', label='alpha=0.001')
	line2, = pp.plot(range(0, N_EPS), estimates[0.01], 'r-', label='alpha=0.01')
	line3, = pp.plot(range(0, N_EPS), estimates[0.1], 'b-', label='alpha=0.1')
	line4, = pp.plot(range(0, N_EPS), estimates[1.], 'k-', label='alpha=1')
	pp.ylabel("RMSE")
	pp.xlabel("Episodes")

	pp.legend(handles=[line1, line2, line3, line4], prop={'size':FONT_SIZE})
	pp.show()

	# Choose different epsilons for RLSTD(0).
	estimates = {}
	for eps in [0.001, 0.01, 0.1, 1., 10., 100., 1000, ]:
		rlstd = RLSTD(4, 1, eps)
		estimates[eps] = []

		for ep in xrange(0, N_EPS):
			print 'Episode: %d' % ep
			boyan.reset()

			prev_state, state = boyan._get_feature(boyan.state), None
			while True:
				state, reward = boyan.step()
				if state is None:
					break
				rlstd.update(prev_state, state, reward)
				prev_state = state
			estimates[eps].append(error(true_theta, rlstd.get_theta()))

	line1, = pp.plot(range(0, N_EPS), estimates[1.], 'k-', label='eps=1')
	line2, = pp.plot(range(0, N_EPS), estimates[10.], 'g-', label='eps=10')
	line3, = pp.plot(range(0, N_EPS), estimates[100], 'b-', label='eps=100')
	pp.legend(handles=[line1, line2, line3], prop={'size':FONT_SIZE})
	pp.ylabel("RMSE")
	pp.xlabel("Episodes")
	pp.show()

