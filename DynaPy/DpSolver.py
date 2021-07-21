from .DpConfigurations import *
from .DpTLCD import *
from .DpBuilding import *

import numpy as np
from math import pi, sqrt
import scipy.linalg as linalg
import sympy as sp

ERROR_MSG = "Error"

class ODESolver(object):
	def __init__(self, mass, damping, stiffness, force, configurations=Configurations(), tlcds=None):
		""" ODE solver for dynamics problems.

		:param mass: np.matrix - Mass matrix including structure and damper masses.
		:param damping: np.matrix - Damping matrix including structure and damper damping coefficients.
		:param stiffness: np.matrix - Stiffness matrix including structure and damper stiffness coefficients.
		:param force: np.matrix - Force vector representing force over time in each DOF.
		:param configurations: object - Object containing boundary conditions and other configurations.

				configurations.method: str - Name of the method to be used in the solver. Possible names:
					'Finite Differences', 'Average Acceleration', 'Linear Acceleration', 'RK4'

				configurations.timeStep: float - Time step between iterations.
				configurations.initialDisplacement: float - Initial displacement of the base.
				configurations.initialVelocity: float - Initial velocity of the base

		:return: None
		"""
		self.mass = mass
		self.damping = damping
		self.stiffness = stiffness
		self.force = force
		self.configurations = configurations
		self.tlcds = tlcds
		self.total_tlcd = 0
		if self.tlcds is not None:
			for i in tlcds['pos']:
				if self.tlcds[i].direction == 'xy':
					self.total_tlcd = 2 * self.tlcds[i].amount
				else:
					self.total_tlcd += self.tlcds[i].amount

		if configurations.method == 'Finite Differences Method':
			if configurations.nonLinearAnalysis and (self.tlcds is not None):
				self.fdm_solver(nonlinear=True)
			else:
				self.fdm_solver(nonlinear=False)
		elif configurations.method == 'Average Acceleration Method':
			if configurations.nonLinearAnalysis and (self.tlcds is not None):
				self.newmark_solver(gamma=1/2, beta=1/4, nonlinear=True)
			else:
				self.newmark_solver(gamma=1/2, beta=1/4, nonlinear=False)
		elif configurations.method == 'Linear Acceleration Method':
			if configurations.nonLinearAnalysis and (self.tlcds is not None):
				self.newmark_solver(gamma=1/2, beta=1/6, nonlinear=True)
			else:
				self.newmark_solver(gamma=1/2, beta=1/6, nonlinear=False)
		elif configurations.method == 'Runge-Kutta Method':
			if configurations.nonLinearAnalysis and (self.tlcds is not None):
				self.rk4_solver(nonlinear=True)
			else:
				self.rk4_solver(nonlinear=False)

	def unpack(self):
		self.M = self.mass
		self.C = self.damping
		self.K = self.stiffness
		self.F = self.force
		self.dt = self.configurations.timeStep
		self.x0 = self.configurations.initialDisplacement
		self.v0 = self.configurations.initialVelocity

		self.x = 0. * self.F
		self.v = 0. * self.F
		self.a = 0. * self.F
		self.t = [i * self.dt for i in range(self.F.shape[1])]

		# TODO make initialDisplacement and initialVelocity vectors that represent both parameters at each DOF
		self.x[:, 0] = self.x0
		self.v[:, 0] = self.v0

		self.a0 = self.M.I * (self.F[:, 0] - self.C * self.v[:, 0] - self.K * self.x[:, 0])
		self.a[:, 0] = self.a0

	def fdm_solver(self, nonlinear=False):
		self.unpack()

		if nonlinear:
			self.count = self.C.shape[0] - self.total_tlcd
			for k in self.tlcds['pos']:
				self.damping_update_fdm(self.tlcds[k], 0)

		self.alpha = (self.M / (self.dt ** 2) - self.C / (2 * self.dt))
		self.beta = (self.K - 2 * self.M / (self.dt ** 2))
		self.gamma = (self.M / (self.dt ** 2) + self.C / (2 * self.dt))

		self.xm1 = self.x[:, 0] - self.v[:, 0] * self.dt + (self.a[:, 0] * self.dt ** 2) / 2
		self.x[:, 1] = self.gamma.I * (self.F[:, 0] - self.beta * self.x[:, 0] - self.alpha * self.xm1)

		for i in list(range(1, len(self.t[1:]))):
			if nonlinear:
				if i >= 2:
					self.count = self.C.shape[0] - self.total_tlcd
					for k in self.tlcds['pos']:
						self.damping_update_fdm(self.tlcds[k], i)

					self.alpha = (self.M / (self.dt ** 2) - self.C / (2 * self.dt))
					self.beta = (self.K - 2 * self.M / (self.dt ** 2))
					self.gamma = (self.M / (self.dt ** 2) + self.C / (2 * self.dt))

			self.x[:, i + 1] = self.gamma.I * (self.F[:, i] - self.beta * self.x[:, i] - self.alpha * self.x[:, i - 1])

		i = len(self.t[1:])
		self.xM1 = self.gamma.I * (self.F[:, i] - self.beta * self.x[:, i] - self.alpha * self.x[:, i - 1])
		self.xMais1 = np.concatenate((self.x[:, 1:], self.xM1), axis=1)
		self.xMenos1 = np.concatenate((self.xm1, self.x[:, 0:-1]), axis=1)

		self.v = (self.xMais1 - self.xMenos1) / (2 * self.dt)
		self.a = (self.xMais1 - 2 * self.x + self.xMenos1) / (self.dt ** 2)

	def damping_update_fdm(self, tlcd, i):
		if tlcd.direction == 'xy':
			correctionStop = self.count + tlcd.amount * 2
		else:
			correctionStop = self.count + tlcd.amount

		if i >= 1:
			self.dampingVelocityArray[0, i + 1] = (self.x[-1, i-2] - self.x[-1, i]) / (2 * self.dt)
			velocity = abs(self.dampingVelocityArray[0, i + 1])
		else:
			self.dampingVelocityArray = copy(self.v[-1, :])
			velocity = self.dampingVelocityArray[0, 0]

		correctionFactor = tlcd.calculate_damping_correction_factor(velocity)
		contractionDampingCoefficient = tlcd.calculate_contraction_damping(velocity)
		
		while self.count < correctionStop:
			self.C[self.count, self.count] = tlcd.dampingCoefficientConstant * correctionFactor
			self.C[self.count, self.count] += contractionDampingCoefficient
			self.count += 1

	
	def newmark_solver(self, gamma=1/2, beta=1/4, nonlinear=False):
		self.unpack()
		
		for i in list(range(0, len(self.t[1:]) - 1)):
			if nonlinear:
				self.damping_update_nm(i)

			k_eff = self.K + gamma/(beta*self.dt) * self.C + 1/(beta*self.dt**2) * self.M
			a = 1/(beta*self.dt) * self.M + gamma/beta * self.C
			b = 1/(2*beta) * self.M + self.dt * ((gamma/(2*beta)) - 1) * self.C

			dp_eff = (self.F[:, i+1] - self.F[:, i]) + (a * self.v[:, i]) + (b * self.a[:, i])
			dx = k_eff.I * dp_eff
			dv = gamma/(beta * self.dt)*dx - gamma/beta*self.v[:, i] + self.dt * (1 - (gamma/(2*beta))) * self.a[:, i]
			da = 1/(beta*self.dt**2)*dx - 1/(beta*self.dt)*self.v[:, i] - 1/(2*beta)*self.a[:, i]
			
			self.x[:, i+1] = self.x[:, i] + dx
			self.v[:, i+1] = self.v[:, i] + dv
			self.a[:, i+1] = self.a[:, i] + da

	
	def damping_update_nm(self, i):
		correctionStart = self.C.shape[1] - 1
		correctionStop = correctionStart - self.tlcd.amount

		if i >= 1:
			velocity = abs(self.v[-1, i])
		else:
			velocity = abs(self.v[-1, 0])

		correctionFactor = self.tlcd.calculate_damping_correction_factor(velocity)
		contractionDampingCoefficient = self.tlcd.calculate_contraction_damping(velocity)

		for j in range(correctionStart, correctionStop, -1):
			self.C[j, j] = self.tlcd.dampingCoefficientConstant * correctionFactor
			self.C[j, j] += contractionDampingCoefficient
			# print(velocity)
			# print(self.tlcd.dampingCoefficientConstant, correctionFactor, contractionDampingCoefficient)
			# print(self.C[j, j])

	def rk4_solver(self, nonlinear=False):
		self.unpack()

		for i in list(range(0, len(self.t[1:]) - 1)):
			if nonlinear:
				self.damping_update_nm(i)

			# First point
			t = self.t[i]
			x = self.x[:, i]
			y1 = self.v[:, i]
			y1_ = self.M.I * (self.F[:, i] - self.C * y1 - self.K * x)

			# Second point
			t = self.t[i] + self.dt/2
			x = self.x[:, i] + self.dt/2 * y1
			y2 = self.v[:, i] + self.dt/2 * y1_
			y2_ = self.M.I * (self.F[:, i] - self.C * y2 - self.K * x)

			# Third point
			t = self.t[i] + self.dt/2
			x = self.x[:, i] + self.dt/2 * y2
			y3 = self.v[:, i] + self.dt/2 * y2_
			y3_ = self.M.I * (self.F[:, i] - self.C * y3 - self.K * x)

			# Fourth point
			t = self.t[i] + self.dt
			x = self.x[:, i] + self.dt * y3
			y4 = self.v[:, i] + self.dt * y3_
			y4_ = self.M.I * (self.F[:, i] - self.C * y4 - self.K * x)

			# Update
			self.x[:, i+1] = self.x[:, i] + self.dt/6 * (y1 + 2*y2 + 2*y3 + y4)
			self.v[:, i+1] = self.v[:, i] + self.dt/6 * (y1_ + 2*y2_ + 2*y3_ + y4_)

def assemble_mass_matrix(diaphragms=None, model='tridimensional', tower=None):
	""" Function that takes a diaphragms object and the model either tridimensional or shear building.

	:param diaphragms: object - Diaphragms object containing data of all diaphragms and TLCD.
	:param model: str - A string representing the model used in calculations.
	:return: np.matrix - Mass matrix of the building equipped with tlcd.
	"""
	if model == 'tridimensional':
		if diaphragms.tlcds is not None:
			n = diaphragms.num_pav*3 + diaphragms.tlcds_amount
			
			lastStory = diaphragms.num_pav - 1
			
			M = np.mat(np.zeros((n, n)))
			count = 0
			for i in range((lastStory+1)*2):
				if (i)%diaphragms.num_pav == 0:
					count = 0
				M[i, i] = diaphragms.total_mass[count]
				count += 1

			for i in range((lastStory+1)*2, (lastStory+1)*3):
				M[i, i] = diaphragms.I
				
			A = lastStory + 1
			count1 = 0
			for j in diaphragms.tlcds['pos']:
				if type(j) == str:
					l = int(j[0])
				else:
					l = j
				if 'x' in diaphragms.tlcds[j].direction:
					for k in range(diaphragms.tlcds[j].amount):
						i = count1 + (lastStory + 1) * 3
						M[i, i] = diaphragms.tlcds[j].mass
						M[i, l-1] = (diaphragms.tlcds[j].width / diaphragms.tlcds[j].length) * diaphragms.tlcds[j].mass
						M[l-1, i] = (diaphragms.tlcds[j].width / diaphragms.tlcds[j].length) * diaphragms.tlcds[j].mass
						count1 += 1
			
			count2 = 0
			for j in diaphragms.tlcds['pos']:
				if type(j) == str:
					l = int(j[0])
				else:
					l = j
				if 'y' in diaphragms.tlcds[j].direction:
					for k in range(diaphragms.tlcds[j].amount):
						i = count2 + (lastStory + 1) * 3 + diaphragms.tlcds_amount_x
						M[i, i] = diaphragms.tlcds[j].mass
						M[i, l+lastStory] = (diaphragms.tlcds[j].width / diaphragms.tlcds[j].length) * diaphragms.tlcds[j].mass
						M[l+lastStory, i] = (diaphragms.tlcds[j].width / diaphragms.tlcds[j].length) * diaphragms.tlcds[j].mass
						count2 += 1

		else:
			n = diaphragms.num_pav*3
			lastStory = diaphragms.num_pav - 1

			M = np.mat(np.zeros((n, n)))
			count = 0
			for i in range((lastStory+1)*2):
				if (i)%diaphragms.num_pav == 0:
					count = 0
				M[i, i] = diaphragms.total_mass[count]
				count += 1

			for i in range((lastStory+1)*2, (lastStory+1)*3):
				M[i, i] = diaphragms.I
				
	elif model == 'shear building':
		if diaphragms.tlcds is not None:
			n = diaphragms.num_pav + diaphragms.tlcds_amount
			
			lastStory = diaphragms.num_pav - 1

			M = np.mat(np.zeros((int(n), int(n))))
			for i in range(lastStory+1):
				M[i, i] = diaphragms.total_mass[i]
				
			count1 = 0
			for j in diaphragms.tlcds['pos']:
				for k in range(diaphragms.tlcds[j].amount):
					i = count1 + lastStory + 1
					M[i, i] = diaphragms.tlcds[j].mass
					M[i, j-1] = (diaphragms.tlcds[j].width / diaphragms.tlcds[j].length) * diaphragms.tlcds[j].mass
					M[j-1, i] = (diaphragms.tlcds[j].width / diaphragms.tlcds[j].length) * diaphragms.tlcds[j].mass
					count1 += 1

		else:
			n = diaphragms.num_pav
			lastStory = diaphragms.num_pav - 1

			M = np.mat(np.zeros((n, n)))
			for i in range(lastStory+1):
				M[i, i] = diaphragms.total_mass[i]
	
	else:
		return ERROR_MSG
	
	""" 
	Tower structure is incomplete
	elif tower is not None:
		if tower.tlcds is not None:
			n = tower.num_pav + tower.tlcds_amount
			
			lastStory = tower.num_pav - 1
			
			M = np.mat(np.zeros((n, n)))
			for i in range(lastStory+1):
				M[i, i] = tower.total_mass[i]
			count = 0
			for j in tower.tlcds['pos']:
				for k in range(tower.tlcds[j].amount):
					i = count + lastStory + 1
					M[i, i] = tower.tlcds[j].mass
					M[i, j-1] = (tower.tlcds[j].width / tower.tlcds[j].length) * tower.tlcds[j].mass
					M[j-1, i] = (tower.tlcds[j].width / tower.tlcds[j].length) * tower.tlcds[j].mass
					count += 1
		else:
			n = tower.num_pav
			lastStory = tower.num_pav - 1

			M = np.mat(np.zeros((n, n)))
			for i in range(lastStory+1):
				M[i, i] = tower.total_mass[i]
	"""
	
	return M

def assemble_lat_stiffness_matrix(frame=None, diaphragms=None):
	""" Function that takes a frame object and a diaphragms object to return the frame lateral stiffness matrix.

	:param frame: object - Frame object containing data of the frame.
	:param diaphragms: object - Diaphragms object containing data of all diaphragms and TLCD.
	:return: np.matrix - Lateral stiffness matrix of the frame equiped with tlcd.
	"""         

	lastStory = len(frame.height) - 1
	n = len(frame.height)

	K1 = np.mat(np.zeros((n, n)))

	for i in range(lastStory + 1):
		K1[i, i] = frame.stiffness[i]

	for i in range(lastStory + 1, 1, -1):
		K1[i - 1, i - 2] = -frame.stiffness[i-1]
		K1[i - 2, i - 1] = -frame.stiffness[i-1]
		K1[i - 2, i - 2] += frame.stiffness[i-1]

	if diaphragms is not None:
		if diaphragms.tlcds is not None:
			m = diaphragms.num_pav + diaphragms.tlcds_amount
			lastStory = diaphragms.num_pav - 1
			K = np.mat(np.zeros((m, m)))

			K[:lastStory+1, :lastStory+1] = K1

			count1 = 0
			for j in diaphragms.tlcds['pos']:
				for i in range(diaphragms.tlcds[j].amount):
					i = count1 + lastStory + 1
					if diaphragms.tlcds[j].direction == frame.direction:
						K[i, i] = diaphragms.tlcds[j].stiffness
					count1 += 1
			return K
	return K1

def assemble_total_stiffness_matrix(frames, diaphragms):
	"""Function that takes a dictionary of frames and a diaphragms object and returns the total stiffness matrix of the building.

	:param frames: dict - Dictionary of frame objects containing data of all the frames.
	:param diaphragms: object - Diaphragms object containing data of all diaphragms and TLCD.
	:return: np.matrix - Total stiffness matrix of the building
	""" 
	K_i = []
	for i in range(len(frames)):
		K_i.append(assemble_lat_stiffness_matrix(frame=frames[i+1]))
	n = len(frames)
	a_i = []
	for i in range(n):
		Ident = np.identity(len(frames[i + 1].height))
		zero = np.zeros((frames[i + 1].num_pav, frames[i + 1].num_pav))
		if frames[i+1].direction == 'x':
			dist = frames[i + 1].dist
			a = np.concatenate((Ident, zero, -dist*Ident), axis=1)
			
		elif frames[i+1].direction == 'y':
			dist = frames[i + 1].dist
			a = np.concatenate((zero, Ident, dist*Ident), axis=1)
		
		a_i.append(np.mat(a))
		
	K1 = 0
	ki = []
	for i in range(n):
		a_T = np.transpose(a_i[i])
		ki.append(a_T*K_i[i]*a_i[i])
		K1 += ki[i]
		
	if diaphragms.tlcds is not None:
		m = diaphragms.num_pav*3 + diaphragms.tlcds_amount
		lastStory = diaphragms.num_pav-1
		K = np.mat(np.zeros((m, m)))
		
		K[:(lastStory+1)*3, :(lastStory+1)*3] = K1
		
		count1 = 0
		for j in diaphragms.tlcds['pos']:
			if 'x' in diaphragms.tlcds[j].direction:
				for i in range(diaphragms.tlcds[j].amount):
					i = count1 + (lastStory + 1) * 3
					K[i, i] = diaphragms.tlcds[j].stiffness
					count1 += 1
						
		count2 = 0
		for j in diaphragms.tlcds['pos']:
			if 'y' in diaphragms.tlcds[j].direction:
				for i in range(diaphragms.tlcds[j].amount):
					i = count2 + (lastStory + 1) * 3 + diaphragms.tlcds_amount_x
					K[i, i] = diaphragms.tlcds[j].stiffness
					count2 += 1
					
	else:
		K = K1
					
	return K

def calc_damping_coeff(diaphragms, frame, damping_ratio=0.02):
	""" Function that takes the diaphragms and frame objects and a damping ratio and returns the damping coefficient.
	
	:param diaphragms: object - Diaphragms object containing data of all diaphragms and TLCD.
	:param frame: object - Frame object containing data of the frame.
	:param dampingRatio: float - Damping ratio of the structure.
	:return: float - Damping coefficient
	"""
	n = diaphragms.num_pav
	crit_damp = 2 * diaphragms.total_mass * np.sqrt(frame.stiffness/diaphragms.total_mass)
	damp_coeff = crit_damp * damping_ratio
	
	return damp_coeff

def calc_natural_frequencies(M, K):
	""" Function that takes the mass and stiffness matrix of the structure and returns the natural frequencies and modal matrix.

	:param M: np.matrix - Mass matrix of the structure.
	:param K: np.matrix - Stiffness matrix of the structure.
	:return: np.matrix, np.matrix - Natural frequencies and modal matrix of the structure.
	"""
	w_squared, Phi = linalg.eig(K, M)
	w = np.sqrt(np.real(w_squared))
	
	return w, Phi

def assemble_damping_matrix(diaphragms=None, frame=None, tower=None, w=None, M=None, K=None,
							dampingRatio=0.02, model='tridimensional'):
	""" Function that takes the model to be considered and, depending on the model calculates the damping matrix.

	:param frame: object - Frame object containing data of the frame.
	:param diaphragms: object - Diaphragms object containing data of all diaphragms and TLCD.
	:param w: np.matrix - Natural frequencies vector.
	:param M: np.matrix - Mass matrix of the structure.
	:param K: np.matrix - Stiffness matrix of the structure.
	:param dampingRatio: float - Damping ratio of the structure.
	:param model: str - A string representing the model used in calculations.
	:return: np.matrix, np.matrix - Damping matrix of the building equiped with tlcd and damping ratio of each degree of freedom.
	"""
	
	if model == 'tridimensional':
		#Rayleigh Method
		lastStory = diaphragms.num_pav - 1
		C = np.mat(np.zeros(M.shape))
		n = M.shape[0]
		w_sorted = np.sort(w[:int(n)])
		w_i = w[0]
		w_j = w[int(2*n/3) - 1]
		a0 = dampingRatio * (2 * w_i * w_j)/(w_i + w_j)
		a1 = dampingRatio * 2/(w_i + w_j)
		csi = a0/2 * 1/w + a1/2 * w
		
		C[:(lastStory+1)*3, :(lastStory+1)*3] = a0 * M[:(lastStory+1)*3, :(lastStory+1)*3] + a1 * K[:(lastStory+1)*3, :(lastStory+1)*3]
		if diaphragms.tlcds is not None:
			count1 = 0
			for j in diaphragms.tlcds['pos']:
				if 'x' in diaphragms.tlcds[j].direction:
					for k in range(diaphragms.tlcds[j].amount):
						#print(j)
						i = count1 + (lastStory + 1) * 3
						C[i, i] = diaphragms.tlcds[j].dampingCoefficient
						count1 += 1

			count2 = 0
			for j in diaphragms.tlcds['pos']:
				if 'y' in diaphragms.tlcds[j].direction:
					for k in range(diaphragms.tlcds[j].amount):
						#print(j)
						i = count2 + (lastStory + 1) * 3 + diaphragms.tlcds_amount_x
						C[i, i] = diaphragms.tlcds[j].dampingCoefficient
						count2 += 1
						
		return C, csi
	
	if model == 'shear building':
		damp_coeff = calc_damping_coeff(diaphragms, frame, damping_ratio=0.02)
		n = len(damp_coeff)
		C1 = np.matrix(np.zeros((n,n)))
		for i in range(n):
			C1[i,i] = damp_coeff[i]
			
		if diaphragms.tlcds is not None:
			m = diaphragms.num_pav + diaphragms.tlcds_amount
			lastStory = diaphragms.num_pav - 1
			C = np.mat(np.zeros((m, m)))

			C[:lastStory+1, :lastStory+1] = C1

			count1 = 0
			for j in diaphragms.tlcds['pos']:
				for i in range(diaphragms.tlcds[j].amount):
					i = count1 + lastStory + 1
					C[i, i] = diaphragms.tlcds[j].dampingCoefficient
					count1 += 1
		else:
			C = C1
		
		n = M.shape[0]
		w_sorted = np.sort(w[:int(n)])
		w_i = w_sorted[0]
		w_j = w_sorted[-1]
		a0 = dampingRatio * (2 * w_i * w_j)/(w_i + w_j)
		a1 = dampingRatio * 2/(w_i + w_j)
		csi = a0/2 * 1/w + a1/2 * w
		
		return C, csi
	
	"""
	Tower part is incomplete
	if tower is not None:
		n = len(w) - tower.tlcds_amount
		crit_damp = 2 * tower.total_mass * np.sqrt(tower.lat_stiffness/tower.total_mass)
		damp_coeff = crit_damp * tower.dampingRatio
		n = len(damp_coeff)
		C1 = np.matrix(np.zeros((n,n)))
		for i in range(n):
			C1[i,i] = damp_coeff[i]  
			
		if tower.tlcds is not None:
			m = tower.num_pav + tower.tlcds_amount
			lastStory = tower.num_pav - 1
			C = np.mat(np.zeros((m, m)))

			C[:lastStory+1, :lastStory+1] = C1

			count1 = 0
			for j in tower.tlcds['pos']:
				for i in range(tower.tlcds[j].amount):
					i = count1 + lastStory + 1
					C[i, i] = tower.tlcds[j].dampingCoefficient
					count1 += 1
		else:
			C = C1
		
		n = M.shape[0]
		w_sorted = np.sort(w[:int(n)])
		w_i = w_sorted[0]
		w_j = w_sorted[-1]
		a0 = tower.dampingRatio * (2 * w_i * w_j)/(w_i + w_j)
		a1 = tower.dampingRatio * 2/(w_i + w_j)
		csi = a0/2 * 1/w + a1/2 * w
		
		return C, csi
	"""

def assemble_force_matrix(mass=None, excitation=None, configurations=None, diaphragms=None,
						  excitation_x=None, excitation_y=None, model='tridimensional'):
	""" Function that calculates the force matrix, given the model and the excitation type.

	:param excitation: object - Object containing type of excitation and its parameters (measured by acceleration) for a shear building model.
	:param mass: np.matrix - Mass matrix of any system.
	:param configurations: object - Object containing time step of iterations.
	:param diaphragms: object - Diaphragms object containing data of all diaphragms and TLCD.
	:param excitation_x: object - Object containing type of excitation in the x direction and its parameters (measured by acceleration) for a tridimensional model.
	:param excitation_y: object - Object containing type of excitation in the y direction and its parameters (measured by acceleration) for a tridimensional model.
	:param model: str - A string representing the model used in calculations.
	:return: np.matrix, np.matrix - Force vector evaluated over time and time vector.
	"""
		
	"""
	Tower part is incomplete
	if tower is not None:
		step = configurations.timeStep
		totalTimeArray = np.mat(np.arange(0, excitation.anlyDuration + step, step))
		excitationTimeArray = np.mat(np.arange(0, excitation.exctDuration + step, step))
		force = 0. * totalTimeArray
		
		if tower.tlcds is None:
			numberOfStories = mass.shape[0]
		else:
			numberOfStories = mass.shape[0] - tower.tlcds_amount

		for i in range(mass.shape[0] - 1):
			force = np.concatenate((force, 0. * totalTimeArray), 0)

		if excitation.type == 'Sine Wave':
			for i in range(numberOfStories):
				storyMass = mass[i, i]
				forceAmplitude = storyMass * excitation.amplitude
				for j in range(excitationTimeArray.shape[1]):
					force[i, j] = forceAmplitude * np.sin(excitation.frequency * totalTimeArray[0, j])

			if tower.tlcds is None:
				return force, totalTimeArray
			else:
				print(numberOfStories)
				count = 0
				for j in tower.tlcds['pos']:
					for n in range(tower.tlcds[j].amount):
						print(count)
						i = count + numberOfStories
						alpha = tower.tlcds[j].width/tower.tlcds[j].length
						tlcdMass = tower.tlcds[j].mass
						forceAmplitude = alpha * tlcdMass * excitation.amplitude
						force[i, :] = forceAmplitude * np.sin(excitation.frequency * totalTimeArray[0, :])
						#for k in range(excitationTimeArray.shape[1]):
						#    force[i, k] = forceAmplitude * np.sin(excitation.frequency * totalTimeArray[0, k])
						count += 1

				return force, totalTimeArray

		elif excitation.type == 'General Excitation':
			a = []
			t0 = 0
			time = [round(t / step, 0) * step for t in list(totalTimeArray.A1)]
			for t in time:
				if t in excitation.t_input:
					t0 = t
					t0_index = excitation.t_input.index(t)
					a.append(excitation.a_input[t0_index])
				else:
					t1 = excitation.t_input[t0_index + 1]
					a0 = excitation.a_input[t0_index]
					a1 = excitation.a_input[t0_index + 1]
					at = ((a1 - a0) / (t1 - t0)) * (t - t0) + a0
					a.append(at)

			# print(list(zip(list(totalTimeArray.A1), a)))
			a = np.array(a)
			for i in range(force.shape[0]):
				storyMass = mass[i, i]
				force[i, :] = storyMass * a

			if tlcd is None:
				return force, totalTimeArray
			else:
				for i in range(tlcd.amount):
					force = np.concatenate((force, 0. * force[0, :]), 0)
				return force, totalTimeArray
	"""
			
	if diaphragms is not None and model == 'tridimensional':
		lastStory = diaphragms.num_pav
		step = configurations.timeStep
		n = diaphragms.num_pav*3
		
		if excitation_x is not None and excitation_y is not None:
			anlyDuration = np.maximum(excitation_x.anlyDuration, excitation_y.anlyDuration)
			exctDuration = np.maximum(excitation_x.exctDuration, excitation_y.exctDuration)
			
		elif excitation_x is not None:
			anlyDuration = excitation_x.anlyDuration
			exctDuration = excitation_x.exctDuration
			
		elif excitation_y is not None:
			anlyDuration = excitation_y.anlyDuration
			exctDuration = excitation_y.exctDuration
			
		totalTimeArray = np.mat(np.arange(0, anlyDuration + step, step))
		excitationTimeArray = np.mat(np.arange(0, exctDuration + step, step))
		
		if diaphragms.tlcds is not None:
			force = np.matrix(np.zeros((n + diaphragms.tlcds_amount, totalTimeArray.shape[1])))
		else:
			force = np.matrix(np.zeros((n, totalTimeArray.shape[1])))
		
		if excitation_x is not None:
			if excitation_x.type == 'Sine Wave':
				totalTimeArray_x = np.mat(np.arange(0, excitation_x.anlyDuration + step, step))
				excitationTimeArray_x = np.mat(np.arange(0, excitation_x.exctDuration + step, step))
				for i in range(lastStory):
					storyMass = mass[i, i]
					forceAmplitude = storyMass * excitation_x.amplitude
					for k in range(excitationTimeArray_x.shape[1]):
						force[i, k] = forceAmplitude * np.sin(excitation_x.frequency * totalTimeArray_x[0, k])
				
				if diaphragms.tlcds is not None:
					count1 = 0
					for j in diaphragms.tlcds['pos']:
						if 'x' in diaphragms.tlcds[j].direction:
							for h in range(diaphragms.tlcds[j].amount):
								i = count1 + (lastStory) * 3
								alpha = diaphragms.tlcds[j].width/diaphragms.tlcds[j].length
								tlcdMass = diaphragms.tlcds[j].mass
								forceAmplitude = alpha * tlcdMass * excitation_x.amplitude
								for k in range(excitationTimeArray_x.shape[1]):
									force[i, k] = forceAmplitude * np.sin(excitation_x.frequency * totalTimeArray[0, k])
								count1 += 1

			elif excitation_x.type == 'General Excitation':
				a = []
				t0 = 0
				time = [round(t / step, 0) * step for t in list(totalTimeArray.A1)]
				for t in time:
					if t in excitation.t_input:
						t0 = t
						t0_index = excitation.t_input.index(t)
						a.append(excitation.a_input[t0_index])
					else:
						t1 = excitation.t_input[t0_index + 1]
						a0 = excitation.a_input[t0_index]
						a1 = excitation.a_input[t0_index + 1]
						at = ((a1 - a0) / (t1 - t0)) * (t - t0) + a0
						a.append(at)

				# print(list(zip(list(totalTimeArray.A1), a)))
				a = np.array(a)
				for i in range(lastStory):
					storyMass = mass[i, i]
					force[i, :] = storyMass * a
				
				if diaphragms.tlcds is not None:
					count1 = 0
					for j in diaphragms.tlcds['pos']:
						if 'x' in diaphragms.tlcds[j].direction:
							for h in range(diaphragms.tlcds[j].amount):
								i = count1 + (lastStory) * 3
								alpha = diaphragms.tlcds[j].width/diaphragms.tlcds[j].length
								tlcdMass = diaphragms.tlcds[j].mass
								force[i, :] = alpha * tlcdMass * a
								count1 += 1

		if excitation_y is not None:
			if excitation_y.type == 'Sine Wave':
				totalTimeArray_y = np.mat(np.arange(0, excitation_y.anlyDuration + step, step))
				excitationTimeArray_y = np.mat(np.arange(0, excitation_y.exctDuration + step, step))
				for i in range(int(n/3), int(2*n/3)):
					storyMass = diaphragms.total_mass[int(i/2)]
					forceAmplitude = storyMass * excitation_y.amplitude
					for k in range(excitationTimeArray_y.shape[1]):
						force[i, k] = forceAmplitude * np.sin(excitation_y.frequency * totalTimeArray_y[0, k])

				if diaphragms.tlcds is not None:
					count2 = 0
					for j in diaphragms.tlcds['pos']:
						if 'y' in diaphragms.tlcds[j].direction:
							for h in range(diaphragms.tlcds[j].amount):
								i = count2 + lastStory * 3 + diaphragms.tlcds_amount_x
								alpha = diaphragms.tlcds[j].width/diaphragms.tlcds[j].length
								tlcdMass = diaphragms.tlcds[j].mass
								forceAmplitude = alpha * tlcdMass * excitation_y.amplitude
								for k in range(excitationTimeArray.shape[1]):
									force[i, k] = forceAmplitude * np.sin(excitation_y.frequency * totalTimeArray[0, k])
								count2 += 1

			elif excitation_y.type == 'General Excitation':
				a = []
				t0 = 0
				time = [round(t / step, 0) * step for t in list(totalTimeArray.A1)]
				for t in time:
					if t in excitation.t_input:
						t0 = t
						t0_index = excitation.t_input.index(t)
						a.append(excitation.a_input[t0_index])
					else:
						t1 = excitation.t_input[t0_index + 1]
						a0 = excitation.a_input[t0_index]
						a1 = excitation.a_input[t0_index + 1]
						at = ((a1 - a0) / (t1 - t0)) * (t - t0) + a0
						a.append(at)

				# print(list(zip(list(totalTimeArray.A1), a)))
				a = np.array(a)
				for i in range(lastStory):
					storyMass = mass[i, i]
					force[i, :] = storyMass * a
				
				if diaphragms.tlcds is not None:
					count1 = 0
					for j in diaphragms.tlcds['pos']:
						if 'y' in diaphragms.tlcds[j].direction:
							for h in range(diaphragms.tlcds[j].amount):
								i = count1 + (lastStory) * 3
								alpha = diaphragms.tlcds[j].width/diaphragms.tlcds[j].length
								tlcdMass = diaphragms.tlcds[j].mass
								force[i, :] = alpha * tlcdMass * a
								count1 += 1
		
		return force, totalTimeArray
	
	if diaphragms is not None and model == 'shear building':
		step = configurations.timeStep
		totalTimeArray = np.mat(np.arange(0, excitation.anlyDuration + step, step))
		excitationTimeArray = np.mat(np.arange(0, excitation.exctDuration + step, step))
		force = 0. * totalTimeArray
		
		if diaphragms.tlcds is None:
			numberOfStories = mass.shape[0]
		else:
			numberOfStories = mass.shape[0] - diaphragms.tlcds_amount

		for i in range(mass.shape[0] - 1):
			force = np.concatenate((force, 0. * totalTimeArray), 0)

		if excitation.type == 'Sine Wave':
			for i in range(numberOfStories):
				storyMass = mass[i, i]
				forceAmplitude = storyMass * excitation.amplitude
				for j in range(excitationTimeArray.shape[1]):
					force[i, j] = forceAmplitude * np.sin(excitation.frequency * totalTimeArray[0, j])

			if diaphragms.tlcds is None:
				return force, totalTimeArray
			else:
				count = 0
				for j in diaphragms.tlcds['pos']:
					for n in range(diaphragms.tlcds[j].amount):
						i = count + numberOfStories
						alpha = diaphragms.tlcds[j].width/diaphragms.tlcds[j].length
						tlcdMass = diaphragms.tlcds[j].mass
						forceAmplitude = alpha * tlcdMass * excitation.amplitude
						force[i, :] = forceAmplitude * np.sin(excitation.frequency * totalTimeArray[0, :])
						#for k in range(excitationTimeArray.shape[1]):
						#    force[i, k] = forceAmplitude * np.sin(excitation.frequency * totalTimeArray[0, k])
						count += 1

		elif excitation.type == 'General Excitation':
			a = []
			t0 = 0
			time = [round(t / step, 0) * step for t in list(totalTimeArray.A1)]
			for t in time:
				if t in excitation.t_input:
					t0 = t
					t0_index = excitation.t_input.index(t)
					a.append(excitation.a_input[t0_index])
				else:
					t1 = excitation.t_input[t0_index + 1]
					a0 = excitation.a_input[t0_index]
					a1 = excitation.a_input[t0_index + 1]
					at = ((a1 - a0) / (t1 - t0)) * (t - t0) + a0
					a.append(at)

			# print(list(zip(list(totalTimeArray.A1), a)))
			a = np.array(a)
			for i in range(force.shape[0]):
				storyMass = mass[i, i]
				force[i, :] = storyMass * a

			if diaphragms.tlcds is not None:
				count1 = 0
				for j in diaphragms.tlcds['pos']:
					for h in range(diaphragms.tlcds[j].amount):
						i = count1 + numberOfStories
						alpha = diaphragms.tlcds[j].width/diaphragms.tlcds[j].length
						tlcdMass = diaphragms.tlcds[j].mass
						force[i, :] = alpha * tlcdMass * a
						count1 += 1
		return force, totalTimeArray

def solve_sdof_system(m, ksi, k, p0, omega, t_lim, x0=0, v0=0):
	""" Function that calculates analytically the displacement of a sdof system for a sine wave excitation.

	:param m: float - Mass of the degree of freedom.
	:param ksi: float - Damping ratio of the degree of freedom.
	:param k: float - Stiffness of the degree of freedom.
	:param p0: float - Force applyed to the degree of freedom.
	:param omega: float - Natural frequency of the degree of freedom.
	:param t_lim: int - Maximum time analysed.
	:param x0: float - Inicial displacement.
	:param v0: float - Inicial velocity.
	:return: np.matrix - Displacement over time of the degree of freedom.
	"""
	omega_n = np.sqrt(k / m)
	omega_d = omega_n * np.sqrt(1 - ksi ** 2)

	C = (p0 / k) * (
		(1 - (omega / omega_n) ** 2) / ((1 - (omega / omega_n) ** 2) ** 2 + (2 * ksi * (omega / omega_n)) ** 2))
	D = (p0 / k) * (
		(-2 * ksi * (omega / omega_n)) / ((1 - (omega / omega_n) ** 2) ** 2 + (2 * ksi * (omega / omega_n)) ** 2))

	A = x0 - D  # x(0) = 0
	B = (v0 + ksi * omega_n * A - omega * C) / omega_d  # x'(0) = 0

	t = np.linspace(0, t_lim, 10000)
	x = np.exp(-ksi * omega_n * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t)) + C * np.sin(
		omega * t) + D * np.cos(omega * t)

	return x

def modal_sup(M, K, F, Ksi, excitation, Phi, t_lim, influence=None):
	""" Function that solves the movement equation through modal superposition.

	:param M: np.matrix - Mass matrix of the system.
	:param K: np.matrix - Stiffness matrix of the system.
	:param F: np.matrix - Force matrix of the system.
	:param Ksi: np.matrix - Damping ratio vector of the system.
	:param excitation: object - Excitation object that contains the information of the excitation applyed to the system.
	:param Phi: np.matrix - Modal matrix of the system.
	:param t_lim: int - Maximum time analysed.
	:param influence: np.array - Influence vector of the excitation force.
	:return: np.matrix - Modal displacements of the degrees of freedom.
	"""
	if influence is None:
		influence = np.matrix(np.ones((M.shape[0], 1)))
	M_n = Phi.T * M * Phi
	M_n[M_n < 0.001] = 0
	K_n = Phi.T * K * Phi
	K_n[K_n < 0.001] = 0
	
	P0 = Phi.T * excitation.amplitude * M * influence
	q = solve_sdof_system(M_n[0, 0], Ksi[0], K_n[0, 0], P0[0, 0], excitation.frequency, t_lim, x0=0, v0=0)
	for i in range(1, M_n.shape[0]):
		m = M_n[i, i]
		k = K_n[i, i]
		p0 = P0[i, 0]
		ksi = Ksi[i]
		q = np.vstack((q, solve_sdof_system(m, ksi, k, p0, excitation.frequency, t_lim, x0=0, v0=0)))
		
	return np.matrix(q)

def damper_tuner(M, configurations, w_n, massPercentage=0.02, amount=1, tlcdType='Pressurized TLCD',
				 P=None, D=None, Z=None, B=None, h=None, model='shear building'):
	""" Function that calculates the parameters of a damper, for a specific frequency.
	
	:param M: np.matrix - Mass matrix of the system.
	:param configurations: object - Object containing configurations for solving the equations.
	:param w_n: float - Frequency to be used to tune the damper.
	:param massPercentage: float - Ratio of the mass of dampers over mass of the structure.
	:param amount: int - Number of dampers in the system.
	:param tlcdType: str - Type of TLCD, either Pressurized or Basic.
	:param D: float - Diameter of the tube, for pressurized dampers.
	:param Z: float - Gas height, for pressurized dampers.
	:param B: float - Width of the tube.
	:param h: float - Water height of the tube.
	:param model: str - A string representing the model used in calculations.
	"""
	if tlcdType == 'Pressurized TLCD':
		if model == 'shear building':
			if B is None and h is None:
				L = (4 * (massPercentage / amount) * np.sum(M)) / (
					configurations.liquidSpecificMass * np.pi * D ** 2)
				B = L * 0.7
				h = 0.3 * L / 2
			else:
				L = B + 2 * h

			if P is None:
				const = (configurations.liquidSpecificMass * ((w_n**2 * L)/2 - configurations.gravity))
				P = const * Z / 1.4
		elif model == 'tridimensional':
			shape = M.shape[0]
			if B is None and h is None:
				L = (4 * (massPercentage / amount) * np.sum(M[:int(shape/3)])) / (
					configurations.liquidSpecificMass * np.pi * D ** 2)
				B = L * 0.7
				h = 0.3 * L / 2
			else:
				L = B + 2 * h

			if P is None:
				const = (configurations.liquidSpecificMass * ((w_n**2 * L)/2 - configurations.gravity))
				P = const * Z / 1.4
		
		return L, P, B, h
		
	if tlcdType == 'Basic TLCD':
		if model == 'shear building':
			L = 2 * configurations.gravity / (w_n**2)
			D = np.sqrt((4 * (massPercentage / amount) * np.sum(M)) / (
				configurations.liquidSpecificMass * np.pi * L))
			B = 0.7 * L
			h = 0.3 * L / 2
		elif model == 'tridimensional':
			shape = M.shape[0]
			L = 2 * configurations.gravity / (w_n**2)
			D = np.sqrt((4 * (massPercentage / amount) * np.sum(M[:int(shape/3)])) / (
				configurations.liquidSpecificMass * np.pi * L))
			B = 0.7 * L
			h = 0.3 * L / 2
		
		return L, D, B, h

def tuner(num_pav, frames, M0, configurations, w, massPercentage, tlcdType, pos, amount=1, D=None, Z=None, B0=None,
		  h0=None, contraction=1, frequency='same', model='shear building', direction='x', number=0,
		  limite=0.1):
	""" Function that returns the parameters of a TLCD damper so that the damper and the structure are tuned.
	
	:param num_pav: int - Number of stories of the structure.
	:param frames: dict - Dictionary of frame objects containing data of all the frames.
	:param M0: np.matrix - Initial mass matrix of the system.
	:param configurations: object - Object containing configurations for solving the equations.
	:param w: np.matrix - Natural frequencies vector.
	:param massPercentage: float - Ratio of the mass of dampers over mass of the structure.
	:param tlcdType: str - Type of TLCD, either Pressurized or Basic.
	:param pos: list - List of positions of the dampers.
	:param amount: int - Number of dampers in the system.
	:param D: float - Diameter of the tube, for pressurized dampers.
	:param Z: float - Gas height, for pressurized dampers.
	:param B0: float - Inicial width of the tube.
	:param h0: float - Initial water height of the tube.
	:param contraction: float - Contraction ratio in the tube.
	:param frequency: str - String for determining if the natural frequencies of the dampers is goint to be equal or different.
	:param model: str - A string representing the model used in calculations.
	:param direction: char - Direction in which to put the dampers.
	:param number: int - Which natural frequency used to tune the damper.
	:param limite: float - The acceptable difference between the frequencies of the structure and the damper.
	:return: dict - Dictionary with the parameters of the dampers.
	"""
	if frequency == 'different':
		w_sorted = np.sort(w)
		w_n = w_sorted[:len(pos)]
		print(w_n)
	elif frequency == 'same':
		w_n = np.repeat(min(w), len(pos))
	if B0 is not None and h0 is not None:
		B = B0
		h = h0
	w_t = 10 * w_n
	w_tn = 10 * w_n
	pos = np.sort(pos)
	counter = 0
	diff_ant = 100000
	if tlcdType == 'Pressurized TLCD':
		if model == 'shear building':
			while (abs(w_n - w_tn) >= 0.1).any():
				tlcds = {'pos' : pos}
				results = {}
				for i in range(w_n.shape[0]):
					if B0 is not None and h0 is not None:
						L, P, B, h = damper_tuner(M0, configurations, w_n[i], massPercentage=massPercentage,
												  amount=amount, tlcdType=tlcdType, D=D, Z=Z, B=B, h=h)
					else:
						L, P, B, h = damper_tuner(M0, configurations, w_n[i], massPercentage=massPercentage,
												  amount=amount, tlcdType=tlcdType, D=D, Z=Z)
					results['L' + str(i)] = L
					results['P' + str(i)] = P
					results['B' + str(i)] = B
					results['h' + str(i)] = h
					tlcds[pos[i]] = (TLCD(tlcdType=tlcdType, gasHeight=Z, gasPressure=P, diameter=D, width=B,
										  waterHeight=h, amount=int(amount/len(pos)), contraction=contraction,
										  pos=pos[i]))

				diaphragms = Diaphragms(num_pav=num_pav, tlcds=tlcds)
				M = assemble_mass_matrix(diaphragms=diaphragms, model='shear building')
				K = assemble_lat_stiffness_matrix(frame=frames, diaphragms=diaphragms)
				w, _ = calc_natural_frequencies(M, K)

				count = 0
				for i in pos:
					w_t[count] = tlcds[i].naturalFrequency
					count += 1
				w_t_sorted = np.sort(w_t)
				w_tn = w_t_sorted
				if frequency == 'different':
					w_sorted = np.sort(w[:num_pav])
					w_n = w_sorted[:len(pos)]
				elif frequency == 'same':
					w_n = np.repeat(min(w[:num_pav]), len(pos))

				diff = sum(abs(w_n - w_tn)) / len(w_n)
				if diff < diff_ant:
					results_real = copy(results)
					diff_ant = diff

				if counter == 1000:
					print('break')
					break
				counter += 1
		elif model == 'tridimensional':
			w_sorted = np.sort(w[:num_pav*3])
			w_n = np.repeat(w_sorted[number], len(pos))
			while (abs(w_n - w_tn) >= limite).any():
				tlcds = {'pos' : pos}
				results = {}
				for i in range(w_n.shape[0]):
					if B0 is not None and h0 is not None:
						L, P, B, h = damper_tuner(M0, configurations, w_n[i], massPercentage=massPercentage,
												  amount=amount, tlcdType=tlcdType, D=D, Z=Z, B=B, h=h, model=model)
					else:
						L, P, B, h = damper_tuner(M0, configurations, w_n[i], massPercentage=massPercentage,
												  amount=amount, tlcdType=tlcdType, D=D, Z=Z, model=model)
					results['L' + str(i)] = L
					results['P' + str(i)] = P
					results['B' + str(i)] = B
					results['h' + str(i)] = h
					tlcds[pos[i]] = (TLCD(tlcdType=tlcdType, gasHeight=Z, gasPressure=P, diameter=D, width=B,
								 waterHeight=h, amount=int(amount/len(pos)), contraction=contraction, pos=pos[i], direction=direction))
				
				diaphragms = Diaphragms(num_pav=num_pav, tlcds=tlcds)
				M = assemble_mass_matrix(diaphragms=diaphragms, model='tridimensional')
				K = assemble_total_stiffness_matrix(frames=frames, diaphragms=diaphragms)
				w, _ = calc_natural_frequencies(M, K)

				count = 0
				for i in pos:
					w_t[count] = tlcds[i].naturalFrequency
					count += 1
				w_t_sorted = np.sort(w_t)
				w_tn = w_t_sorted
				if frequency == 'different':
					w_sorted = np.sort(w[:num_pav*3])
					w_n = w_sorted[:len(pos)]
				elif frequency == 'same':
					w_sorted = np.sort(w[:num_pav*3])
					w_n = np.repeat(w_sorted[number], len(pos))

				diff = sum(abs(w_n - w_tn)) / len(w_n)
				if diff < diff_ant:
					results_real = copy(results)
					diff_ant = diff
				if counter == 1000:
					print('break')
					break
				counter += 1

		return results_real
			
	if tlcdType == 'Basic TLCD':
		if model == 'shear building':
			while (abs(w_n - w_tn) >= 0.1).any():
				tlcds = {'pos' : pos}
				results = {}
				for i in range(w_n.shape[0]):
					L, D, B, h = damper_tuner(M0, configurations, w_n[i], massPercentage=massPercentage,
											  amount=amount, tlcdType=tlcdType)
					results['L' + str(i)] = L
					results['D' + str(i)] = D
					results['B' + str(i)] = B
					results['h' + str(i)] = h
					tlcds[pos[i]] = (TLCD(tlcdType=tlcdType, diameter=D, width=B, waterHeight=h,
										  amount=int(amount/len(pos)), contraction=contraction, pos=pos[i]))

				diaphragms = Diaphragms(num_pav=num_pav, tlcds=tlcds)
				M = assemble_mass_matrix(diaphragms=diaphragms, model='shear building')
				K = assemble_lat_stiffness_matrix(frame=frames, diaphragms=diaphragms)
				w, _ = calc_natural_frequencies(M, K)

				count = 0
				for i in pos:
					w_t[count] = tlcds[i].naturalFrequency
					count += 1
				w_t_sorted = np.sort(w_t)
				w_tn = w_t_sorted
				if frequency == 'different':
					w_sorted = np.sort(w[:num_pav])
					w_n = w_sorted[:len(pos)]
				elif frequency == 'same':
					w_n = np.repeat(min(w[:num_pav]), len(pos))

				diff = sum(abs(w_n - w_tn)) / len(w_n)
				if diff < diff_ant:
					results_real = copy(results)
					diff_ant = diff

				if counter == 1000:
					print('break')
					break
				counter += 1
			
		elif model == 'tridimensional':
			w_sorted = np.sort(w[:num_pav*3])
			w_n = np.repeat(w_sorted[number], len(pos))
			while (abs(w_n - w_tn) >= 0.1).any():
				tlcds = {'pos' : pos}
				results = {}
				for i in range(w_n.shape[0]):
					L, D, B, h = damper_tuner(M0, configurations, w_n[i], massPercentage=massPercentage,
											  amount=amount, tlcdType=tlcdType, model=model)
					results['L' + str(i)] = L
					results['D' + str(i)] = D
					results['B' + str(i)] = B
					results['h' + str(i)] = h
					tlcds[pos[i]] = (TLCD(tlcdType=tlcdType, diameter=D, width=B, waterHeight=h, direction=direction,
										  amount=int(amount/len(pos)), contraction=contraction, pos=pos[i]))

				diaphragms = Diaphragms(num_pav=num_pav, tlcds=tlcds)
				M = assemble_mass_matrix(diaphragms=diaphragms, model='tridimensional')
				K = assemble_total_stiffness_matrix(frames=frames, diaphragms=diaphragms)
				w, _ = calc_natural_frequencies(M, K)

				count = 0
				for i in pos:
					w_t[count] = tlcds[i].naturalFrequency
					count += 1
				w_t_sorted = np.sort(w_t)
				w_tn = w_t_sorted
				if frequency == 'different':
					w_sorted = np.sort(w[:num_pav*3])
					w_n = w_sorted[:len(pos)]
				elif frequency == 'same':
					w_sorted = np.sort(w[:num_pav*3])
					w_n = np.repeat(w_sorted[number], len(pos))

				diff = sum(abs(w_n - w_tn)) / len(w_n)
				if diff < diff_ant:
					results_real = copy(results)
					diff_ant = diff

				if counter == 1000:
					print('break')
					break
				counter += 1
		
		return results_real