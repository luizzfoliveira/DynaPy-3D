import numpy as np
from copy import copy

class Frame(object):
	def __init__(self, num_pav, direction, dist, num_frame=1, height=3., b_width=5., mass=10.e3, width=.35, 
				 depth=.35, E=25.e9, support='Fix-Fix'):
		"""
		:param num_pav: int - Number of stories of the structure
		:param direction: str - Direction of the frame (either x or y)
		:param dist: float - Distance to the center of mass of the diaphragms
		:param num_frame: int - Number of frames
		:param height: float or list of floats - Either the height of every story or a list of heights of each story (m)
		:param b_width: float or list of floats - Either the width of every frame or a list of widths of each frame (m)
		:param mass: float or list of floats - Either the mass of every story or a list of masses of each story (kg)
		:param width: float - Width of the diaphragm (m)
		:param depth: float - Depth of the diaphragm (m)
		:param E: float - Elasticity module of the estructure (Pa)
		:param support: str - Type of support of the column base
		"""
		self.num_frame = num_frame
		self.num_pav = num_pav
		self.direction = direction
		self.dist = dist
		self.width = width
		self.depth = depth
		self.E = E
		self.support = support
		
		if type(height) == int or type(height) == float:
			self.height = np.array([height for i in range(num_pav)])
		else:
			self.height = np.array(height)
			
		if type(b_width) == int or type(b_width) == float:
			self.b_width = np.array([b_width for i in range(num_frame)])
		else:
			self.b_width = np.array(b_width)
			
		if type(mass) == int or type(mass) == float:
			self.mass = np.array([mass for i in range(len(self.height))])
		else:
			self.mass = np.array(mass)
		
		self.I = (self.width*self.depth**3)/12
		if support == 'Fix-Fix':
			self.stiffness = (num_frame+1)*12*self.E*self.I/(self.height**3)
		elif support == 'Fix-Pin' or support == 'Pin-Fix':
			self.stiffness = (num_frame+1)*7.5*self.E*self.I/(self.height**3)
		elif support == 'Pin-Pin':
			self.stiffness = (num_frame+1)*3*self.E*self.I/(self.height**3)

class Diaphragms(object):
	def __init__(self, num_pav, width=5, depth=5, mass=10.e3, E=25.e9, tlcds=None):
		"""
		:param num_pav: int - Number of stories of the structure
		:param width: float - Width of the diaphragm (m)
		:param depth: float - Depth of the diaphragm (m)
		:param mass: float or list of floats - Either the mass of every story or a list of masses of each story (kg)
		:param E: float - Elasticity module of the estructure (Pa)
		:param tlcds: dict - Dictionary of object TLCD, with the key relating to the positions of the TLCDs
		"""
		self.num_pav = num_pav
		self.E = E
		self.width = width
		self.depth = depth
		self.tlcds = tlcds
		
		if type(mass) == int or type(mass) == float:
			self.mass = np.array([mass for i in range(self.num_pav)])
		else:
			self.mass = np.array(mass)
			
		self.I = (self.width*self.depth**3)/12
		
		if self.tlcds is not None:
			self.total_mass = copy(self.mass)
			self.tlcds_amount = 0
			self.tlcds_amount_x = 0
			self.tlcds_amount_y = 0
			for i in self.tlcds['pos']:
				if type(i) == str:
					n = int(i[0])
				else:
					n = i
				self.total_mass[n-1] += self.tlcds[i].mass * self.tlcds[i].amount
				self.tlcds_amount += self.tlcds[i].amount
				if self.tlcds[i].direction == 'x':
					self.tlcds_amount_x += self.tlcds[i].amount
				elif self.tlcds[i].direction == 'y':
					self.tlcds_amount_y += self.tlcds[i].amount
				elif self.tlcds[i].direction == 'xy':
					self.tlcds_amount_y += self.tlcds[i].amount
					self.tlcds_amount_x += self.tlcds[i].amount
					self.total_mass[i-1] += self.tlcds[i].mass * self.tlcds[i].amount
					self.tlcds_amount += self.tlcds[i].amount
		else:
			self.total_mass = self.mass
			self.tlcds_amount = 0
	
	def addTLCD(self, tlcds):
		if self.tlcds == None:
			self.tlcds = tlcds
		else:
			for key, value in tlcds.items():
				self.tlcds[key] = value
		self.total_mass = copy(self.mass)
		self.tlcds_amount = 0
		self.tlcds_amount_x = 0
		self.tlcds_amount_y = 0
		for i in self.tlcds['pos']:
			if type(i) == str:
				n = int(i[0])
			else:
				n = i
			self.total_mass[n-1] += self.tlcds[i].mass * self.tlcds[i].amount
			self.tlcds_amount += self.tlcds[i].amount
			if self.tlcds[i].direction == 'x':
				self.tlcds_amount_x += self.tlcds[i].amount
			elif self.tlcds[i].direction == 'y':
				self.tlcds_amount_y += self.tlcds[i].amount
			elif self.tlcds[i].direction == 'xy':
				self.tlcds_amount_y += self.tlcds[i].amount
				self.tlcds_amount_x += self.tlcds[i].amount
				self.total_mass[i-1] += self.tlcds[i].mass * self.tlcds[i].amount
				self.tlcds_amount += self.tlcds[i].amount
	
	def removeTLCD(self):
		if self.tlcds is not None:
			self.total_mass = self.mass
			self.tlcds_amount = 0
			self.amount_x = 0
			self.amount_y = 0
			self.tlcds = None