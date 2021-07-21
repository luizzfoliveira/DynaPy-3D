from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from itertools import cycle

# TODO add docstrings


class PltCanvas(FigureCanvas):
	def __init__(self, parent=None, width=5, height=4, dpi=100):
		self.parent = parent
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.axes = self.fig.add_subplot(111)

		FigureCanvas.__init__(self, self.fig)
		self.setParent(parent)

		FigureCanvas.setSizePolicy(self,
								   QSizePolicy.Expanding,
								   QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)

	def plot_displacement(self, dynamicResponse, plotList, numberOfStories=1, model='tridimensional'):
		self.axes.cla()

		cycol = cycle('brgcmk')

		t = dynamicResponse.t

		if model == 'shear building':
			for i, j in plotList:
				if j:
					if not ('TLCD' in i):
						n = int(i.split('Story ')[1]) - 1
						x = dynamicResponse.x[n, :].A1
						self.axes.plot(t, x, c=next(cycol), label=i)
					else:
						tlcdNumber = int(i.strip('TLCD at story (direction: x & y)'))
						n = numberOfStories + tlcdNumber - 1
						x = dynamicResponse.x[n, :].A1
						self.axes.plot(t, x, c=next(cycol), label='TLCD {}'.format(tlcdNumber))
		else:
			for i, j in plotList:
				if j:
					if not ('TLCD' in i):
						direction = i.split()[-1]
						if 'x' in direction:
							n = (int(i.split()[1]) - 1) % numberOfStories
						elif 'y' in direction:
							n = (int(i.split()[1]) - 1) % numberOfStories + numberOfStories
						else:
							n = (int(i.split()[1]) - 1) % numberOfStories + numberOfStories * 2
						x = dynamicResponse.x[n, :].A1
						self.axes.plot(t, x, c=next(cycol), label=i)
					else:
						tlcdNumber = int(i.strip('TLCD at story (direction: x & y)'))
						n = numberOfStories * 3 + tlcdNumber - 1
						x = dynamicResponse.x[n, :].A1
						self.axes.plot(t, x, c=next(cycol), label='TLCD {}'.format(tlcdNumber))

		self.axes.legend(fontsize=11)
		self.axes.set_title('Displacement Vs. Time')

		self.axes.set_xlabel('t (s)')
		self.axes.set_ylabel('x (m)')
		self.fig.tight_layout()
		self.draw()

	def plot_velocity(self, dynamicResponse, plotList, numberOfStories, model):
		self.axes.cla()

		cycol = cycle('brgcmk')

		t = dynamicResponse.t

		if model == 'shear building':
			for i, j in plotList:
				if j:
					if not ('TLCD' in i):
						n = int(i.split('Story ')[1]) - 1
						v = dynamicResponse.v[n, :].A1
						self.axes.plot(t, v, c=next(cycol), label=i)
					else:
						tlcdNumber = int(i.strip('TLCD at story (direction: x & y)'))
						n = numberOfStories + tlcdNumber - 1
						v = dynamicResponse.v[n, :].A1
						self.axes.plot(t, v, c=next(cycol), label='TLCD {}'.format(tlcdNumber))
		else:
			for i, j in plotList:
				if j:
					if not ('TLCD' in i):
						direction = i.split()[-1]
						if 'x' in direction:
							n = (int(i.split()[1]) - 1) % numberOfStories
						elif 'y' in direction:
							n = (int(i.split()[1]) - 1) % numberOfStories + numberOfStories
						else:
							n = (int(i.split()[1]) - 1) % numberOfStories + numberOfStories * 2
						v = dynamicResponse.v[n, :].A1
						self.axes.plot(t, v, c=next(cycol), label=i)
					else:
						tlcdNumber = int(i.strip('TLCD at story (direction: x & y)'))
						n = numberOfStories * 3 + tlcdNumber - 1
						v = dynamicResponse.v[n, :].A1
						self.axes.plot(t, v, c=next(cycol), label='TLCD {}'.format(tlcdNumber))
		
		self.axes.legend(fontsize=11)
		self.axes.set_title('Velocity Vs. Time')

		self.axes.set_xlabel('t (s)')
		self.axes.set_ylabel('v (m/s)')
		self.fig.tight_layout()
		self.draw()

	def plot_acceleration(self, dynamicResponse, plotList, numberOfStories, model):
		self.axes.cla()

		cycol = cycle('brgcmk')

		t = dynamicResponse.t

		if model == 'shear building':
			for i, j in plotList:
				if j:
					if not ('TLCD' in i):
						n = int(i.split('Story ')[1]) - 1
						a = dynamicResponse.a[n, :].A1
						self.axes.plot(t, a, c=next(cycol), label=i)
					else:
						tlcdNumber = int(i.strip('TLCD at story (direction: x & y)'))
						n = numberOfStories + tlcdNumber - 1
						a = dynamicResponse.a[n, :].A1
						self.axes.plot(t, a, c=next(cycol), label='TLCD {}'.format(tlcdNumber))
		else:
			for i, j in plotList:
				if j:
					if not ('TLCD' in i):
						direction = i.split()[-1]
						if 'x' in direction:
							n = (int(i.split()[1]) - 1) % numberOfStories
						elif 'y' in direction:
							n = (int(i.split()[1]) - 1) % numberOfStories + numberOfStories
						else:
							n = (int(i.split()[1]) - 1) % numberOfStories + numberOfStories * 2
						a = dynamicResponse.a[n, :].A1
						self.axes.plot(t, a, c=next(cycol), label=i)
					else:
						tlcdNumber = int(i.strip('TLCD at story (direction: x & y)'))
						n = numberOfStories * 3 + tlcdNumber - 1
						a = dynamicResponse.a[n, :].A1
						self.axes.plot(t, a, c=next(cycol), label='TLCD {}'.format(tlcdNumber))

		self.axes.legend(fontsize=11)
		self.axes.set_title('Acceleration Vs. Time')

		self.axes.set_xlabel('t (s)')
		self.axes.set_ylabel(r'a (m/$s^2$)')
		self.fig.tight_layout()
		self.draw()

	def plot_excitation(self, t, a, unit='m/$s^2$'):
		self.axes.cla()
		self.axes.plot(t, a)
		self.axes.set_xlabel('t (s)')
		self.axes.set_ylabel(r'a ({})'.format(unit))
		self.draw()

	def plot_dmf(self, outputDMF, plotList):
		self.axes.cla()

		cycol = cycle('brgcmk')
		frequencies = outputDMF.frequencies
		dmf = outputDMF.dmf

		for i, j in plotList:
			if j:
				n = int(i.split('Story ')[1]) - 1
				self.axes.plot(frequencies, dmf[:, n].A1, c=next(cycol), label=i)

		self.axes.legend(fontsize=11)
		self.axes.set_title('DMF Vs. Excitation Frequency')
		self.axes.set_xlabel('Excitation Frequency (rad/s)')
		self.axes.set_ylabel('DMF')
		self.fig.tight_layout()
		self.draw()

	def plot_displacement_frequency(self, outputDMF, plotList):
		self.axes.cla()

		cycol = cycle('brgcmk')
		frequencies = outputDMF.frequencies
		x = outputDMF.displacements

		for i, j in plotList:
			if j:
				n = int(i.split('Story ')[1]) - 1
				self.axes.plot(frequencies, x[:, n].A1, c=next(cycol), label=i)

		self.axes.legend(fontsize=11)
		self.axes.set_title('Maximum Displacement Vs. Excitation Frequency')
		self.axes.set_xlabel('Excitation Frequency (rad/s)')
		self.axes.set_ylabel('Maximum Displacement (m)')
		self.fig.tight_layout()
		self.draw()

	def reset_canvas(self):
		self.axes.cla()
		self.draw()

	def plot_dis_vel(self, dynamicResponse, plotList, numberOfStories, model):
		self.axes.cla()

		cycol = cycle('brgcmk')

		if model == 'shear building':
			for i, j in plotList:
				if j:
					if not ('TLCD' in i):
						n = int(i.split('Story ')[1]) - 1
						x = dynamicResponse.x[n, :].A1
						v = dynamicResponse.v[n, :].A1
						self.axes.plot(x, v, c=next(cycol), label=i)
					else:
						tlcdNumber = int(i.strip('TLCD at story (direction: x & y)'))
						n = numberOfStories + tlcdNumber - 1
						x = dynamicResponse.x[n, :].A1
						v = dynamicResponse.v[n, :].A1
						self.axes.plot(x, v, c=next(cycol), label='TLCD {}'.format(tlcdNumber))
		else:
			for i, j in plotList:
				if j:
					if not ('TLCD' in i):
						direction = i.split()[-1]
						if 'x' in direction:
							n = (int(i.split()[1]) - 1) % numberOfStories
						elif 'y' in direction:
							n = (int(i.split()[1]) - 1) % numberOfStories + numberOfStories
						else:
							n = (int(i.split()[1]) - 1) % numberOfStories + numberOfStories * 2
						x = dynamicResponse.x[n, :].A1
						v = dynamicResponse.v[n, :].A1
						self.axes.plot(x, v, c=next(cycol), label=i)
					else:
						tlcdNumber = int(i.strip('TLCD at story (direction: x & y)'))
						n = numberOfStories * 3 + tlcdNumber - 1
						x = dynamicResponse.x[n, :].A1
						v = dynamicResponse.v[n, :].A1
						self.axes.plot(x, v, c=next(cycol), label='TLCD {}'.format(tlcdNumber))

		self.axes.legend(fontsize=11)
		self.axes.set_title('Displacement Vs. Velocity')

		self.axes.set_xlabel('x (m)')
		self.axes.set_ylabel('v (m/s)')
		self.fig.tight_layout()
		self.draw()

	def stepbystep_subplots(self, n, x, y, labels):
		self.fig.clf()
		
		if n == 1:
			self.ax1 = self.fig.add_subplot(111)
			self.ax1.plot(x[0], y[0])
		elif n == 2:
			self.ax1 = self.fig.add_subplot(212)
			self.ax1.plot(x[0], y[0])
			self.ax2 = self.fig.add_subplot(211)
			self.ax2.plot(x[1], y[1])
		elif n == 3:
			self.ax1 = self.fig.add_subplot(313)
			self.ax1.plot(x[0], y[0])
			self.ax2 = self.fig.add_subplot(312)
			self.ax2.plot(x[1], y[1])
			self.ax3 = self.fig.add_subplot(311)
			self.ax3.plot(x[2], y[2])

		try:
			self.ax1.set_xlabel(labels[0])
			self.ax1.set_ylabel(labels[1])
			self.ax2.set_ylabel(labels[2])
			self.ax3.set_ylabel(labels[3])
		except:
			pass
		
		self.fig.tight_layout()
		self.draw()