import os
import re
import sys
import functools
import pandas as pd
from matplotlib import pyplot as plt

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from DynaPy import *
from GUI.mainWindowGUI import Ui_MainWindow
from GUI.tunerDialog import Ui_Dialog
from GUI.boundaryConditionsDialog import Ui_BoundaryConditionsDialog
from GUI.fluidParametersDialog import Ui_FluidParametersDialog
from GUI.dmfSettingsDialog import Ui_DMFSettingsDialog
from GUI.stepSizeDialog import Ui_StepSizeDialog
from GUI.structureDampingDialog import Ui_StructureDampingDialog

class Input(object):
	def __init__(self):
		self.frames = {}
		self.tlcds = {'pos' : []}
		self.diaphragms = None
		self.excitation = None
		self.configurations = Configurations()

def printError(errorMsg):
	QMessageBox.warning(DpMainWindow(), "Error", errorMsg, QMessageBox.Ok)

class DpMainWindow(QMainWindow, Ui_MainWindow):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setupUi(self)
		self.setWindowIcon(QIcon('./img/icon_64.ico'))
		self.connectDialogs()
		self.Input = Input()
		self.frameNumbers = []
		self.report = None
		self.connect()

		# Structure Canvas
		self.structureWidget.structureCanvas = StructureCanvas(self.structureWidget)
		self.structureWidget.grid = QGridLayout()
		self.structureWidget.grid.addWidget(self.structureWidget.structureCanvas, 1, 1)
		self.structureWidget.setLayout(self.structureWidget.grid)
		self.planWidget.planCanvas = PlanCanvas(self.planWidget)
		self.planWidget.grid = QGridLayout()
		self.planWidget.grid.addWidget(self.planWidget.planCanvas, 1, 1)
		self.planWidget.setLayout(self.planWidget.grid)

		#TLCD Canvas
		self.tlcdWidget.tlcdCanvas = TLCDCanvas(self.tlcdWidget)
		self.tlcdWidget.grid = QGridLayout()
		self.tlcdWidget.grid.addWidget(self.tlcdWidget.tlcdCanvas, 1, 1)
		self.tlcdWidget.setLayout(self.tlcdWidget.grid)

		# Excitation Canvas
		self.excitationWidget.excitationCanvas = PltCanvas()
		self.excitationWidget.mpl_toolbar = NavigationToolbar(self.excitationWidget.excitationCanvas, self)
		self.excitationWidget.gridLabel = QLabel('Show Grid', self)
		self.excitationWidget.gridChkBox = QCheckBox(self)
		self.excitationWidget.gridChkBox.stateChanged.connect(self.excitation_grid_toggle)

		self.excitationWidget.gridLayout = QGridLayout()
		self.excitationWidget.gridLayout.addWidget(self.excitationWidget.excitationCanvas, 1, 1, 1, 3)
		self.excitationWidget.gridLayout.addWidget(self.excitationWidget.gridLabel, 2, 1)
		self.excitationWidget.gridLayout.addWidget(self.excitationWidget.gridChkBox, 2, 2)
		self.excitationWidget.gridLayout.addWidget(self.excitationWidget.mpl_toolbar, 2, 3)

		self.excitationWidget.setLayout(self.excitationWidget.gridLayout)

		# Plot Canvas
		self.dynRespWidget.dynRespCanvas = PltCanvas()
		self.dynRespWidget.mpl_toolbar = NavigationToolbar(self.dynRespWidget.dynRespCanvas, self)
		self.dynRespWidget.exportBtn = QPushButton('Export CSV', self)
		self.dynRespWidget.gridLabel = QLabel('Show Grid', self)
		self.dynRespWidget.gridChkBox = QCheckBox(self)
		self.dynRespWidget.gridChkBox.stateChanged.connect(self.dynamic_response_grid_toggle)
		self.dynRespWidget.exportBtn.clicked.connect(self.dynamic_response_export_csv)

		self.dynRespWidget.gridLayout = QGridLayout()
		self.dynRespWidget.gridLayout.addWidget(self.dynRespWidget.dynRespCanvas, 1, 1, 1, 4)
		self.dynRespWidget.gridLayout.addWidget(self.dynRespWidget.gridLabel, 2, 1)
		self.dynRespWidget.gridLayout.addWidget(self.dynRespWidget.gridChkBox, 2, 2)
		self.dynRespWidget.gridLayout.addWidget(self.dynRespWidget.mpl_toolbar, 2, 3)
		self.dynRespWidget.gridLayout.addWidget(self.dynRespWidget.exportBtn, 2, 4)

		self.dynRespWidget.setLayout(self.dynRespWidget.gridLayout)

	def connect(self):
		self.frameNumberComboBox.currentIndexChanged.connect(self.changeStructureCanvas)
		self.addFrameBtn.clicked.connect(self.getFrameInput)
		self.removeFrameBtn.clicked.connect(self.removeFrame)
		self.updateDiaphragmsBtn.clicked.connect(self.getDiaphragmsInput)
		self.tlcdModelComboBox.currentIndexChanged.connect(self.changeTLCDWindow)
		self.confirmTlcdBtn.clicked.connect(self.getTLCDInput)
		self.excitationTypeComboBox.currentIndexChanged.connect(self.changeExctWindow)
		self.importExcitationButton.clicked.connect(self.import_excitation)
		self.confirmExcitationButton.clicked.connect(self.getExcitationInput)
		self.actionRunDynamicResponse.triggered.connect(self.runCalculations)
		self.actionFiniteDifferenceMethod.triggered.connect(functools.partial(self.uncheckMethods,'Finite Differences Method'))
		self.actionNewmarkMethod.triggered.connect(functools.partial(self.uncheckMethods, 'Newmark Method'))
		self.actionRungeKuttaMethod.triggered.connect(functools.partial(self.uncheckMethods, 'Ruge-Kutta Method'))
		self.actionLinearAccelerationMethod.triggered.connect(functools.partial(self.uncheckMethods, 'Linear Acceleration Method'))
		self.actionAverageAccelerationMethod.triggered.connect(functools.partial(self.uncheckMethods, 'Average Acceleration Method'))
		self.actionBoundaryConditions.triggered.connect(self.boundaryConditions)
		self.actionFluidParameters.triggered.connect(self.fluidParameters)
		self.actionDMFSettings.triggered.connect(self.dmfSettings)
		self.actionStepSize.triggered.connect(self.stepSize)
		self.actionStructureDamping.triggered.connect(self.structureDamping)
		self.actionMaximize.triggered.connect(self.showMaximized)
		self.actionFullScreen.triggered.connect(self.toggle_full_screen)
		self.actionAbout.triggered.connect(self.about)
		self.actionNewFile.triggered.connect(self.newFile)
		self.actionExportReport.triggered.connect(self.exportReport)
		self.actionQuit.triggered.connect(self.close)
		self.addToPlotButton.clicked.connect(self.dynamic_response_add_list2_item)
		self.removeFromPlotButton.clicked.connect(self.dynamic_response_remove_list2_item)
		self.plotButton.clicked.connect(self.plot_dyn_resp)

	def newFile(self):
		""" Resets all GUI inputs, inputData variable and save file.

		:return: None
		"""

		quit_msg = "This operation will erase all unsaved data. Are you sure you want to open a new file?"
		reply = QMessageBox.question(self, 'Confirm New File',
									 quit_msg, QMessageBox.Yes, QMessageBox.No)

		if reply == QMessageBox.No:
			return

		# Set stories combobox index to 0
		self.frameNumberComboBox.setCurrentIndex(0)

		# Reset inputData
		self.Input = Input()
		self.frameNumbers = []

		# Reset method
		self.uncheckMethods('Finite Differences Method')

		# Reset save file
		self.fileName = None
		#self.setWindowTitle('Dynapy')

		# Reset GUI
		for i in range(self.frameNumberComboBox.count() - 1, 0, -1):
			self.frameNumberComboBox.removeItem(i)
		self.numberStoriesLineEdit.setText('')
		self.storyMassLineEdit.setText('')
		self.buildingWidthLineEdit.setText('')
		self.buildingDepthLineEdit.setText('')
		self.numberColumnsLineEdit.setText('')
		self.frameDirectionComboBox.setCurrentIndex(0)
		self.frameDistLineEdit.setText('')
		self.storyHeightLineEdit.setText('')
		self.columnWidthLineEdit.setText('')
		self.columnDepthLineEdit.setText('')
		self.elasticityModuleLineEdit.setText('')
		self.supportTypeComboBox.setCurrentIndex(0)
		self.structureWidget.structureCanvas.painter(None, None, None)
		self.planWidget.planCanvas.firstPainter(None)

		self.tlcdModelComboBox.setCurrentIndex(0)
		self.diameterSimpleTlcdLineEdit.setText('')
		self.waterLevelSimpleTlcdLineEdit.setText('')
		self.widthSimpleTlcdLineEdit.setText('')
		self.amountSimpleTlcdLineEdit.setText('')
		self.positionTlcdLineEdit.setText('')
		self.directionTlcdComboBox.setCurrentIndex(0)
		self.contractionSimpleTlcdLineEdit.setText('')
		self.diameterPressureTlcdLineEdit.setText('')
		self.waterLevelPressureTlcdLineEdit.setText('')
		self.widthPressureTlcdLineEdit.setText('')
		self.gasHeightPressureTlcdLineEdit.setText('')
		self.gasPressurePressureTlcdLineEdit.setText('')
		self.amountPressureTlcdLineEdit.setText('')
		self.positionPressureTlcdLineEdit.setText('')
		self.directionPressureTlcdComboBox.setCurrentIndex(0)
		self.contractionPressureTlcdLineEdit.setText('')
		self.tlcdWidget.tlcdCanvas.painter(None)

		self.excitationTypeComboBox.setCurrentIndex(0)
		self.sineAmplitudeLineEdit.setText('')
		self.sineFrequencyLineEdit.setText('')
		self.sineAnalysisDurationLineEdit.setText('')
		self.sineExcitationDurationLineEdit.setText('')
		self.sineDirectionComboBox.setCurrentIndex(0)
		self.sineFrequencyRatioCheckBox.setChecked(False)
		#self.excitationWidget.excitationCanvas.plot_excitation([], [])
		self.excitationFileLineEdit.setText('')
		self.generalDirectionComboBox.setCurrentIndex(0)

		self.reportTextBrowser.setText("""Report not yet generated.
Fill all the input fields and and Run the calculations to generate report.""")
		self.reportTextBrowser.setFont(QFont("Times", 14))

		self.list1.clear()
		self.list2.clear()
		self.dynRespWidget.dynRespCanvas.reset_canvas()

		self.tabWidget.setCurrentIndex(0)
	
	def exportReport(self):
		if self.report is None:
			return printError("No Report Yet")
		fileName = QFileDialog.getSaveFileName(self, 'Save as', './Untitled.txt', filter="Text File (*.txt)")[0]
		print(fileName)
		file = open(fileName, 'w', encoding='utf-8')
		file.write(self.report)
		file.close

	def toggle_full_screen(self):
		"""
		Method that checks if application is on full screen or not and toggles state.
		:return: None
		"""
		if self.isFullScreen():
			self.showMaximized()
		else:
			self.showFullScreen()
	
	def about(self):
		"""
		Action for showing the application 'About'.
		Shortcut: F1
		:return: None
		"""
		about_title = "DynaPy TLCD Analyser - Version 0.1 of 3D analysis"
		about_msg = "Authors: Luiz Felipe Carvalhedo de Oliveira and Mario Raul Freitas\n\n" + \
					"This software takes as input data a shear-build or a tridimensional buiding made of multiple frames, equipped with a Tuned Liquid Column " + \
					"Damper (TLCD) and calculates the dynamic response at each story. Furthermore, this software " + \
					"is capable of making analysis for different frequencies and calculate the best possible configuration of a TLCD " + \
					"so that it is tuned to the building natural frequency. This software is intended for educational use. " + \
					"The authors do not take responsibility for any sort of misuse of the software or for its " + \
					"results. The user of this software is responsible for any conclusion taken using " + \
					"this software. There is no commitment or warranty implied in the use of the software"
		QMessageBox.information(self, about_title, about_msg, QMessageBox.Ok)
	
	def structureDamping(self):
		structure = StructureDampingDialog()
		structure.exec()
		self.Input.configurations.dampingRatio = structure.dampingRatio

	def stepSize(self):
		step = StepSizeDialog()
		step.exec()
		self.Input.configurations.timeStep = step.timeStep

	def dmfSettings(self):
		settings = DMFSettingsDialog()
		settings.exec()
		self.Input.configurations.dmfDiscretizationPoints = settings.discretizationPoints
		self.Input.configurations.dmfUpperLimitFactor = settings.upperLimitFactor
	
	def fluidParameters(self):
		fParameters = FluidParametersDialog()
		fParameters.exec()
		self.Input.configurations.liquidSpecificMass = fParameters.specificMass
		self.Input.configurations.kineticViscosity = fParameters.kineticViscosity
	
	def boundaryConditions(self):
		bConditions = BoundaryConditionsDialog()
		bConditions.exec()
		self.Input.configurations.initialDisplacement = bConditions.x0
		self.Input.configurations.initialVelocity = bConditions.v0

	def uncheckMethods(self, method):
		if method == 'Finite Differences Method':
			self.actionFiniteDifferenceMethod.setChecked(True)
		elif method == 'Newmark Method':
			self.actionNewmarkMethod.setChecked(True)
		elif method == 'Ruge-Kutta Method':
			self.actionRungeKuttaMethod.setChecked(True)
		elif method == 'Linear Acceleration Method':
			self.actionLinearAccelerationMethod.setChecked(True)
		elif method == 'Average Acceleration Method':
			self.actionAverageAccelerationMethod.setChecked(True)
		if method != 'Finite Differences Method':
			self.actionFiniteDifferenceMethod.setChecked(False)
		if method != 'Newmark Method':
			self.actionNewmarkMethod.setChecked(False)
		if method != 'Ruge-Kutta Method':
			self.actionRungeKuttaMethod.setChecked(False)
		if method != 'Linear Acceleration Method':
			self.actionLinearAccelerationMethod.setChecked(False)
		if method != 'Average Acceleration Method':
			self.actionAverageAccelerationMethod.setChecked(False)
		self.Input.configurations.method = method

	def getFrameInput(self):
		frameNumber = int(get_text(self.frameNumberComboBox))
		if frameNumber not in self.frameNumbers:
			self.frameNumbers.append(frameNumber)
			self.frameNumberComboBox.addItem(str(frameNumber + 1))
		num_stories = int(get_text(self.numberStoriesLineEdit))
		num_frames = int(get_text(self.numberColumnsLineEdit)) - 1
		width = float(get_text(self.columnWidthLineEdit))
		depth = float(get_text(self.columnDepthLineEdit))
		E = float(get_text(self.elasticityModuleLineEdit)) * 1e9
		height = get_text(self.storyHeightLineEdit)
		height_list = re.split(',| ', height)
		height_list = [float(i) for i in height_list if i != '']
		if len(height_list) == 1:
			height_list = height_list[0]
		direction = get_text(self.frameDirectionComboBox)
		support = get_text(self.supportTypeComboBox)
		dist = float(get_text(self.frameDistLineEdit))
		self.Input.frames[frameNumber] = Frame(num_pav=num_stories, direction=direction, dist=dist,
		num_frame=num_frames, height=height_list, width=width, depth=depth, E=E, support=support)
		self.frameNumberComboBox.setCurrentIndex(self.frameNumberComboBox.currentIndex() + 1)
		if self.Input.diaphragms is None:
			printError("Add Diaphragm(s) To Get Structure Drawing")
			return
		self.structureWidget.structureCanvas.painter(self.Input.frames[frameNumber], self.Input.diaphragms, frameNumber)
		self.planWidget.planCanvas.painter(self.Input.frames)
	
	def changeStructureCanvas(self):
		frameNumber = int(get_text(self.frameNumberComboBox))
		if frameNumber in self.frameNumbers:
			self.structureWidget.structureCanvas.painter(self.Input.frames[frameNumber], self.Input.diaphragms, frameNumber)
	
	def removeFrame(self):
		frameNumber = int(get_text(self.frameNumberComboBox))
		if frameNumber > len(self.frameNumbers):
			return printError("Frame Not Created Yet")
		# Check if is the last frame
		if len(self.frameNumbers) == 0:
			printError("No Frames Yet")
			return
		if frameNumber == self.frameNumbers[-1]:
			del(self.Input.frames[frameNumber])
			del(self.frameNumbers[-1])
			self.frameNumberComboBox.removeItem(frameNumber)
		else:
			for i in range(frameNumber - 1, len(self.frameNumbers) - 1):
				self.Input.frames[self.frameNumbers[i]] = self.Input.frames[self.frameNumbers[i + 1]]
			del(self.Input.frames[self.frameNumbers[-1]])
			self.frameNumberComboBox.removeItem(self.frameNumbers[-1])
			del(self.frameNumbers[-1])
		self.frameNumberComboBox.setCurrentIndex(self.frameNumbers[-1])

	
	def getDiaphragmsInput(self):
		num_stories = int(get_text(self.numberStoriesLineEdit))
		width = float(get_text(self.buildingWidthLineEdit))
		depth = float(get_text(self.buildingDepthLineEdit))
		E = float(get_text(self.elasticityModuleLineEdit)) * 1e9
		mass = get_text(self.storyMassLineEdit)
		mass_list = re.split(',| ', mass)
		mass_list = [float(i) * 1000 for i in mass_list if i != '']
		if len(mass_list) == 1:
			mass_list = mass_list[0]
		self.Input.diaphragms = Diaphragms(num_pav=num_stories, width=width, depth=depth, mass=mass_list, E=E)
		self.planWidget.planCanvas.firstPainter(self.Input.diaphragms)
	
	def changeTLCDWindow(self):
		window = get_text(self.tlcdModelComboBox)
		if window == 'None':
			self.tlcdStackedWidget.setCurrentIndex(0)
		elif window == 'Basic TLCD':
			self.tlcdStackedWidget.setCurrentIndex(1)
		elif window == 'Pressurized TLCD':
			self.tlcdStackedWidget.setCurrentIndex(2)

	def getTLCDInput(self):
		tlcdType = get_text(self.tlcdModelComboBox)
		if tlcdType == 'None':
			if len(self.Input.tlcds) > 1:
				self.Input.diaphragms.removeTLCD()
				self.Input.tlcds = {'pos' : []}
				self.tlcdWidget.tlcdCanvas.painter(None)
			return
		if len(self.Input.frames) == 0:
			printError("Missing Frame(s)")
			return
		if self.Input.diaphragms is None:
			printError("Missing Diaphragm(s)")
			return
		self.Input.configurations.nonLinearAnalysis = True
		if tlcdType == 'Basic TLCD':
			if self.Input.diaphragms.tlcds is not None:
				self.Input.diaphragms.tlcds = None
			D = float(get_text(self.diameterSimpleTlcdLineEdit)) / 100
			waterHeight = float(get_text(self.waterLevelSimpleTlcdLineEdit)) / 100
			width = float(get_text(self.widthSimpleTlcdLineEdit)) / 100
			amount = int(get_text(self.amountSimpleTlcdLineEdit))
			position = get_text(self.positionTlcdLineEdit)
			if position == 'Last':
				position_list = [self.Input.frames[1].num_pav]
			else:
				position_list = re.split(',| ', position)
				position_list = [int(i) for i in position_list if i != '']
				position_list = list(dict.fromkeys(position_list))
			self.Input.tlcds['pos'] = position_list
			direction = get_text(self.directionTlcdComboBox)
			contraction = float(get_text(self.contractionSimpleTlcdLineEdit))
			for i in position_list:
				self.Input.tlcds[i] = TLCD(tlcdType=tlcdType, diameter=D, width=width, waterHeight=waterHeight, amount=amount,
				contraction=contraction, pos=i, direction=direction, configurations=self.Input.configurations)
		elif tlcdType == 'Pressurized TLCD':
			if self.Input.diaphragms.tlcds is not None:
				self.Input.diaphragms.tlcds = None
			D = float(get_text(self.diameterPressureTlcdLineEdit)) / 100
			waterHeight = float(get_text(self.waterLevelPressureTlcdLineEdit)) / 100
			width = float(get_text(self.widthPressureTlcdLineEdit)) / 100
			gasHeight = float(get_text(self.gasHeightPressureTlcdLineEdit)) / 100
			gasPressure = float(get_text(self.gasPressurePressureTlcdLineEdit)) * 101325
			amount = int(get_text(self.amountPressureTlcdLineEdit))
			position = get_text(self.positionPressureTlcdLineEdit)
			if position == 'Last':
				position_list = [self.Input.frames[1].num_pav]
			else:
				position_list = re.split(',| ', position)
				position_list = [int(i) for i in position_list if i != '']
				position_list = list(dict.fromkeys(position_list))
			self.Input.tlcds['pos'] = position_list
			direction = get_text(self.directionTlcdComboBox)
			contraction = float(get_text(self.contractionSimpleTlcdLineEdit))
			for i in position_list:
				self.Input.tlcds[i] = TLCD(tlcdType=tlcdType, diameter=D, width=width, waterHeight=waterHeight, gasHeight=gasHeight,
				gasPressure=gasPressure, amount=amount, contraction=contraction, pos=i, direction=direction, configurations=self.Input.configurations)
		self.Input.diaphragms.addTLCD(self.Input.tlcds)
		self.tlcdWidget.tlcdCanvas.painter(self.Input.tlcds[position_list[0]])

	def changeExctWindow(self):
		window = get_text(self.excitationTypeComboBox)
		if window == 'Sine Wave':
			self.excitationStackedWidget.setCurrentIndex(0)
		elif window == 'General Excitation':
			self.excitationStackedWidget.setCurrentIndex(1)

	def getExcitationInput(self):
		exctType = get_text(self.excitationTypeComboBox)
		if exctType == 'Sine Wave':
			direction = get_text(self.sineDirectionComboBox)
			amplitude = float(get_text(self.sineAmplitudeLineEdit))
			frequency = float(get_text(self.sineFrequencyLineEdit))
			exctDuration = float(get_text(self.sineExcitationDurationLineEdit))
			anlyDuration = float(get_text(self.sineAnalysisDurationLineEdit))
			self.Input.excitation = Excitation(exctType=exctType, direction=direction, amplitude=amplitude, frequency=frequency, exctDuration=exctDuration, anlyDuration=anlyDuration)
			tAnly = np.arange(0, anlyDuration + self.Input.configurations.timeStep,
							self.Input.configurations.timeStep)
			tExct = np.arange(0, exctDuration + self.Input.configurations.timeStep,
								self.Input.configurations.timeStep)
			a = amplitude * np.sin(self.Input.excitation.frequency * tExct)
			a = np.hstack((a, np.array([0 for i in range(len(tAnly) - len(tExct))])))
			self.excitationWidget.excitationCanvas.plot_excitation(tAnly, a)
		elif exctType == 'General Excitation':
			direction = get_text(self.generalDirectionComboBox)
			fName = get_text(self.excitationFileLineEdit)
			try:
				df = pd.read_csv(fName, header=None)
				t = np.array(df[0])
				a = np.array(df[1])
				self.Input.excitation = Excitation(exctType=exctType, direction=direction, t=t, a=a, fileName=fName)
				self.excitationWidget.excitationCanvas.plot_excitation(self.Input.excitation.t_input,
																	   self.Input.excitation.a_input)
			except:
				printError("Invalid Text File")
	
	def import_excitation(self):
		fileName = QFileDialog.getOpenFileName(self, 'Load File', './save/Excitations', filter="Text File (*.txt)")[0]
		self.excitationFileLineEdit.setText(fileName)
	
	def excitation_grid_toggle(self):
		""" Toggles plot grid on and off

		:return: None
		"""
		self.excitationWidget.excitationCanvas.axes.grid(self.excitationWidget.gridChkBox.isChecked())
		self.excitationWidget.excitationCanvas.draw()

	def connectDialogs(self):
		self.actionTlcdOptimization.triggered.connect(self.tlcdOptimization)

	def runCalculations(self):
		if len(self.Input.frames) == 0:
			printError("Missing Frame(s)")
			return
		if self.Input.diaphragms is None:
			printError("Missing Diaphragm(s)")
			return
		if self.Input.excitation == None:
			printError("Missing Excitation")
			return
		diaphragms = self.Input.diaphragms
		frames = self.Input.frames
		if len(self.Input.frames.keys()) == 1:
			model = 'shear building'
			self.M = assemble_mass_matrix(diaphragms=diaphragms, model=model)
			self.K = assemble_lat_stiffness_matrix(frame=frames[1], diaphragms=diaphragms)
			self.w_n , _ = calc_natural_frequencies(self.M, self.K)
			self.C, self.Ksi = assemble_damping_matrix(diaphragms=diaphragms, frame=frames[1], M=self.M, K=self.K, w=self.w_n, model=model)
			self.F, _ = assemble_force_matrix(mass=self.M, configurations=self.Input.configurations, diaphragms=diaphragms, excitation=self.Input.excitation, model='shear building')
		else:
			model = 'tridimensional'
			self.M = assemble_mass_matrix(diaphragms=diaphragms, model=model)
			self.K = assemble_total_stiffness_matrix(frames=frames, diaphragms=diaphragms)
			self.w_n , _ = calc_natural_frequencies(self.M, self.K)
			self.C, self.Ksi = assemble_damping_matrix(diaphragms=diaphragms, M=self.M, K=self.K, w=self.w_n, model=model)
			if 'x' in self.Input.excitation.direction:
				excitation_x = self.Input.excitation
			else:
				excitation_x = None
			if 'y' in self.Input.excitation.direction:
				excitation_y = self.Input.excitation
			else:
				excitation_y = None
			self.F, _ = assemble_force_matrix(mass=self.M, configurations=self.Input.configurations, diaphragms=diaphragms, excitation_x=excitation_x, excitation_y=excitation_y, model='tridimensional')
		self.solver = ODESolver(mass=self.M, damping=self.C, stiffness=self.K, force=self.F, configurations=self.Input.configurations, tlcds=diaphragms.tlcds)
		self.writeReport()
		self.tabWidget.setCurrentIndex(3)
		self.dynamic_response_add_list1_items()
	
	def writeReport(self):
		# Check if diverge
		if np.isnan(self.solver.x[0, -1]):
			self.reportTextBrowser.setText("Decrease Time Step")
			return
		# Building Data
		numFramesX = 0
		numFramesY = 0
		for i in self.Input.frames:
			if 'x' in self.Input.frames[i].direction:
				numFramesX += 1
			else:
				numFramesY += 1
		numStories = len(self.Input.frames[1].height)

		for i in range(numStories):
			buildingData = """Story {}:
Mass: {} ton
Height: {} m""".format(i + 1, self.Input.diaphragms.mass[i] / 1000, self.Input.frames[1].height[i])
		
		# TLCD Data
		if len(self.Input.tlcds) <= 1:
			tlcdData = "No TLCD"
		elif self.Input.tlcds[self.Input.tlcds['pos'][0]].type == 'Basic TLCD':
			tlcdData = """Model: {}
Diameter: {} cm
Water Height: {} cm
Width: {} m
Position(s): {}""".format(self.Input.tlcds[self.Input.tlcds['pos'][0]].type, self.Input.tlcds[self.Input.tlcds['pos'][0]].diameter * 100,
					  self.Input.tlcds[self.Input.tlcds['pos'][0]].waterHeight * 100, self.Input.tlcds[self.Input.tlcds['pos'][0]].width,
					  self.Input.tlcds['pos'])
		else:
			tlcdData = """Model: {}
Diameter: {} cm
Water Height: {} cm
Width: {} m
Gas Height: {} cm
Gas Pressure: {} atm
Position(s): {}""".format(self.Input.tlcds[self.Input.tlcds['pos'][0]].type, self.Input.tlcds[self.Input.tlcds['pos'][0]].diameter * 100,
					  self.Input.tlcds[self.Input.tlcds['pos'][0]].waterHeight * 100, self.Input.tlcds[self.Input.tlcds['pos'][0]].width,
					  self.Input.tlcds[self.Input.tlcds['pos'][0]].gasHeiht * 100, self.Input.tlcds[self.Input.tlcds['pos'][0]].gasPressure / 101325,
					  self.Input.tlcds['pos'])
		
		# Excitation Data
		if self.Input.excitation.type == 'Sine Wave':
			exctData = """Excitation type: {}
Amplitude: {} m/s²
Frequency: {} rad/s
Excitation Duration: {} s
Analysis Duration: {} s
Direction: {}""".format(self.Input.excitation.type, self.Input.excitation.amplitude,
						self.Input.excitation.frequency, self.Input.excitation.exctDuration,
						self.Input.excitation.anlyDuration, self.Input.excitation.direction)
		elif self.Input.excitation.type == 'General Excitation':
			exctData = """Excitation type: {}
File: {}""".format(self.Input.excitation.type, self.Input.excitation.fileName)

		# Configurations
		configData = """Method: {}
Time Step: {} s
Initial Displacement: {} m
Initial Velocity: {} m
Structure Damping Ratio: {}
Fluid Specific Mass: {} (kg/m3)
Kinetic Viscosity: {} (m²/s)
Gravity Acceleration: {} (m/s²)""".format(self.Input.configurations.method, self.Input.configurations.timeStep,
										  self.Input.configurations.initialDisplacement, self.Input.configurations.initialVelocity,
										  self.Input.configurations.dampingRatio, self.Input.configurations.liquidSpecificMass,
										  self.Input.configurations.kineticViscosity, self.Input.configurations.gravity)

		equation = '[M]{a} + [C]{v} + [k]{x} = {F(t)}'

		if len(self.Input.frames) > 1:
			modelData = """Tridimensional
Number Of Frames In x: {}
Number Of Frames In y: {}""".format(numFramesX, numFramesY)
		else:
			modelData = 'Bidimensional'
		
		self.report = """DynaPy TLCD Analyser - Report\nAnalysis of the System Structure-TLCD Under Single Excitation Case
---------------------------------------------------------------------------------------
### Input Variables ###

# Structure

Model: {}
Number Of Stories: {}

{}

# TLCD

{}

# Excitation

{}

# Configurations

{}

### Movement Equation ###
{}

Mass Matrix (kg)
{}

Damping Matrix (kg/s)
{}

Stiffness Matrix (N/m)
{}

Force Vector (N)
{}""".format(modelData, numStories, buildingData, tlcdData, exctData, configData,
			 equation, self.solver.M, self.solver.C, self.solver.K, self.F)

		self.reportTextBrowser.setText(self.report)
	
	def dynamic_response_add_list1_items(self):
		""" Adds all stories and the TLCD to list 1. Takes from inputData.

		:return: None
		"""
		self.list1.clear()

		numberOfStories = len(self.Input.frames[1].height)

		if len(self.Input.frames) == 1:
			for i in range(numberOfStories):
				self.list1.addItem('Story {}'.format(i + 1))
		else:
			for i in range(numberOfStories * 3):
				storyNumber = i % numberOfStories + 1
				if i < numberOfStories:
					d = 'x'
				elif i < numberOfStories * 2:
					d = 'y'
				else:
					d = 'theta'
				self.list1.addItem('Story {} (direction {})'.format(storyNumber, d))

		if self.Input.tlcds is not None:
			for i in self.Input.tlcds['pos']:
				self.list1.addItem('TLCD at story {} (direction: {})'.format(i, self.Input.tlcds[i].direction))

	def dynamic_response_add_list2_item(self):
		""" Adds the item selected on list 1 to list 2 without making duplicates. If successfull, advances one row on
		list 1 and sorts list 2 alphabetically.

		:return: None
		"""
		try:
			item = get_text(self.list1)
			row = self.list1.row(self.list1.currentItem())
		except AttributeError:
			return

		if not self.list2.findItems(item, Qt.MatchExactly):
			self.list2.addItem(item)

		if row < self.list1.count() - 1:
			self.list1.setCurrentRow(row + 1)
		self.list2.sortItems(Qt.AscendingOrder)

	def dynamic_response_remove_list2_item(self):
		""" Removes the selected item from list 2

		:return: None
		"""
		item = self.list2.currentItem()
		self.list2.takeItem(self.list2.row(item))

	def plot_dyn_resp(self):
		""" Reads the plot type and the QListWidget of DOFs to plot. Makes a list of DOFs to plot and send
		outputData.dynamicResponse and plotList to plot_displacement

		:return:
		"""
		plotType = get_text(self.plotTypeComboBox)
		plotList = []
		if len(self.Input.frames) == 1:
			model = 'shear building'
		else:
			model = 'tridimensional'

		for i in range(self.list1.count()):
			self.list1.setCurrentRow(i)
			item = get_text(self.list1)
			if not self.list2.findItems(item, Qt.MatchExactly):
				plotList.append((get_text(self.list1), False))
			else:
				plotList.append((get_text(self.list1), True))

		if plotType == 'Displacement Vs. Time':
			self.dynRespWidget.dynRespCanvas.plot_displacement(self.solver, plotList,
															   numberOfStories=len(self.Input.frames[1].height), model=model)
		elif plotType == 'Velocity Vs. Time':
			self.dynRespWidget.dynRespCanvas.plot_velocity(self.solver, plotList, numberOfStories=len(self.Input.frames[1].height), model=model)
		elif plotType == 'Acceleration Vs. Time':
			self.dynRespWidget.dynRespCanvas.plot_acceleration(self.solver, plotList, numberOfStories=len(self.Input.frames[1].height), model=model)
		elif plotType == 'Displacement Vs. Velocity':
			self.dynRespWidget.dynRespCanvas.plot_dis_vel(self.solver, plotList, numberOfStories=len(self.Input.frames[1].height), model=model)
	
	def dynamic_response_grid_toggle(self):
		""" Toggles plot grid on and off

		:return: None
		"""
		self.dynRespWidget.dynRespCanvas.axes.grid(self.dynRespWidget.gridChkBox.isChecked())
		self.dynRespWidget.dynRespCanvas.draw()

	def dynamic_response_export_csv(self):
		""" Exports CSV of the plotted data 

		:return: None
		"""
		dataList = []

		plotList = []

		numberOfStories = len(self.Input.frames[1].height)

		for i in range(self.list1.count()):
			self.list1.setCurrentRow(i)
			item = get_text(self.list1)
			if not self.list2.findItems(item, Qt.MatchExactly):
				plotList.append((get_text(self.list1), False))
			else:
				plotList.append((get_text(self.list1), True))

		t = self.solver.t
		dataList.append(t)
		if len(self.Input.frames) == 1:
			for i, j in plotList:
				if j:
					if not ('TLCD' in i):
						n = int(i.split('Story ')[1]) - 1
						x = self.solver.x[n, :].A1
					else:
						tlcdNumber = int(i.strip('TLCD at story (direction: x & y)'))
						n = numberOfStories + tlcdNumber - 1
						x = self.solver.x[n, :].A1
					dataList.append(list(x))
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
						x = self.solver.x[n, :].A1
					else:
						tlcdNumber = int(i.strip('TLCD at story (direction: x & y)'))
						n = numberOfStories * 3 + tlcdNumber - 1
						x = self.solver.x[n, :].A1
					dataList.append(list(x))

		dataList = list(zip(*dataList))

		filename = QFileDialog.getSaveFileName(self, 'Save as', './save', filter="CSV File (*.csv)")[0]
		if filename != '': 
			with open(filename, 'w') as file:
				for i in dataList:
					line = str(i)
					line = line.strip('(')
					line = line.strip(')')
					file.write('{}\n'.format(line))

	def tlcdOptimization(self):
		tuner = TunerDialog(Input=self.Input)
		tuner.exec()
		if tuner.ran:
			self.tabWidget.setCurrentIndex(1)
			if tuner.tlcdType == 'Basic TLCD':
				self.tlcdModelComboBox.setCurrentIndex(1)
				self.tlcdStackedWidget.setCurrentIndex(1)
				self.diameterSimpleTlcdLineEdit.setText(str(tuner.results['D0'] * 100))
				self.waterLevelSimpleTlcdLineEdit.setText(str(tuner.results['h0'] * 100))
				self.widthSimpleTlcdLineEdit.setText(str(tuner.results['B0'] * 100))
				self.positionTlcdLineEdit.setText(tuner.position)
				self.amountSimpleTlcdLineEdit.setText(str(tuner.amount))
				self.contractionSimpleTlcdLineEdit.setText(str(tuner.contraction))
				if tuner.direction == 'x':
					self.directionTlcdComboBox.setCurrentIndex(0)
				elif tuner.direction == 'y':
					self.directionTlcdComboBox.setCurrentIndex(1)
				else:
					self.directionTlcdComboBox.setCurrentIndex(2)
			elif tuner.tlcdType == 'Pressurized TLCD':
				self.tlcdModelComboBox.setCurrentIndex(2)
				self.tlcdStackedWidget.setCurrentIndex(2)
				self.diameterPressureTlcdLineEdit.setText(str(tuner.D * 100))
				self.waterLevelPressureTlcdLineEdit.setText(str(tuner.results['h0'] * 100))
				self.widthPressureTlcdLineEdit.setText(str(tuner.results['B0'] * 100))
				self.gasPressurePressureTlcdLineEdit.setText(str(tuner.results['P0'] / 101325))
				self.gasHeightPressureTlcdLineEdit.setText(str(tuner.gasHeight * 100))
				self.positionPressurTlcdLineEdit.setText(tuner.position)
				self.amountPressureTlcdLineEdit.setText(str(tuner.amount))
				self.contractionSimpleTlcdLineEdit.setText(str(tuner.contraction))
				if tuner.direction == 'x':
					self.directionPressureTlcdComboBox.setCurrentIndex(0)
				elif tuner.direction == 'y':
					self.directionPressureTlcdComboBox.setCurrentIndex(1)
				else:
					self.directionPressureTlcdComboBox.setCurrentIndex(2)

class TunerDialog(QDialog, Ui_Dialog):
	def __init__(self, Input, parent=None):
		super().__init__(parent)
		self.setupUi(self)
		self.ran = False
		self.Input = Input
		self.configurations = self.Input.configurations
		self.connect()
	
	def connect(self):
		self.tlcdTypeDialogComboBox.activated.connect(self.changeWindow)
		self.buttonBox.accepted.connect(self.tuneTLCD)
	
	def changeWindow(self):
		tlcdType = get_text(self.tlcdTypeDialogComboBox)
		if tlcdType == 'Basic':
			self.tlcdStackedWidget.setCurrentIndex(0)
		elif tlcdType == 'Pressurized':
			self.tlcdStackedWidget.setCurrentIndex(1)

	def tuneTLCD(self):
		if len(self.Input.frames) == 0:
			printError("Missing Frame(s)")
			return
		if self.Input.diaphragms is None:
			printError("Missing Diaphragm(s)")
			return
		self.tlcdType = get_text(self.tlcdTypeDialogComboBox) + ' TLCD'
		self.ran = True
		numStories = self.Input.diaphragms.num_pav
		frames = self.Input.frames
		diaphragms = self.Input.diaphragms
		if self.tlcdType == 'Basic TLCD':
			self.amount = int(get_text(self.amountTlcdDialogLineEdit))
			self.position = get_text(self.positionTlcdDialogLineEdit)
			if self.position == 'Last':
				position_list = [self.Input.frames[1].num_pav]
			else:
				position_list = re.split(',| ', self.position)
				position_list = [int(i) for i in position_list if i != '']
				if len(self.position) == 1:
					position_list = position_list[0]
			self.direction = get_text(self.directionTlcdDialogComboBox)
			self.contraction = float(get_text(self.contractionTlcdDialogLineEdit))
			self.massPercentage = float(get_text(self.massPercentageDialogLineEdit)) / 100
			if len(self.Input.frames.keys()) == 1:
				model = 'shear building'
				M0 = assemble_mass_matrix(diaphragms=diaphragms, model=model)
				K0 = assemble_lat_stiffness_matrix(frame=frames[1], diaphragms=diaphragms)
				w_n , _ = calc_natural_frequencies(M0, K0)
				self.results = tuner(num_pav=numStories, frames=frames[1], M0=M0, configurations=self.configurations, w=w_n, massPercentage=self.massPercentage,
				tlcdType=self.tlcdType, pos=position_list, amount=self.amount, contraction=self.contraction, model=model, direction=self.direction)
			else:
				model = 'tridimensional'
				M0 = assemble_mass_matrix(diaphragms=diaphragms, model=model)
				K0 = assemble_total_stiffness_matrix(frames=frames, diaphragms=diaphragms)
				w_n , _ = calc_natural_frequencies(M0, K0)
				self.results = tuner(num_pav=numStories, frames=frames, M0=M0, configurations=self.configurations, w=w_n, massPercentage=self.massPercentage,
				tlcdType=self.tlcdType, pos=position_list, amount=self.amount, contraction=self.contraction, model=model, direction=self.direction)
		elif self.tlcdType == 'Pressurized TLCD':
			self.amount = int(get_text(self.amountPtlcdDialogLineEdit))
			self.position = get_text(self.positionPtlcdDialogLineEdit)
			if self.position == 'Last':
				position_list = [self.Input.frames[1].num_pav]
			else:
				position_list = re.split(',| ', self.position)
				position_list = [int(i) for i in position_list if i != '']
				if len(self.position) == 1:
					position_list = position_list[0]
			self.direction = get_text(self.directionPtlcdDialogComboBox)
			self.contraction = float(get_text(self.contractionPtlcdDialogLineEdit))
			self.massPercentage = float(get_text(self.massPercentageDialogLineEdit_2)) / 100
			self.D = float(get_text(self.diameterPtlcdDialogLineEdit)) / 100
			self.gasHeight = float(get_text(self.gasHeightPtlcdDialogLineEdit)) / 100
			if len(self.Input.frames.keys()) == 1:
				model = 'shear building'
				M0 = assemble_mass_matrix(diaphragms=diaphragms, model=model)
				K0 = assemble_lat_stiffness_matrix(frame=frames[1], diaphragms=diaphragms)
				w_n , _ = calc_natural_frequencies(M0, K0)
				self.results = tuner(num_pav=numStories, frames=frames[1], M0=M0, configurations=self.configurations, w=w_n, massPercentage=self.massPercentage,
				tlcdType=self.tlcdType, pos=self.position, amount=self.amount, D=self.D, Z=self.gasHeight, contraction=self.contraction, model=model, direction=self.direction)
			else:
				model = 'tridimensional'
				M0 = assemble_mass_matrix(diaphragms=diaphragms, model=model)
				K0 = assemble_total_stiffness_matrix(frames=frames, diaphragms=diaphragms)
				w_n , _ = calc_natural_frequencies(M0, K0)
				self.results = tuner(num_pav=numStories, frames=frames, M0=M0, configurations=self.configurations, w=w_n, massPercentage=self.massPercentage,
				tlcdType=self.tlcdType, pos=self.position, amount=self.amount, D=self.D, Z=self.gasHeight, contraction=self.contraction, model=model, direction=self.direction)

class BoundaryConditionsDialog(QDialog, Ui_BoundaryConditionsDialog):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setupUi(self)
		self.x0 = 0.
		self.v0 = 0.
		
		self.buttonBox.accepted.connect(self.setBoundaryConditions)

	def setBoundaryConditions(self):
		self.x0 = float(get_text(self.initialDisplacementLineEdit))
		self.v0 = float(get_text(self.initialVelocityLineEdit))

class FluidParametersDialog(QDialog, Ui_FluidParametersDialog):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setupUi(self)
		self.specificMass = 998.2071
		self.kineticViscosity = 1.003e-06

		self.buttonBox.accepted.connect(self.setFluidParameters)
	
	def setFluidParameters(self):
		self.specificMass = float(get_text(self.liquidSpecificMassLineEdit))
		self.kineticViscosity = float(get_text(self.liquidKineticViscosityLineEdit))
	
class DMFSettingsDialog(QDialog, Ui_DMFSettingsDialog):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setupUi(self)
		self.discretizationPoints = 200
		self.upperLimitFactor = 2

		self.buttonBox.accepted.connect(self.setDMFSettings)

	def setDMFSettings(self):
		self.discretizationPoints = float(get_text(self.discretizationPointsLineEdit))
		self.upperLimitFactor = float(get_text(self.upperLimitFactorLineEdit))

class StepSizeDialog(QDialog, Ui_StepSizeDialog):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setupUi(self)
		self.timeStep = 0.005

		self.buttonBox.accepted.connect(self.setStepSize)
	
	def setStepSize(self):
		self.timeStep = float(get_text(self.timeStepLineEdit))

class StructureDampingDialog(QDialog, Ui_StructureDampingDialog):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setupUi(self)
		self.dampingRatio = 0.02

		self.buttonBox.accepted.connect(self.setStructureDamping)
	
	def setStructureDamping(self):
		self.dampingRatio = float(get_text(self.structureDampingLineEdit))

def main():
	global app
	app = QApplication(sys.argv)
	win = DpMainWindow()
	win.show()
	sys.exit(app.exec_())

if __name__ == "__main__":
	main()
