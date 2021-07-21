from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class PlanCanvas(QGraphicsView):
	def __init__(self, parent):
		super(PlanCanvas, self).__init__(parent)

		self.frame_list = []
		self.frameNums = []

		# self.setGeometry(0, 0, 200, 300)
		self.scene1 = QGraphicsScene()
		self.setScene(self.scene1)

		self.blackColor = QColor(0, 0, 0, 255)
		self.whiteColor = QColor(255, 255, 255, 255)

		self.pen4 = QPen()
		self.pen4.setWidth(4)
		self.pen4.setColor(self.blackColor)

		self.pen6 = QPen()
		self.pen6.setWidth(6)
		self.pen6.setColor(self.blackColor)

		self.brush = QBrush()
		self.brush.setColor(self.blackColor)
		self.brush.setStyle(Qt.SolidPattern)

		self.brush2 = QBrush()
		self.brush2.setColor(self.whiteColor)
		self.brush2.setStyle(Qt.SolidPattern)

		self.font40 = QFont("Times", 40)

		self.font20 = QFont("Times", 20)

		# self.painter2()

	def firstPainter(self, diaphragms):
		self.scene1 = QGraphicsScene()
		self.setScene(self.scene1)
		if diaphragms == None:
			return
		level = 0
		w = 600
		d = 600
		diaphragm = QGraphicsRectItem(0, level, w, d)
		diaphragm.setPen(self.pen4)
		width = diaphragms.width
		depth = diaphragms.depth

		# Diaphragms Dimensions
		w_dimension = QGraphicsTextItem("{} m".format(width))
		w_dimension.setFont(self.font40)
		w_dimension.setPos(-150, 260)
		d_dimension = QGraphicsTextItem("{} m".format(depth))
		d_dimension.setFont(self.font40)
		d_dimension.setPos(260, -80)

		# Center
		x0 = QGraphicsLineItem(300, 300, 350, 300)
		x0.setPen(self.pen6)
		x0_arrowUp = QGraphicsLineItem(350, 300, 340, 290)
		x0_arrowUp.setPen(self.pen6)
		x0_arrowDown = QGraphicsLineItem(350, 300, 340, 310)
		x0_arrowDown.setPen(self.pen6)
		y0 = QGraphicsLineItem(300, 300, 300, 250)
		y0.setPen(self.pen6)
		y0_arrowLeft = QGraphicsLineItem(300, 250, 290, 260)
		y0_arrowLeft.setPen(self.pen6)
		y0_arrowRight = QGraphicsLineItem(300, 250, 310, 260)
		y0_arrowRight.setPen(self.pen6)
		arc = QGraphicsEllipseItem()
		arc.setRect(275, 275, 50, 50)
		arc.setStartAngle(90 * 16)
		arc.setSpanAngle(270 * 16)
		arc.setPen(self.pen6)
		arc_arrowLeft = QGraphicsLineItem(325, 300, 315, 310)
		arc_arrowLeft.setPen(self.pen6)
		arc_arrowRight = QGraphicsLineItem(325, 300, 335, 310)
		arc_arrowRight.setPen(self.pen6)

		xText = QGraphicsTextItem('x')
		xText.setFont(self.font20)
		xText.setPos(350, 300)
		yText = QGraphicsTextItem('y')
		yText.setFont(self.font20)
		yText.setPos(300, 200)
		thetaText = QGraphicsTextItem()
		thetaText.setHtml("&theta")
		thetaText.setFont(self.font20)
		thetaText.setPos(300, 325)

		self.scene1.addItem(x0)
		self.scene1.addItem(xText)
		self.scene1.addItem(x0_arrowUp)
		self.scene1.addItem(x0_arrowDown)
		self.scene1.addItem(y0)
		self.scene1.addItem(yText)
		self.scene1.addItem(y0_arrowLeft)
		self.scene1.addItem(y0_arrowRight)
		self.scene1.addItem(arc)
		self.scene1.addItem(thetaText)
		self.scene1.addItem(arc_arrowLeft)
		self.scene1.addItem(arc_arrowRight)
		self.scene1.addItem(diaphragm)
		self.scene1.addItem(w_dimension)
		self.scene1.addItem(d_dimension)
		self.setViewportMargins(10, 10, 10, 10)
		self.fitInView(self.scene1.itemsBoundingRect(), Qt.KeepAspectRatio)

	def painter(self, frames):
		for i, j in zip(self.frame_list, self.frameNums):
			self.scene1.removeItem(i)
			self.scene1.removeItem(j)
		x0 = 300
		y0 = 300
		width = 600
		depth = 600
		biggestDistance = 0
		for i in range(1, len(frames) + 1):
			if abs(frames[i].dist) > biggestDistance:
				biggestDistance = abs(frames[i].dist)
		if biggestDistance == 0:
			biggestDistance = 1

		self.frame_list = []
		self.frameNums = []
		for i in range(1, len(frames) + 1):
			width = 400
			depth = 5
			d = frames[i].dist
			if frames[i].direction == 'x':
				self.frame_list.append(QGraphicsRectItem(x0 - width / 2, y0 - d / biggestDistance * 235 + depth / 2, width, depth))
				self.frame_list[i - 1].setPen(self.pen4)
				self.frame_list[i - 1].setBrush(self.brush)
				self.frameNums.append(QGraphicsTextItem('{}'.format(i)))
				self.frameNums[i - 1].setFont(self.font40)
				self.frameNums[i - 1].setPos(x0 - width / 2, y0 - d / biggestDistance * 235 + depth / 2)
			else:
				self.frame_list.append(QGraphicsRectItem(x0 + d / biggestDistance * 235 - depth / 2, y0 - width / 2, depth, width))
				self.frame_list[i - 1].setPen(self.pen4)
				self.frame_list[i - 1].setBrush(self.brush)
				self.frameNums.append(QGraphicsTextItem('{}'.format(i)))
				self.frameNums[i - 1].setFont(self.font40)
				self.frameNums[i - 1].setPos(x0 + d / biggestDistance * 235 - depth / 2 - 60, y0 - width / 2)
	
			self.scene1.addItem(self.frame_list[i - 1])
			self.scene1.addItem(self.frameNums[i - 1])

		self.setViewportMargins(10, 10, 10, 10)
		self.fitInView(self.scene1.itemsBoundingRect(), Qt.KeepAspectRatio)
		