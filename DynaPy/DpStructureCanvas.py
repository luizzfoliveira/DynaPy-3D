from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class StructureCanvas(QGraphicsView):
	def __init__(self, parent):
		super(StructureCanvas, self).__init__(parent)

		# self.setGeometry(0, 0, 200, 300)
		self.scene1 = QGraphicsScene()
		self.setScene(self.scene1)

		self.blackColor = QColor(0, 0, 0, 255)
		self.whiteColor = QColor(255, 255, 255, 255)

		self.pen4 = QPen()
		self.pen4.setWidth(4)
		self.pen4.setColor(self.blackColor)

		self.brush = QBrush()
		self.brush.setColor(self.blackColor)
		self.brush.setStyle(Qt.SolidPattern)

		self.brush2 = QBrush()
		self.brush2.setColor(self.whiteColor)
		self.brush2.setStyle(Qt.SolidPattern)

		self.font40 = QFont("Times", 40)

		# self.painter2()

	def painter(self, frame, diaphragms, frameNum):
		self.scene1 = QGraphicsScene()
		self.setScene(self.scene1)
		if (frame == None):
			return
		frameNumText = QGraphicsTextItem("Frame {}\nDirection {}\nDistance To Center: {}".format(frameNum, frame.direction, frame.dist))
		frameNumText.setFont(self.font40)
		frameNumText.setPos(-300, 0)
		
		self.scene1.addItem(frameNumText)
		level = 0
		b = 300 * frame.num_frame
		h = 20
		for i in range(len(frame.height)):
			height = frame.height[i] * 100
			beam = QGraphicsRectItem(0, -(level+height),
									 b, -h)
			beam.setPen(self.pen4)
			beam.setBrush(self.brush)

			columns = []
			for j in range(frame.num_frame + 1):
				columns.append(QGraphicsLineItem(j * b / frame.num_frame, -level,
												 j * b / frame.num_frame, -(level+height)))
				columns[j].setPen(self.pen4)

			storyNum = QGraphicsTextItem('{}'.format(i + 1))
			storyNum.setFont(self.font40)
			storyNum.setTextWidth(storyNum.boundingRect().width())
			storyNum.setPos(b/2-storyNum.textWidth()/2, -(level+height/2)-30)
			numCircle = QGraphicsEllipseItem(b/2-storyNum.textWidth()/2-20,
											 -(level+height/2)-30,
											 storyNum.boundingRect().width()+40,
											 80)

			storyMass = QGraphicsTextItem('{} ton'.format(diaphragms.mass[i] / 1000))
			storyMass.setFont(self.font40)
			storyMass.setTextWidth(storyMass.boundingRect().width())
			# storyMass.setPos(b+20, -(level+height+40))
			storyMass.setPos(b/2-storyMass.textWidth()/2, -(level+height-10))


			storyHeight = QGraphicsSimpleTextItem('{} m'.format(frame.height[i]))
			storyHeight.setPos(b+20, -(level+height/2)-65)
			storyHeight.setFont(self.font40)

			storySection = QGraphicsSimpleTextItem('({} x {}) cm'.format(frame.width * 100, frame.depth * 100))
			storySection.setPos(b+20, -(level+height/2)-10)
			storySection.setFont(self.font40)

			storyE = QGraphicsTextItem('{} GPa'.format(frame.E / 1e9))
			storyE.setFont(self.font40)
			storyE.setTextWidth(storyE.boundingRect().width())
			storyE.setPos(-(storyE.textWidth()+20), -(level+height/2)-65)

			if frame.support == 'Fix-Fix':
				support = '(F-F)'
			elif frame.support == 'Fix-Pin':
				support = '(F-P)'
			elif frame.support == 'Pin-Fix':
				support = '(P-F)'
			elif frame.support == 'Pin-Pin':
				support = '(P-P)'

			storySupport = QGraphicsTextItem('{}'.format(support))
			storySupport.setFont(self.font40)
			storySupport.setTextWidth(storySupport.boundingRect().width())
			storySupport.setPos(-(storySupport.textWidth()+20), -(level+height/2)-10)

			self.scene1.addItem(beam)
			for j in range(len(columns)):
				self.scene1.addItem(columns[j])
			""" self.scene1.addItem(column1)
			self.scene1.addItem(column2) """
			self.scene1.addItem(storyNum)
			self.scene1.addItem(storyMass)
			self.scene1.addItem(storyHeight)
			self.scene1.addItem(storySection)
			self.scene1.addItem(storyE)
			self.scene1.addItem(storySupport)
			self.scene1.addItem(numCircle)

			if i == 0:
				for j in range(len(columns)):
					if support == '(F-F)':
						l1e = QGraphicsLineItem(-30, 0,
												30, 0)
						l2e = QGraphicsLineItem(-40, 10,
												-30, 0)
						l3e = QGraphicsLineItem(-25, 10,
												-15, 0)
						l4e = QGraphicsLineItem(-10, 10,
												0, 0)
						l5e = QGraphicsLineItem(5, 10,
												15, 0)
						l6e = QGraphicsLineItem(20, 10,
												30, 0)
						l1d = QGraphicsLineItem(-30 + j * b / frame.num_frame, 0,
												30 + j * b / frame.num_frame, 0)
						l2d = QGraphicsLineItem(-40 + j * b / frame.num_frame, 10,
												-30 + j * b / frame.num_frame, 0)
						l3d = QGraphicsLineItem(-25 + j * b / frame.num_frame, 10,
												-15 + j * b / frame.num_frame, 0)
						l4d = QGraphicsLineItem(-10 + j * b / frame.num_frame, 10,
												0 + j * b / frame.num_frame, 0)
						l5d = QGraphicsLineItem(5 + j * b / frame.num_frame, 10,
												15 + j * b / frame.num_frame, 0)
						l6d = QGraphicsLineItem(20 + j * b / frame.num_frame, 10,
												30 + j * b / frame.num_frame, 0)

						l1e.setPen(self.pen4)
						l2e.setPen(self.pen4)
						l3e.setPen(self.pen4)
						l4e.setPen(self.pen4)
						l5e.setPen(self.pen4)
						l6e.setPen(self.pen4)
						l1d.setPen(self.pen4)
						l2d.setPen(self.pen4)
						l3d.setPen(self.pen4)
						l4d.setPen(self.pen4)
						l5d.setPen(self.pen4)
						l6d.setPen(self.pen4)

						self.scene1.addItem(l1e)
						self.scene1.addItem(l2e)
						self.scene1.addItem(l3e)
						self.scene1.addItem(l4e)
						self.scene1.addItem(l5e)
						self.scene1.addItem(l6e)
						self.scene1.addItem(l1d)
						self.scene1.addItem(l2d)
						self.scene1.addItem(l3d)
						self.scene1.addItem(l4d)
						self.scene1.addItem(l5d)
						self.scene1.addItem(l6d)

					elif support == '(F-P)':
						l1e = QGraphicsLineItem(-30, 0,
												30, 0)
						l2e = QGraphicsLineItem(-40, 10,
												-30, 0)
						l3e = QGraphicsLineItem(-25, 10,
												-15, 0)
						l4e = QGraphicsLineItem(-10, 10,
												0, 0)
						l5e = QGraphicsLineItem(5, 10,
												15, 0)
						l6e = QGraphicsLineItem(20, 10,
												30, 0)
						l1d = QGraphicsLineItem(-45 + j * b / frame.num_frame, 40,
												45 + j * b / frame.num_frame, 40)
						l2d = QGraphicsLineItem(-40 + j * b / frame.num_frame, 50,
												-30 + j * b / frame.num_frame, 40)
						l3d = QGraphicsLineItem(-25 + j * b / frame.num_frame, 50,
												-15 + j * b / frame.num_frame, 40)
						l4d = QGraphicsLineItem(-10 + j * b / frame.num_frame, 50,
												0 + j * b / frame.num_frame, 40)
						l5d = QGraphicsLineItem(5 + j * b / frame.num_frame, 50,
												15 + j * b / frame.num_frame, 40)
						l6d = QGraphicsLineItem(20 + j * b / frame.num_frame, 50,
												30 + j * b / frame.num_frame, 40)
						l7d = QGraphicsLineItem(-30 + j * b / frame.num_frame, 40,
												0 + j * b / frame.num_frame, 0)
						l8d = QGraphicsLineItem(30 + j * b / frame.num_frame, 40,
												0 + j * b / frame.num_frame, 0)

						l1e.setPen(self.pen4)
						l2e.setPen(self.pen4)
						l3e.setPen(self.pen4)
						l4e.setPen(self.pen4)
						l5e.setPen(self.pen4)
						l6e.setPen(self.pen4)
						l1d.setPen(self.pen4)
						l2d.setPen(self.pen4)
						l3d.setPen(self.pen4)
						l4d.setPen(self.pen4)
						l5d.setPen(self.pen4)
						l6d.setPen(self.pen4)
						l7d.setPen(self.pen4)
						l8d.setPen(self.pen4)

						self.scene1.addItem(l1e)
						self.scene1.addItem(l2e)
						self.scene1.addItem(l3e)
						self.scene1.addItem(l4e)
						self.scene1.addItem(l5e)
						self.scene1.addItem(l6e)
						self.scene1.addItem(l1d)
						self.scene1.addItem(l2d)
						self.scene1.addItem(l3d)
						self.scene1.addItem(l4d)
						self.scene1.addItem(l5d)
						self.scene1.addItem(l6d)
						self.scene1.addItem(l7d)
						self.scene1.addItem(l8d)

					elif support == '(P-F)':
						l1e = QGraphicsLineItem(-45, 40,
												45, 40)
						l2e = QGraphicsLineItem(-40, 50,
												-30, 40)
						l3e = QGraphicsLineItem(-25, 50,
												-15, 40)
						l4e = QGraphicsLineItem(-10, 50,
												0, 40)
						l5e = QGraphicsLineItem(5, 50,
												15, 40)
						l6e = QGraphicsLineItem(20, 50,
												30, 40)
						l7e = QGraphicsLineItem(-30, 40,
												0, 0)
						l8e = QGraphicsLineItem(30, 40,
												0, 0)
						l1d = QGraphicsLineItem(-30 + j * b / frame.num_frame, 0,
												30 + j * b / frame.num_frame, 0)
						l2d = QGraphicsLineItem(-40 + j * b / frame.num_frame, 10,
												-30 + j * b / frame.num_frame, 0)
						l3d = QGraphicsLineItem(-25 + j * b / frame.num_frame, 10,
												-15 + j * b / frame.num_frame, 0)
						l4d = QGraphicsLineItem(-10 + j * b / frame.num_frame, 10,
												0 + j * b / frame.num_frame, 0)
						l5d = QGraphicsLineItem(5 + j * b / frame.num_frame, 10,
												15 + j * b / frame.num_frame, 0)
						l6d = QGraphicsLineItem(20 + j * b / frame.num_frame, 10,
												30 + j * b / frame.num_frame, 0)

						l1e.setPen(self.pen4)
						l2e.setPen(self.pen4)
						l3e.setPen(self.pen4)
						l4e.setPen(self.pen4)
						l5e.setPen(self.pen4)
						l6e.setPen(self.pen4)
						l7e.setPen(self.pen4)
						l8e.setPen(self.pen4)
						l1d.setPen(self.pen4)
						l2d.setPen(self.pen4)
						l3d.setPen(self.pen4)
						l4d.setPen(self.pen4)
						l5d.setPen(self.pen4)
						l6d.setPen(self.pen4)

						self.scene1.addItem(l1e)
						self.scene1.addItem(l2e)
						self.scene1.addItem(l3e)
						self.scene1.addItem(l4e)
						self.scene1.addItem(l5e)
						self.scene1.addItem(l6e)
						self.scene1.addItem(l7e)
						self.scene1.addItem(l8e)
						self.scene1.addItem(l1d)
						self.scene1.addItem(l2d)
						self.scene1.addItem(l3d)
						self.scene1.addItem(l4d)
						self.scene1.addItem(l5d)
						self.scene1.addItem(l6d)

					elif support == '(P-P)':
						l1e = QGraphicsLineItem(-45, 40,
												45, 40)
						l2e = QGraphicsLineItem(-40, 50,
												-30, 40)
						l3e = QGraphicsLineItem(-25, 50,
												-15, 40)
						l4e = QGraphicsLineItem(-10, 50,
												0, 40)
						l5e = QGraphicsLineItem(5, 50,
												15, 40)
						l6e = QGraphicsLineItem(20, 50,
												30, 40)
						l7e = QGraphicsLineItem(-30, 40,
												0, 0)
						l8e = QGraphicsLineItem(30, 40,
												0, 0)
						l1d = QGraphicsLineItem(-45 + j * b / frame.num_frame, 40,
												45 + j * b / frame.num_frame, 40)
						l2d = QGraphicsLineItem(-40 + j * b / frame.num_frame, 50,
												-30 + j * b / frame.num_frame, 40)
						l3d = QGraphicsLineItem(-25 + j * b / frame.num_frame, 50,
												-15 + j * b / frame.num_frame, 40)
						l4d = QGraphicsLineItem(-10 + j * b / frame.num_frame, 50,
												0 + j * b / frame.num_frame, 40)
						l5d = QGraphicsLineItem(5 + j * b / frame.num_frame, 50,
												15 + j * b / frame.num_frame, 40)
						l6d = QGraphicsLineItem(20 + j * b / frame.num_frame, 50,
												30 + j * b / frame.num_frame, 40)
						l7d = QGraphicsLineItem(-30 + j * b / frame.num_frame, 40,
												0 + j * b / frame.num_frame, 0)
						l8d = QGraphicsLineItem(30 + j * b / frame.num_frame, 40,
												0 + j * b / frame.num_frame, 0)

						l1e.setPen(self.pen4)
						l2e.setPen(self.pen4)
						l3e.setPen(self.pen4)
						l4e.setPen(self.pen4)
						l5e.setPen(self.pen4)
						l6e.setPen(self.pen4)
						l7e.setPen(self.pen4)
						l8e.setPen(self.pen4)
						l1d.setPen(self.pen4)
						l2d.setPen(self.pen4)
						l3d.setPen(self.pen4)
						l4d.setPen(self.pen4)
						l5d.setPen(self.pen4)
						l6d.setPen(self.pen4)
						l7d.setPen(self.pen4)
						l8d.setPen(self.pen4)

						self.scene1.addItem(l1e)
						self.scene1.addItem(l2e)
						self.scene1.addItem(l3e)
						self.scene1.addItem(l4e)
						self.scene1.addItem(l5e)
						self.scene1.addItem(l6e)
						self.scene1.addItem(l7e)
						self.scene1.addItem(l8e)
						self.scene1.addItem(l1d)
						self.scene1.addItem(l2d)
						self.scene1.addItem(l3d)
						self.scene1.addItem(l4d)
						self.scene1.addItem(l5d)
						self.scene1.addItem(l6d)
						self.scene1.addItem(l7d)
						self.scene1.addItem(l8d)

				else:
					if support == '(F-F)':
						pass
					if support == '(F-P)':
						cd = QGraphicsEllipseItem(j * b / frame.num_frame-10, -(level+20+h), 20, 20)
						cd.setBrush(self.brush2)
						self.scene1.addItem(cd)
					if support == '(P-F)':
						ce = QGraphicsEllipseItem(-10, -(level+20+h), 20, 20)
						ce.setBrush(self.brush2)
						self.scene1.addItem(ce)
					if support == '(P-P)':
						ce = QGraphicsEllipseItem(-10, -(level+20+h), 20, 20)
						ce.setBrush(self.brush2)
						self.scene1.addItem(ce)
						cd = QGraphicsEllipseItem(j * b / frame.num_frame-10, -(level+20+h), 20, 20)
						cd.setBrush(self.brush2)
						self.scene1.addItem(cd)

			level += frame.height[i] * 100

		# self.setGeometry(0, 0, self.sizeHint().width(), self.sizeHint().height())
		self.setViewportMargins(10, 10, 10, 10)
		self.fitInView(self.scene1.itemsBoundingRect(), Qt.KeepAspectRatio)
