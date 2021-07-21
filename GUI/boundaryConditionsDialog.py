# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI/boundaryConditionsDialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_BoundaryConditionsDialog(object):
    def setupUi(self, BoundaryConditionsDialog):
        BoundaryConditionsDialog.setObjectName("BoundaryConditionsDialog")
        BoundaryConditionsDialog.resize(250, 10)
        self.verticalLayout = QtWidgets.QVBoxLayout(BoundaryConditionsDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(BoundaryConditionsDialog)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.initialDisplacementLineEdit = QtWidgets.QLineEdit(BoundaryConditionsDialog)
        self.initialDisplacementLineEdit.setObjectName("initialDisplacementLineEdit")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.initialDisplacementLineEdit)
        self.label_2 = QtWidgets.QLabel(BoundaryConditionsDialog)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.initialVelocityLineEdit = QtWidgets.QLineEdit(BoundaryConditionsDialog)
        self.initialVelocityLineEdit.setObjectName("initialVelocityLineEdit")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.initialVelocityLineEdit)
        self.verticalLayout.addLayout(self.formLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(BoundaryConditionsDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(BoundaryConditionsDialog)
        self.buttonBox.accepted.connect(BoundaryConditionsDialog.accept)
        self.buttonBox.rejected.connect(BoundaryConditionsDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(BoundaryConditionsDialog)

    def retranslateUi(self, BoundaryConditionsDialog):
        _translate = QtCore.QCoreApplication.translate
        BoundaryConditionsDialog.setWindowTitle(_translate("BoundaryConditionsDialog", "Boundary Conditions"))
        self.label.setText(_translate("BoundaryConditionsDialog", "Initial Displacement: (m)"))
        self.initialDisplacementLineEdit.setPlaceholderText(_translate("BoundaryConditionsDialog", "0.0"))
        self.label_2.setText(_translate("BoundaryConditionsDialog", "Initial Velocity: (m/s)"))
        self.initialVelocityLineEdit.setPlaceholderText(_translate("BoundaryConditionsDialog", "0.0"))
