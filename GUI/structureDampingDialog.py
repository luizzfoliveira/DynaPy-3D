# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI/structureDampingDialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_StructureDampingDialog(object):
    def setupUi(self, StructureDampingDialog):
        StructureDampingDialog.setObjectName("StructureDampingDialog")
        StructureDampingDialog.resize(250, 10)
        self.verticalLayout = QtWidgets.QVBoxLayout(StructureDampingDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(StructureDampingDialog)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.structureDampingLineEdit = QtWidgets.QLineEdit(StructureDampingDialog)
        self.structureDampingLineEdit.setObjectName("structureDampingLineEdit")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.structureDampingLineEdit)
        self.verticalLayout.addLayout(self.formLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(StructureDampingDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(StructureDampingDialog)
        self.buttonBox.accepted.connect(StructureDampingDialog.accept)
        self.buttonBox.rejected.connect(StructureDampingDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(StructureDampingDialog)

    def retranslateUi(self, StructureDampingDialog):
        _translate = QtCore.QCoreApplication.translate
        StructureDampingDialog.setWindowTitle(_translate("StructureDampingDialog", "Structure Damping"))
        self.label.setText(_translate("StructureDampingDialog", "Structure Damping Ratio:"))
        self.structureDampingLineEdit.setPlaceholderText(_translate("StructureDampingDialog", "0.02"))