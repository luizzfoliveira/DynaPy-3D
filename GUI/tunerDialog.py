# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI/tunerDialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(330, 359)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.tlcdStackedWidget = QtWidgets.QStackedWidget(Dialog)
        self.tlcdStackedWidget.setObjectName("tlcdStackedWidget")
        self.page_8 = QtWidgets.QWidget()
        self.page_8.setObjectName("page_8")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.page_8)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.contractionTlcdDialogLineEdit = QtWidgets.QLineEdit(self.page_8)
        self.contractionTlcdDialogLineEdit.setObjectName("contractionTlcdDialogLineEdit")
        self.gridLayout_8.addWidget(self.contractionTlcdDialogLineEdit, 4, 1, 1, 1)
        self.label_29 = QtWidgets.QLabel(self.page_8)
        self.label_29.setObjectName("label_29")
        self.gridLayout_8.addWidget(self.label_29, 4, 0, 1, 1)
        self.amountTlcdDialogLineEdit = QtWidgets.QLineEdit(self.page_8)
        self.amountTlcdDialogLineEdit.setObjectName("amountTlcdDialogLineEdit")
        self.gridLayout_8.addWidget(self.amountTlcdDialogLineEdit, 0, 1, 1, 1)
        self.directionTlcdDialogComboBox = QtWidgets.QComboBox(self.page_8)
        self.directionTlcdDialogComboBox.setObjectName("directionTlcdDialogComboBox")
        self.directionTlcdDialogComboBox.addItem("")
        self.directionTlcdDialogComboBox.addItem("")
        self.gridLayout_8.addWidget(self.directionTlcdDialogComboBox, 3, 1, 1, 1)
        self.label_30 = QtWidgets.QLabel(self.page_8)
        self.label_30.setObjectName("label_30")
        self.gridLayout_8.addWidget(self.label_30, 0, 0, 1, 1)
        self.label_33 = QtWidgets.QLabel(self.page_8)
        self.label_33.setObjectName("label_33")
        self.gridLayout_8.addWidget(self.label_33, 2, 0, 1, 1)
        self.label_35 = QtWidgets.QLabel(self.page_8)
        self.label_35.setObjectName("label_35")
        self.gridLayout_8.addWidget(self.label_35, 3, 0, 1, 1)
        self.positionTlcdDialogLineEdit = QtWidgets.QLineEdit(self.page_8)
        self.positionTlcdDialogLineEdit.setText("")
        self.positionTlcdDialogLineEdit.setObjectName("positionTlcdDialogLineEdit")
        self.gridLayout_8.addWidget(self.positionTlcdDialogLineEdit, 2, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.page_8)
        self.label_2.setObjectName("label_2")
        self.gridLayout_8.addWidget(self.label_2, 5, 0, 1, 1)
        self.massPercentageDialogLineEdit = QtWidgets.QLineEdit(self.page_8)
        self.massPercentageDialogLineEdit.setObjectName("massPercentageDialogLineEdit")
        self.gridLayout_8.addWidget(self.massPercentageDialogLineEdit, 5, 1, 1, 1)
        self.tlcdStackedWidget.addWidget(self.page_8)
        self.page_4 = QtWidgets.QWidget()
        self.page_4.setObjectName("page_4")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.page_4)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_42 = QtWidgets.QLabel(self.page_4)
        self.label_42.setObjectName("label_42")
        self.gridLayout_3.addWidget(self.label_42, 4, 0, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.page_4)
        self.label_18.setObjectName("label_18")
        self.gridLayout_3.addWidget(self.label_18, 3, 0, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.page_4)
        self.label_20.setObjectName("label_20")
        self.gridLayout_3.addWidget(self.label_20, 0, 0, 1, 1)
        self.positionPtlcdDialogLineEdit = QtWidgets.QLineEdit(self.page_4)
        self.positionPtlcdDialogLineEdit.setObjectName("positionPtlcdDialogLineEdit")
        self.gridLayout_3.addWidget(self.positionPtlcdDialogLineEdit, 3, 2, 1, 1)
        self.label_37 = QtWidgets.QLabel(self.page_4)
        self.label_37.setObjectName("label_37")
        self.gridLayout_3.addWidget(self.label_37, 5, 0, 1, 1)
        self.gasHeightPtlcdDialogLineEdit = QtWidgets.QLineEdit(self.page_4)
        self.gasHeightPtlcdDialogLineEdit.setObjectName("gasHeightPtlcdDialogLineEdit")
        self.gridLayout_3.addWidget(self.gasHeightPtlcdDialogLineEdit, 1, 2, 1, 1)
        self.directionPtlcdDialogComboBox = QtWidgets.QComboBox(self.page_4)
        self.directionPtlcdDialogComboBox.setObjectName("directionPtlcdDialogComboBox")
        self.directionPtlcdDialogComboBox.addItem("")
        self.directionPtlcdDialogComboBox.addItem("")
        self.gridLayout_3.addWidget(self.directionPtlcdDialogComboBox, 4, 2, 1, 1)
        self.label_36 = QtWidgets.QLabel(self.page_4)
        self.label_36.setObjectName("label_36")
        self.gridLayout_3.addWidget(self.label_36, 2, 0, 1, 1)
        self.contractionPtlcdDialogLineEdit = QtWidgets.QLineEdit(self.page_4)
        self.contractionPtlcdDialogLineEdit.setObjectName("contractionPtlcdDialogLineEdit")
        self.gridLayout_3.addWidget(self.contractionPtlcdDialogLineEdit, 5, 2, 1, 1)
        self.label_40 = QtWidgets.QLabel(self.page_4)
        self.label_40.setObjectName("label_40")
        self.gridLayout_3.addWidget(self.label_40, 1, 0, 1, 2)
        self.diameterPtlcdDialogLineEdit = QtWidgets.QLineEdit(self.page_4)
        self.diameterPtlcdDialogLineEdit.setEnabled(True)
        self.diameterPtlcdDialogLineEdit.setObjectName("diameterPtlcdDialogLineEdit")
        self.gridLayout_3.addWidget(self.diameterPtlcdDialogLineEdit, 0, 2, 1, 1)
        self.amountPtlcdDialogLineEdit = QtWidgets.QLineEdit(self.page_4)
        self.amountPtlcdDialogLineEdit.setObjectName("amountPtlcdDialogLineEdit")
        self.gridLayout_3.addWidget(self.amountPtlcdDialogLineEdit, 2, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.page_4)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 6, 0, 1, 1)
        self.massPercentageDialogLineEdit_2 = QtWidgets.QLineEdit(self.page_4)
        self.massPercentageDialogLineEdit_2.setObjectName("massPercentageDialogLineEdit_2")
        self.gridLayout_3.addWidget(self.massPercentageDialogLineEdit_2, 6, 2, 1, 1)
        self.tlcdStackedWidget.addWidget(self.page_4)
        self.gridLayout.addWidget(self.tlcdStackedWidget, 1, 0, 1, 2)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.tlcdTypeDialogComboBox = QtWidgets.QComboBox(Dialog)
        self.tlcdTypeDialogComboBox.setObjectName("tlcdTypeDialogComboBox")
        self.tlcdTypeDialogComboBox.addItem("")
        self.tlcdTypeDialogComboBox.addItem("")
        self.gridLayout.addWidget(self.tlcdTypeDialogComboBox, 0, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Dialog)
        self.tlcdStackedWidget.setCurrentIndex(0)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "TLCD Optimization"))
        self.contractionTlcdDialogLineEdit.setPlaceholderText(_translate("Dialog", "1"))
        self.label_29.setText(_translate("Dialog", "Coef. of Contraction:"))
        self.amountTlcdDialogLineEdit.setPlaceholderText(_translate("Dialog", "1"))
        self.directionTlcdDialogComboBox.setItemText(0, _translate("Dialog", "x"))
        self.directionTlcdDialogComboBox.setItemText(1, _translate("Dialog", "y"))
        self.label_30.setText(_translate("Dialog", "Amount:"))
        self.label_33.setText(_translate("Dialog", "Position:"))
        self.label_35.setText(_translate("Dialog", "Direction:"))
        self.positionTlcdDialogLineEdit.setPlaceholderText(_translate("Dialog", "Last"))
        self.label_2.setText(_translate("Dialog", "Mass Percentage: (%)"))
        self.massPercentageDialogLineEdit.setPlaceholderText(_translate("Dialog", "2"))
        self.label_42.setText(_translate("Dialog", "Direction"))
        self.label_18.setText(_translate("Dialog", "Position"))
        self.label_20.setText(_translate("Dialog", "Diameter: (cm)"))
        self.positionPtlcdDialogLineEdit.setPlaceholderText(_translate("Dialog", "Last"))
        self.label_37.setText(_translate("Dialog", "Coef. of Contraction"))
        self.gasHeightPtlcdDialogLineEdit.setPlaceholderText(_translate("Dialog", "40"))
        self.directionPtlcdDialogComboBox.setItemText(0, _translate("Dialog", "x"))
        self.directionPtlcdDialogComboBox.setItemText(1, _translate("Dialog", "y"))
        self.label_36.setText(_translate("Dialog", "Amount"))
        self.contractionPtlcdDialogLineEdit.setPlaceholderText(_translate("Dialog", "1"))
        self.label_40.setText(_translate("Dialog", "Gas Height(cm)"))
        self.diameterPtlcdDialogLineEdit.setPlaceholderText(_translate("Dialog", "30"))
        self.amountPtlcdDialogLineEdit.setPlaceholderText(_translate("Dialog", "1"))
        self.label_3.setText(_translate("Dialog", "Mass Percentage: (%)"))
        self.massPercentageDialogLineEdit_2.setPlaceholderText(_translate("Dialog", "2"))
        self.label.setText(_translate("Dialog", "TLCD Type"))
        self.tlcdTypeDialogComboBox.setItemText(0, _translate("Dialog", "Basic"))
        self.tlcdTypeDialogComboBox.setItemText(1, _translate("Dialog", "Pressurized"))