import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton
from PyQt5.QtWidgets import QWidget, QLabel, QTextBrowser, QDoubleSpinBox, QPlainTextEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.optimize import curve_fit

# Função de ajuste


def fit_function(x, a1, a2, a3, Ea1, Ea2):
    KbT = 1.0
    return a1 * np.exp(-Ea1 / (KbT * x)) + a2 * np.exp(-Ea2 / (KbT * x)) + a3


class LabfitApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('LabFit - Photoluminescence')
        self.setGeometry(170, 10, 271, 41)
        self.data = None
        self.init_ui()

    def open_file_dialog(self):
        options = QFileDialog.Options()
        file_dialog = QFileDialog()
        selected_file, _ = file_dialog.getOpenFileName(
            self, 'Selecione um Arquivo', '', 'Todos os Arquivos (*);;Text Files (*.txt);;CSV Files (*.csv);;Output Files (*.out)', options=options)

        if selected_file:
            # Define o texto no campo de entrada
            self.plainTextEdit.setPlainText(selected_file)
            try:
                self.data = np.loadtxt(selected_file)
                self.textBrowser.clear()
                self.update_plot()
            except Exception as e:
                self.textBrowser.setPlainText(
                    f"Erro ao carregar arquivo: {str(e)}")
        else:
            self.data = np.genfromtxt(str(input()), delimiter=',')

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.setGeometry(100, 100, 791, 630)

        # Definindo geometria dos widgets
        self.textBrowser = QTextBrowser(self)
        self.textBrowser.setGeometry(170, 10, 271, 41)
        self.textBrowser.setHtml('<span style="font-size:18pt; font-weight:600; color:#910039;">LabFit - Photoluminescence PL</span>')

        file_label = QLabel('File path:', self)
        file_label.setGeometry(20, 530, 61, 31)

        self.plainTextEdit = QPlainTextEdit(self)
        self.plainTextEdit.setGeometry(90, 530, 400, 30)

        self.pushButton = QPushButton('Exit', self)
        self.pushButton.setGeometry(670, 550, 113, 32)
        self.pushButton.clicked.connect(self.close)

        labels = ['a1:', 'a2:', 'a3:', 'Ea1:', 'Ea2:']
        self.doubleSpinBoxes = []

        for i, label in enumerate(labels):
            spin_box = QDoubleSpinBox(self)
            spin_box.setGeometry(710, 180 + i * 40, 62, 22)
            label_widget = QLabel(label, self)
            label_widget.setGeometry(660, 180 + i * 40, 100, 30)
            self.doubleSpinBoxes.append(spin_box)
            # Conecta a alteração de valor ao método de atualização
            spin_box.valueChanged.connect(self.update_plot)

        self.pushButton_2 = QPushButton('Open File', self)
        self.pushButton_2.setGeometry(500, 530, 113, 32)
        self.pushButton_2.clicked.connect(self.open_file_dialog)

        # Criando a figura separadamente
        # O widget responsável por criar um quadrado para exibição do plot.
        # self.graphicsView = FigureCanvas(plt.figure(figsize=(8, 4)))
        # layout.addWidget(self.graphicsView)                             # Aqui esse widget é adicionado à janela (o layout).
        self.figure = plt.figure(figsize=(8, 6))
        self.graphicsView = FigureCanvas(self.figure)
        self.graphicsView.setGeometry(100, 100, 569, 439)

    def update_plot(self):
        if self.data is not None:
            x = np.arange(len(self.data))
            params = [spin_box.value() for spin_box in self.doubleSpinBoxes]
            y_fit = fit_function(x, *params)

            self.graphicsView.figure.clear()
            ax = self.graphicsView.figure.add_subplot(111)
            ax.plot(x, self.data, '-', label='Dados')
            ax.plot(x, y_fit, label=f'Ajuste')
            ax.set_xlabel('Índice')
            ax.set_ylabel('Valor')
            ax.legend()
            ax.set_title('Ajuste Polinomial')

            self.graphicsView.draw()


def main():
    app = QApplication(sys.argv)
    window = LabfitApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

