from data_processing_binary_for_outliers import y_data as y_bin
from data_processing_binary_for_outliers import x_data as x_bin
from yellowbrick.regressor import CooksDistance
import scipy as sp
from yellowbrick.datasets import load_concrete
import numpy as np
from collections import Counter
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

class CooksDistanceMEU(CooksDistance):
    def fit(self, X, y, y_bin):

        self._model.fit(X, y)

        leverage = (X * np.linalg.pinv(X).T).sum(1)

        rank = np.linalg.matrix_rank(X)
        df = X.shape[0] - rank

        residuals = y - self._model.predict(X)
        mse = np.dot(residuals, residuals) / df

        residuals_studentized = residuals / np.sqrt(mse) / np.sqrt(1 - leverage)
        self.distance_ = residuals_studentized ** 2 / X.shape[1]
        self.distance_ *= leverage / (1 - leverage)

        self.p_values_ = sp.stats.f.sf(self.distance_, X.shape[1], df)

        self.influence_threshold_ = 4 / X.shape[0]
        self.outlier_percentage_ = (
            sum(self.distance_ > self.influence_threshold_) / X.shape[0]
        )
        self.outlier_percentage_ *= 100.0

        self.draw(y_bin)
        return self

    def draw(self, y_bin):

        for i in range(len(y_bin)):
            linefmt = 'olive' if y_bin[i]=='BTAL' else 'teal'

            _, _, baseline = self.ax.stem([i],
                [self.distance_[i]], linefmt=linefmt, markerfmt=self.markerfmt,
                use_line_collection=True
            )

        self.ax.set_xlim(0, len(self.distance_))

        if self.draw_threshold:
            label = r"{:0.2f}% > $I_t$ ($I_t=\frac {{4}} {{n}}$)".format(
                self.outlier_percentage_
            )
            self.ax.axhline(
                self.influence_threshold_,
                ls="--",
                label=label,
                c=baseline.get_color(),
                lw=baseline.get_linewidth(),
            )

        return self.ax

    def finalize(self):

        self.set_title("Cook's Distance Outlier Detection")
        self.ax.set_xlabel("instance index")
        self.ax.set_ylabel("influence (I)")

        if self.draw_threshold:
            self.ax.legend(loc=9, frameon=True)


def out_bin(x_bin, y_bin):
    dict = {"BTAL": 0, "AF": 2}
    y = y_bin.copy()
    y_bin = np.array([dict[disease] for disease in y_bin])

    # Instantiate and fit the visualizer
    visualizer = CooksDistanceMEU(title='Cook\'s Distance outlier detection for the β-thalassemia and IDA data')
    visualizer.fit(x_bin.astype(float), y_bin.astype(float), y)

    cooks_distance = visualizer.distance_
    outliers = cooks_distance[
        cooks_distance > visualizer.influence_threshold_].index  # visualizer.influence_threshold_ = 4/len(y_data)

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'Calibri'
    with PdfPages('test.pdf') as pdf:
        pdf.savefig()

    print(outliers)
    return visualizer, outliers

def bin_pie(outlier_binary, y_bin):

    labels = 'β-talassemia', 'IDA'
    out = y_bin[outlier_binary]
    out_disease = Counter(out)
    percentage = []
    for disease in out_disease.keys():
        percentage.append((out_disease[disease]/sum(out_disease.values()))*100)

    percentage = np.around(np.array(percentage), 2)

    explode = (0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(percentage, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Outliers per disease', fontsize=16)


    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'Calibri'

    with PdfPages('test.pdf') as pdf:
        pdf.savefig()

    return fig1



binary, outlier_binary = out_bin(x_bin, y_bin)
#pie = bin_pie(outlier_binary, y_bin)

binary.show()
#pie.show()

#tamanho da imagem 0.655 largura e comprimento