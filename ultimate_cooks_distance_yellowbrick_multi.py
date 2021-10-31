from data_processing_multi_for_outliers import y_data as y_multi
from data_processing_multi_for_outliers import x_data as x_multi
from yellowbrick.regressor import CooksDistance
from yellowbrick.datasets import load_concrete
import numpy as np
import scipy as sp
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

class CooksDistanceMEU(CooksDistance):
    def fit(self, X, y, y_multi):

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

        self.draw(y_multi)
        return self

    def draw(self, y_multi):

        for i in range(len(y_multi)):
            if y_multi[i]=='BTAL': linefmt = 'olive'
            elif y_multi[i]=='AF': linefmt = 'teal'
            elif y_multi[i] == 'α-TAL': linefmt = 'indianred'
            else: linefmt = 'orange'

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
                lw=baseline.get_linewidth()
            )

        return self.ax

    def finalize(self):

        self.set_title("Cook's Distance Outlier Detection")
        self.ax.set_xlabel("instance index")
        self.ax.set_ylabel("influence (I)")

        if self.draw_threshold:
            self.ax.legend(loc=9, frameon=True)

def out_multi(x_multi, y_multi):

    dict = {"BTAL": 0, "α-TAL": 1, "AF": 2, "Control": 3}
    y = y_multi.copy()
    y_multi = np.array([dict[disease] for disease in y_multi])

    # Instantiate and fit the visualizer
    visualizer = CooksDistanceMEU(title='Cook\'s Distance outlier detection for all the data')
    visualizer.fit(x_multi.astype(float), y_multi.astype(float), y)

    cooks_distance = visualizer.distance_
    outliers = cooks_distance[cooks_distance > visualizer.influence_threshold_].index # visualizer.influence_threshold_ = 4/len(y_data)

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'Calibri'
    with PdfPages('test.pdf') as pdf:
        pdf.savefig()


    print (outliers)
    return visualizer, outliers



multi, outlier_multi = out_multi(x_multi, y_multi)

multi.show()

#tamanho da imagem 0.655 largura e comprimento