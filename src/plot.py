"""
Plot surface of the objective function
"""
import os
from argparse import ArgumentParser
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from .utils.data_utils import simple_trans
from .utils.optim_utils import get_specific_matrices

class DrawFig:
    """
    drawing class
    
    Parameters:
    ===========================
    1. path: Path of the numpy array
    2. file: File name of the numpy array
    3. save: Save the plot or not
    """
    def __init__(self, path: str, file: str, save: bool) -> None:
        self.path = path
        self.file = file
        self.save = save
        self.row = int(self.file.split('_')[1])
        self.mat = np.load(f"{self.path}{self.file}", allow_pickle = True)[:, 1:]
        self.platform = self.mat[:, 0]
        self.k = [2.6, 2.8, 3.0, 3.2]


    def get_tot_voice(self, count_1d: np.ndarray, voice_1d: np.ndarray) -> np.ndarray:
        """
        Compute total voice of each KOL
        
        Parameters:
        ===========================
        1. count_1d: count matrix, size = (row * col, )
        2. voice_1d: voice matrix, size = (row * col, )
        """
        tot_voice = count_1d * voice_1d
        tot_voice = np.expand_dims(tot_voice, axis = 1).reshape(self.row, int(self.mat.shape[1]/3))
        return tot_voice


    def simulate_voice_dec(self, mat: np.ndarray, pos: tuple) -> np.ndarray:
        """
        Simulate voice decrease of each KOL
        
        Parameters:
        ===========================
        1. voice_mat: voice matrix, size = (row, col)
        2. pos: position of the specific post
        """
        sim_count = np.linspace(1, 10, 91)
        if pos[1] == 0:
            trans_count = simple_trans(sim_count, self.k[0])
        elif pos[1] == 1:
            trans_count = simple_trans(sim_count, self.k[1])
        elif pos[1] == 2:
            trans_count = simple_trans(sim_count, self.k[2])
        else:
            trans_count = simple_trans(sim_count, self.k[3])
        dec_ratio = trans_count / sim_count
        vc_ratio = mat[pos[0], pos[1]]
        trans_vcratio = vc_ratio * dec_ratio
        return trans_vcratio


    def draw_heatmap(
        self, columns_name: list, matrix: np.ndarray,
        title: str, y_position: float
    ) -> go.Figure:
        """
        Heatmap drawing utils
        
        Parameters:
        ===========================
        1. columns_name: column names of the matrix
        2. matrix: matrix to draw
        3. title: title of the colorbar
        4. y_position: vertical position of the colorbar
        """
        fig = go.Figure(
            data = go.Heatmap(
                x = [f'KOL_{i}' for i in range(1, self.row + 1)],
                y = columns_name,
                z = matrix,
                text = matrix,
                texttemplate = "%{text:.2s}",
                textfont = {"size":10},
                colorscale = 'Blues',
                colorbar = {
                    'title': title,
                    'titleside': 'top',
                    'titlefont': {'size': 14},
                    'tickfont': {'size': 14},
                    'len': 0.33,
                    'y': y_position
                }
            )
        )
        return fig


    def line_chart(self, mat: np.ndarray) -> go.Figure:
        """
        Line chart drawing utils
        """
        fig = go.Figure()
        kol_positions = [(1, 1), (2, 1), (3, 1), (6, 1), (7, 1), (8, 1)]
        for idx, pos in enumerate(kol_positions, start=2):
            print(pos)
            fig.add_trace(
                go.Scatter(
                    x = np.linspace(1, 10, 91),
                    y = self.simulate_voice_dec(mat, pos),
                    name = f'KOL{idx}_post'
                )
            )
        fig.add_shape(
            type = 'line', x0 = 1, y0 = mat[0, 3],
            x1 = 10, y1 = mat[0, 3],
            line = dict(color = 'black', width = 1, dash = 'dash'),
            xref = 'x', yref = 'y',
        )
        fig.add_shape(
            type = 'line', x0 = 1, y0 = mat[1, 3],
            x1 = 10, y1 = mat[1, 3],
            line = dict( color = 'gray', width = 1, dash = 'dash'),
            xref = 'x', yref = 'y'
        )
        fig.update_layout(
            title = "根據 k 值的聲量遞減現象",
            xaxis_title = "X-axis",
            yaxis_title = "Y-axis",
        )
        return fig


    def main(self) -> None:
        """
        Main function of the class
        """
        cnt_data = get_specific_matrices(self.mat)[0].T
        voice_data = get_specific_matrices(self.mat)[2].T
        vp_ratio_data = get_specific_matrices(self.mat)[3].T
        count_fig = self.draw_heatmap(
            ['直播篇數', '一般貼文篇數', '短影音篇數', '影片篇數'],
            cnt_data, '每個 KOL 在不同型態文章的發文篇數', 0.9
        )
        vp_ratio_fig = self.draw_heatmap(
            ['直播 CP 值', '一般貼文 CP 值', '短影音 CP 值', '影片 CP 值'],
            vp_ratio_data, '每個 KOL 不同型態文章的 CP 值 (聲量 / 成本)', 0.5
        )
        tot_voice_fig = self.draw_heatmap(
            ['直播總聲量', '一般貼文總聲量', '短影音總聲量', '影片總聲量'],
            self.get_tot_voice(
                cnt_data.T.flatten(),
                voice_data.T.flatten()
            ).T,
            '每個 KOL 在不同型態文章的總聲量',
            0.1
        )
        full_fig = make_subplots(rows = 3, cols = 1)
        for trace in count_fig.data:
            full_fig.add_trace(trace, row = 1, col = 1)
        for trace in vp_ratio_fig.data:
            full_fig.add_trace(trace, row = 2, col = 1)
        for trace in tot_voice_fig.data:
            full_fig.add_trace(trace, row = 3, col = 1)
        full_fig.update_layout(
            title = 'Heatmap of 發文篇數、CP 值、總聲量',
            height = 900, width = 1200,
        )
        for i in range(1, 4):  # Assuming you have 3 rows
            full_fig.update_xaxes(tickangle = 45, row = i, col = 1)

        ## simulate_fig = self.line_chart(self.get_specific_matrices()[3])
        ## simulate_fig.show()
        full_fig.show()
        if self.save is True:
            pio.write_html(
                full_fig,
                f'{os.getcwd()}/data/plotly_fig/opt_heat.html'
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-f', '--file',
        type=str, help='File name of the ndarray'
    )
    parser.add_argument(
        '-s', '--save',
        type=bool, default=False, help='Save the plot'
    )
    args = parser.parse_args()
    figures = DrawFig(
        f'{os.getcwd()}/data/param_mat/',
        args.file,
        args.save,
    )
    figures.main()
