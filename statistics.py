import numpy as np
import torch
from bokeh.plotting import figure, output_file, show, save
from bokeh.models import FuncTickFormatter, NumeralTickFormatter


class Statistics(object):

    @staticmethod
    def get_statistics(mode, *args, **kwargs):
        return {'image_classification': StatImage}[mode](*args, **kwargs)


class StatImage(object):
    def __init__(self, params):

        self._loss = []
        self._error = []
        self._norm = []
        self._epochs = params.epochs
        self._live_results = params.live_results
        self._log = params
        self._sim_num = params.simulation_number
        self._folder = params.folder_name
        self._iterations_per_epoch = params.iterations_per_epoch

    def save_loss(self, loss):
        self._loss.append(loss)

    def save_error(self, error):
        self._error.append(error)

    def save_norm(self, weights_dict):
        norm = torch.zeros(1)
        for name in weights_dict:
            norm = norm + torch.sum((weights_dict[name]).view(-1, 1) ** 2)
        self._norm.append(torch.sqrt(norm).numpy()[0])

    def _visualize_norm_normalized(self, handle=None, legend=None, color=None, resolution=None):
        if handle is None:
            return
        max_norm = self._norm[-1]
        norm = np.divide(self._norm, max_norm)
        if resolution == 'epoch':
            t = np.arange(1, self._epochs + 1)
        else:
            t = np.arange(self._iterations_per_epoch, (self._epochs + 1) * self._iterations_per_epoch,
                          self._iterations_per_epoch)
        handle.line(t, norm, line_width=3, line_dash='solid', legend=legend, line_color=color)

    def _visualize_norm(self, handle=None, legend=None, color=None, resolution=None):
        if handle is None:
            return
        norm = self._norm
        if resolution == 'epoch':
            t = np.arange(1, self._epochs + 1)
        else:
            t = np.arange(self._iterations_per_epoch, (self._epochs + 1) * self._iterations_per_epoch,
                          self._iterations_per_epoch)
        handle.line(t, norm, line_width=3, line_dash='solid', legend=legend, line_color=color)

    def _visualize_loss(self, handle=None, legend=None, color=None, line_dash=None, resolution=None):
        if handle is None:
            return
        loss = self._loss
        if resolution == 'epoch':
            t = np.arange(1, self._epochs + 1)
        else:
            t = np.arange(self._iterations_per_epoch, (self._epochs + 1) * self._iterations_per_epoch,
                          self._iterations_per_epoch)
        handle.line(t, loss, line_width=3, line_dash=line_dash, legend=legend, line_color=color)

    def _visualize_error(self, handle=None, legend=None, color=None, line_dash=None, resolution=None):
        if handle is None:
            return
        error = self._error
        if resolution == 'epoch':
            t = np.arange(1, self._epochs + 1)
        else:
            t = np.arange(self._iterations_per_epoch, (self._epochs + 1) * self._iterations_per_epoch,
                          self._iterations_per_epoch)
        handle.line(t, error, line_width=3, line_dash=line_dash, legend=legend, line_color=color)

    def export_data(self, handle_loss=None, handle_error=None, handle_norm_normalized=None, handle_norm=None,
                    legend=None, color=None, line_dash=None, resolution=None):
        self._visualize_loss(handle_loss, legend, color, line_dash, resolution)
        self._visualize_error(handle_error, legend, color, line_dash, resolution)
        self._visualize_norm_normalized(handle_norm_normalized, legend, color, resolution)
        self._visualize_norm(handle_norm, legend, color, resolution)
