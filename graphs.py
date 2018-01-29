import argparse
import os
import json
import pickle
from bokeh.plotting import figure, curdoc, output_file, show, save
from bokeh.palettes import Category10 as palette
from bokeh.models import NumeralTickFormatter


def create_graphs(sim_num=None, variable=None, resolution=None):
    CONFIGURATIONS_DIR = os.path.join(os.path.dirname(__file__))
    if sim_num is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--sim_num', action='store')
        parser.add_argument('--var', action='store')
        parser.add_argument('--resolution', action='store', default='epoch')
        args = vars(parser.parse_args())
        sim_num = args['sim_num']
        variable = args['var']
        resolution = args['resolution']
    folder_name = 'simulation_{}'.format(sim_num)
    colors = palette[10]

    p_loss = figure(plot_width=600, plot_height=600, min_border=10, min_border_left=50,
                    x_axis_label='t [Epochs]', y_axis_label='L(w(t))',
                    title="Training Loss", y_axis_type='log', x_axis_type='log')
    p_loss.background_fill_color = "#fafafa"

    p_norm = figure(plot_width=600, plot_height=600, min_border=10, min_border_left=50,
                    x_axis_label='t [Epochs]', y_axis_label='||w(t)|| Normalized',
                    title="The Norm of w(t) - Normalized", x_axis_type='log')
    p_norm.background_fill_color = "#fafafa"

    p_error = figure(plot_width=600, plot_height=600, min_border=10, min_border_left=50,
                     x_axis_label='t [Epochs]', y_axis_label='Error Rate',
                     title="Training Error", x_axis_type='log')
    p_error.background_fill_color = "#fafafa"
    p_error.yaxis[0].formatter = NumeralTickFormatter(format="0.0%")

    for idx, file in enumerate(os.listdir(os.path.join(CONFIGURATIONS_DIR, folder_name))):
        if file.endswith('.log'):
            continue
        with open(os.path.join(CONFIGURATIONS_DIR, folder_name, file), 'rb') as pickle_in:
            stats_test, stats_train = pickle.load(pickle_in)
        with open(os.path.join(CONFIGURATIONS_DIR, folder_name, file + '.log'), 'rb') as log_file:
            params_dict = json.load(log_file)
        var = params_dict[variable]

        legend = str(var)
        stats_train.export_data(handle_loss=p_loss,
                                handle_error=p_error,
                                handle_norm=p_norm,
                                legend='train - ' + legend,
                                color=colors[idx % 10],
                                line_dash='solid',
                                resolution=resolution)
        stats_test.export_data(handle_loss=p_loss,
                               handle_error=p_error,
                               handle_norm=p_norm,
                               legend='test - ' + legend,
                               color=colors[idx % 10],
                               line_dash='dashed',
                               resolution=resolution)
    p_loss.legend.click_policy = "hide"
    output_file(folder_name + '/loss.html')
    save(p_loss)
    p_error.legend.click_policy = "hide"
    output_file(folder_name + '/error.html')
    save(p_error)
    p_norm.legend.click_policy = "hide"
    p_norm.legend.location = "top_left"
    output_file(folder_name + '/norm.html')
    save(p_norm)


if __name__ == '__main__':
    create_graphs()
