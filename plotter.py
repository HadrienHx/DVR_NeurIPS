import numpy as np 
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
matplotlib.rc('font', size=25)
# matplotlib.rcParams.update({'font.size': 20})

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt 
import json 
import os
import logging 


class Plotter(object):
    def __init__(self, filename=None, filter_out={}):
        self.min_error = np.inf 
        self.curves = []
        self.log = logging.getLogger("Main")
        if filename is not None:
            self.load(filename, filter_out=filter_out)

    def add_curve(self, name, error, time, config, step_with_iter):
        self.min_error = min(self.min_error, np.min(error))
        self.curves.append([name, error, time, config, step_with_iter])

    def reconstruct_time(self, step_with_iter, preset_times):
        step_type, iters = step_with_iter
        r_time = np.zeros(len(step_type) + 2)
        for i, t in enumerate(step_type):
            r_time[i + 1] = r_time[i] + preset_times[t]
        r_time[-1] = r_time[-2]
        return r_time[iters]

    def plot(self, show=True, preset_times=None):
        if len(self.curves) == 0:
            return

        plt.figure(figsize=(8,6))
        plt.subplots_adjust(top=0.98, bottom=0.12, left=0.16, right=0.965, hspace=0.2, wspace=0.2)
    
        handles = []

        for i, (name, error, time, _, step_with_iter) in enumerate(self.curves):

            if preset_times is None:
                plot_time = time
            else:
                plot_time = self.reconstruct_time(step_with_iter, preset_times)

            err = np.array(error) - self.min_error
            h, = plt.plot(plot_time, err, label=name) 
                            
            handles.append(h)

        font_size = 30
        plt.yscale("log")
        plt.tick_params(labelsize=25)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 4),)
        plt.legend(handles=handles, prop={'size': 27})

        plt.ylabel(r"Suboptimality")
        plt.xlabel(r"Time")

        if show:
            plt.show()

    def save(self, output_path=None, save_png=False, filename=None):
        path = os.path.join(output_path, filename)
        self.log.info(path)
        self.dump(path)
        if save_png:
            plt.savefig(os.path.join(output_path + ".png"))
            
    def dump(self, path):
        if len(self.curves) > 0:
            json.dump(self.curves, open(path, "w"))

    def load(self, filename, filter_out={}):
        if type(filename) == list:
            for f in filename:
                self.load(f) 

        elif type(filename) == str:
            for name, error, time, args, step_with_iter in json.load(open(filename)):
                if name not in filter_out:
                    self.add_curve(name, error, time, args, step_with_iter)