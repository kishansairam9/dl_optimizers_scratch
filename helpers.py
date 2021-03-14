import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import matplotlib.colors as mcolors
from time import sleep
import copy

def get_plot(func_dict, no_plot=False):
    if 'bounds' in func_dict:
        return FuncPlot(func_dict['func'], func_dict['x_range'], func_dict['y_range'], func_dict['lin_count'], func_dict['lines'], label=func_dict['label'], norm = mcolors.BoundaryNorm(func_dict['bounds'], plt.cm.rainbow.N), levels=func_dict['bounds'], line_plot=func_dict.get('line_plot', False), ab=func_dict.get('ab', False), no_plot=no_plot)
    return FuncPlot(func_dict['func'], func_dict['x_range'], func_dict['y_range'], func_dict['lin_count'], func_dict['lines'], label=func_dict['label'], line_plot=func_dict.get('line_plot', False), ab=func_dict.get('ab', False), no_plot=no_plot)

def converged(tresh, prev_val, next_val):
    return np.allclose([prev_val], [next_val], rtol=0, atol=tresh)

class FuncPlot:
    def __init__(self, func, x_range, y_range, lin_count, lines, label=None, levels=None, norm=None, line_plot=False, ab=False, no_plot=False):
        self.no_plot = no_plot
        if self.no_plot:
            return
        self.x_range, self.y_range = x_range, y_range
        x = np.linspace(*x_range, lin_count)
        y = np.linspace(*y_range, lin_count)
        X, Y = np.meshgrid(x, y)
        Z = func(X, Y)
        fig, ax = plt.subplots()
        if line_plot:
            cs = ax.contour(X, Y, Z, lines, cmap='rainbow', levels=levels, norm=norm)
        else:
            cs = ax.contourf(X, Y, Z, lines, cmap='rainbow', levels=levels, norm=norm)
        fig.colorbar(cs)
        ax.set_xlabel('x' if not ab else 'a')
        ax.set_ylabel('y' if not ab else 'b')
        if label:
            ax.set_title(label)
        self.fig, self.ax = fig, ax

    def add_arrow(self, start, end, arrow_width=0.04):
        if self.no_plot:
            return
        for i, (s,e) in enumerate(zip(start, end)):
            mi = min(s,e)
            ma = max(s,e)
            if i == 0:
                if ma < self.x_range[0]-0.01 or mi > self.x_range[1]+0.01:
                    return
            else:
                if ma < self.y_range[0]-0.01 or mi > self.y_range[1]+0.01:
                    return
        upd = [e-s for s,e in zip(start, end)]
        self.ax.arrow(*start, *upd, width=arrow_width, color='black')
        
    def show_plot(self):
        if self.no_plot:
            return
        clear_output(wait=True)
        display(self.fig)
        
class Experiment:
    def __init__(self, func_dict, update_dict, tresh=None, no_plot=False):
        self.func_dict = func_dict
        self.tresh = 1e-4 if not tresh else tresh
        self.update_dict = update_dict
        self.no_plot = no_plot
        
    def get_rand_vec(self):
        return np.array([random.uniform(*self.func_dict['x_range']), random.uniform(*self.func_dict['y_range'])])
    
    def descent(self, force_vec=None, arrow_width=0.012, continue_old=False, tresh=None, get_err=False, get_fn=None):
        self.plot = get_plot(self.func_dict, no_plot=self.no_plot) # Create new plot everytime
        params = self.params if continue_old else {'vec': self.get_rand_vec() if force_vec is None else force_vec, 'iters': 0}
        self.tresh = tresh if tresh else self.tresh
        err = []
        if get_fn:
            fn = []
        done = 0
        while done < 4:
            if get_err:
                err.append(self.func_dict['func'](*list(params['vec'])))
            if get_fn:
                fn.append(get_fn(*list(params['vec'])))
            next_params = self.update_dict['func'](copy.deepcopy(params), self.func_dict['grad'], self.update_dict['hparams'])
            vec_change = (params['vec'], next_params['vec'])
            abs_change_vec = np.abs(vec_change[0] - vec_change[1])
            
            self.plot.add_arrow(*vec_change, arrow_width)
            if converged(self.tresh, self.func_dict['func'](*list(params['vec'])), self.func_dict['func'](*list(next_params['vec']))):
                done += 1
            else:
                done = 0
            params = next_params
            if np.isnan(params['vec']).any() or np.isinf(params['vec']).any(): 
                print("Descent diverged and overflow has occured for given initialization params!")
                return
            if not self.no_plot:
                self.plot.ax.set_title(f"{self.func_dict['label']}:= {params['iters']} iters of {self.update_dict['label']}; LR: {self.update_dict['hparams']['lr']}")
            self.plot.show_plot()
            if self.no_plot:
                clear_output(wait=True)
            print("Abs change in postition in this update: ", abs_change_vec)
            print(f"At: ", params['vec'], " has value: ", self.func_dict['func'](*list(params['vec'])))
        print(f"Converged, change in function value after update over 5 continous iterations <= {self.tresh}")
        self.params = params
        if get_err:
            err.append(self.func_dict['func'](*list(params['vec'])))
            ret = err
            if get_fn:
                fn.append(get_fn(*list(params['vec'])))
                ret = err, fn
            return ret