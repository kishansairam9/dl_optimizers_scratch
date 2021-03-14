import numpy as np
from time import sleep

def const_upd(params, grad_fn, hparams):
    gradient = grad_fn(*list(params['vec']))
    params['vec'] = params['vec'] - hparams['lr'] * gradient
    params['iters'] += 1
    return params

def const_stepsize(lr):
    return  {
        'hparams': {
            'lr': lr
        }, 
        'func': const_upd,
        'label': 'Constant Stepsize',
    }

def polyak_upd(params, grad_fn, hparams):
    gradient = grad_fn(*list(params['vec']))
    prev_delW = params.get('delW_store', np.array([0.0, 0.0]))
    params['delW_store'] = hparams['beta'] * prev_delW - hparams['lr'] * gradient
    params['vec'] += params['delW_store']
    params['iters'] += 1
    return params

def polyak_method(lr, beta=0.9):
    return {
        'hparams': {
            'lr': lr,
            'beta': beta
        },
        'func': polyak_upd,
        'label': 'Polyak Momentum'
    }

def nesterov_upd(params, grad_fn, hparams):
    prev_delW = params.get('delW_store', np.array([0.0, 0.0]))
    gradient = grad_fn(*list(params['vec'] + hparams['beta'] * prev_delW))
    params['delW_store'] = hparams['beta'] * prev_delW - hparams['lr'] * gradient
    params['vec'] += params['delW_store']
    params['iters'] += 1
    return params

def nesterov_method(lr, beta=0.9):
    return {
        'hparams': {
            'lr': lr,
            'beta': beta
        },
        'func': nesterov_upd,
        'label': 'Nesterov Accelerated Gradient'
    }

def adam_upd(params, grad_fn, hparams):
    gradient = grad_fn(*list(params['vec']))
    d = hparams['delta']
    g = hparams['gamma']
    npd, npg = hparams['power_optm'][0]*d, hparams['power_optm'][1]*g
    hparams['power_optm'] = (npd, npg)
    params['m_store'] = d * params.get('m_store', np.array([0.0, 0.0])) + (1 - d) * gradient
    params['v_store'] = g * params.get('v_store', np.array([0.0, 0.0])) + (1 - g) * np.square(gradient)
    norm_m = params['m_store'] / (1 - npd)
    norm_v = params['v_store'] / (1 - npg)
    params['vec'][0] -= hparams['lr'] * norm_m[0] / (norm_v[0]**0.5 + hparams['elipson'])
    params['vec'][1] -= hparams['lr'] * norm_m[1] / (norm_v[1]**0.5 + hparams['elipson'])
    params['iters'] += 1
    return params

def adam_descent(lr=0.001, delta=0.9, gamma=0.999, elipson=1e-8):
    return {
        'hparams': {
            'lr': lr,
            'delta': delta,
            'gamma': gamma,
            'elipson': elipson,
            'power_optm': (1, 1)
        },
        'func': adam_upd,
        'label': 'ADAM Optimizer'
    }