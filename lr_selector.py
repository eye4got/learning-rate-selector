import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keras
import tensorflow as tf

import logging
import os
from typing import Tuple
from bisect import bisect_left


class LearningRateFinder(keras.callbacks.Callback):
    def __init__(self, min_lr=1e-6, max_lr=1, max_loss=5, num_steps=100):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.history = {}
        self.iteration = 0
        self.learning_rates = np.logspace(np.log10(min_lr), np.log10(max_lr), num_steps)
        self.max_loss = max_loss

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        if logs.get('loss') > self.max_loss or self.iteration == self.num_steps:
            self.model.stop_training = True
        
        else:
            self.history.setdefault('lr', []).append(self.learning_rates[self.iteration])
            self.history.setdefault('loss', []).append(logs.get('loss'))
            self.model.optimizer.learning_rate = self.learning_rates[self.iteration]
            self.iteration += 1
            
# Taken from fastai: https://github.com/fastai/fastai/blob/a00f6090418db26ff72fe86727d3a9ce63bf62c9/fastai/callback/schedule.py#L213C1-L231C59
def _valley(lrs:list, losses:list) -> Tuple[float, float]:
    "Suggests a learning rate from the longest valley and returns its index"
    n = len(losses)

    max_start, max_end = 0, 0

    # find the longest valley
    lds = [1]*n

    for i in range(1,n):
        for j in range(0,i):
            if (losses[i] < losses[j]) and (lds[i] < lds[j] + 1):
                lds[i] = lds[j] + 1
            if lds[max_end] < lds[i]:
                max_end = i
                max_start = max_end - lds[max_end]
    
    sections = (max_end - max_start) / 3
    idx = max_start + int(sections) + int(sections/2)

    return lrs[idx], losses[idx]


# TODO: perform sensitivity analysis on Slide rule parameters
def _slide(lrs:list, losses, lr_diff:int=15, thresh:float=.005, adjust_value:float=1.)-> Tuple[float, float]:
    "Suggests a learning rate following an interval slide rule and returns its index"

    if len(losses) < 2:
        return lrs[0], losses[0]

    loss_grad = np.gradient(losses)

    r_idx = -1
    l_idx = r_idx - lr_diff
    local_min_lr = lrs[l_idx]
    while (l_idx >= -len(losses)) and (abs(loss_grad[r_idx] - loss_grad[l_idx]) > thresh):
        local_min_lr = lrs[l_idx]
        r_idx -= 1
        l_idx -= 1
    
    suggestion = float(local_min_lr) * adjust_value
    suggestion_loss = np.interp(np.log10(suggestion), np.log10(lrs), losses)

    return suggestion, suggestion_loss


def select_learning_rate(model, x, y, default_lr:float=1e-2, max_loss:float=5, num_training:int=200, min_lr:float=1e-5, max_lr:float=5, plot:bool=False):

    lr_finder = LearningRateFinder(min_lr=min_lr, max_lr=max_lr, max_loss=max_loss, num_steps=num_training)
    history = model.fit(x=x, y=y, epochs=num_training, callbacks=[lr_finder], verbose=False)

    if len(lr_finder.history['lr']) < num_training:
        logging.info(f'LR Finder Experienced Diverging Loss! Iterations run: {len(lr_finder.history["lr"])}')

        if len(lr_finder.history['lr']) < 10:
            logging.warning(f'Less than 10 iterations tested, no meaningful suggestion will be provided')
            return -1, []

    valley_lr, valley_loss = _valley(lr_finder.history['lr'], lr_finder.history['loss'])
    slide_lr, slide_loss = _slide(lr_finder.history['lr'], lr_finder.history['loss'])
    avg_lr = np.exp((np.log(valley_lr) + np.log(slide_lr)) / 2)
    avg_loss = np.interp(np.log10(avg_lr), np.log10(lr_finder.history['lr']), lr_finder.history['loss'])

    # Remove LR checkpoints
    [os.remove(os.path.join(os.getcwd(), f)) for f in os.listdir(os.getcwd()) if 'lr_find' in f]

    def_loss_pos = bisect_left(lr_finder.history['lr'], default_lr)

    if plot:
        plt.subplots(figsize=(15, 8))
        plt.plot(lr_finder.history['lr'], lr_finder.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')

        if def_loss_pos < lr_finder.iteration:
            loss_for_suggestion = lr_finder.history['loss'][def_loss_pos]
            plt.plot([default_lr], [loss_for_suggestion], 'o-', color='blue', markeredgewidth=5, alpha=0.5, label='Default')
        
        plt.plot([slide_lr], [slide_loss], 'o-', color='purple', markeredgewidth=5, alpha=0.5, label='Slide')
        plt.plot([valley_lr], [valley_loss], 'o-', color='green', markeredgewidth=5, alpha=0.5, label='Valley')
        plt.plot([avg_lr], [avg_loss], 'o-', color='yellow', markeredgewidth=5, alpha=0.5, label='Combined')

        plt.legend(loc='upper left')
        plt.show()

    return avg_lr, history