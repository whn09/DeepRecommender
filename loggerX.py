# THIS FILE IS COPY-PASTED FROM HERE: https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import torch
from tensorboardX import SummaryWriter
import numpy as np
import scipy.misc

try:
  from StringIO import StringIO  # Python 2.7
except ImportError:
  from io import BytesIO  # Python 3.x


class Logger(object):
  def __init__(self, log_dir):
    """Create a summary writer logging to log_dir."""
    self.writer = SummaryWriter(log_dir)

  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    self.writer.add_scalar(tag, value, step)
    
  def graph_summary(self, model):
    """Log a graph variable."""
    self.writer.add_graph(model, ()) # TODO can't display now

  def image_summary(self, tag, images, step):
    """Log a list of images."""

  def histo_summary(self, tag, values, step, bins=1000):
    """Log a histogram of the tensor of values."""