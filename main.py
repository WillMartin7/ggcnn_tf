from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tflite_runtime.interpreter import load_delegate  # coral
from tflite_runtime.interpreter import Interpreter

interpreter = Interpreter('model_rgb_edgetpu.tflite', experimental_delegates=[load_delegate('libedgetpu.so.1.0')])  # coral
interpreter.allocate_tensors()

