from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tflite_runtime.interpreter import load_delegate  # coral
from tflite_runtime.interpreter import Interpreter

import matplotlib.pyplot as plt
from skimage.filters import gaussian
import numpy as np

interpreter = Interpreter('model_rgb_edgetpu.tflite', experimental_delegates=[load_delegate('libedgetpu.so.1.0')])  # coral
interpreter.allocate_tensors()

########################

def plot_output(rgb_img, depth_img, grasp_position_img, grasp_angle_img, ground_truth_bbs, no_grasps=1, grasp_width_img=None):
        """
        Visualise the outputs of the network.
        rgb_img, depth_img, grasp_position_img, grasp_angle_img should all be the same size.
        :param rgb_img: Original RGB image
        :param depth_img: Corresponding Depth Image (what was passed to the network)
        :param grasp_position_img: The grasp quality output of the GG-CNN
        :param grasp_angle_img: The grasp angle output of the GG-CNN
        :param ground_truth_bbs: np.array, e.g. loaded by dataset_processing.grasp.BoundingBoxes.load_from_file. Empty array is ok.
        :param no_grasps: Number of local-maxima of grasp_position_img to generate grasps for
        :param grasp_width_img: The grasp width output of the GG-CNN.
        """
        grasp_position_img = gaussian(grasp_position_img, 0, preserve_range=True)

        if grasp_width_img is not None:
            grasp_width_img = gaussian(grasp_width_img, 1.0, preserve_range=True)

        gt_bbs = BoundingBoxes.load_from_array(ground_truth_bbs)
        gs = detect_grasps(grasp_position_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps, ang_threshold=0)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(2, 2, 1)
        ax.imshow(rgb_img)
        

        for g in gs:
            g.plot(ax, color='r')
        

        for g in gt_bbs:
            g.plot(ax, color='g')
        

        ax = fig.add_subplot(2, 2, 2)
        ax.imshow(depth_img)
        for g in gs:
            g.plot(ax, color='r')
        

        for g in gt_bbs:
            g.plot(ax, color='g')

        ax = fig.add_subplot(2, 2, 3)
        plot1 = ax.imshow(grasp_position_img, cmap='Reds', vmin=0, vmax=1)
        plt.colorbar(plot1)
        

        ax = fig.add_subplot(2, 2, 4)
        plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
        plt.colorbar(plot)
        plt.show()

#######################

dataset_fn = 'dataset_210301_0839.hdf5'
f = h5py.File(dataset_fn, 'r')

img_ids = np.array(f['test/img_id'])
rgb_imgs = np.array(f['test/rgb'])
depth_imgs = np.array(f['test/depth_inpainted'])
bbs_all = np.array(f['test/bounding_boxes'])

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# details
print("== Input details ==")
print("name:", input_details[0]['name'])
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])
# print(input_details[0]['index'])

input_scale, input_zero_point = input_details[0]["quantization"]
print(input_scale, input_zero_point)

print("\n== Output details ==")
print("name:", output_details[0]['name'])
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])

img = 8

input_data = rgb_imgs[img] #np.expand_dims(depth_imgs[img], -1)
# input_data = np.asarray([input_data]) # got to be an easier way
# print(input_data)
# input_data = input_data * 127
input_data = (input_data / input_scale) + input_zero_point
input_data = np.expand_dims(input_data, axis=0).astype(input_details[0]["dtype"])
# input_data = np.array(input_data, dtype=np.int8)
# print(input_data)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# # # The function `get_tensor()` returns a copy of the tensor data.
# # # Use `tensor()` in order to get a pointer to the tensor.
# model_output_data_0 = (interpreter.get_tensor(output_details[0]['index']) - input_zero_point) * input_scale
# model_output_data_1 = (interpreter.get_tensor(output_details[1]['index']) - input_zero_point) * input_scale
# model_output_data_2 = (interpreter.get_tensor(output_details[2]['index']) - input_zero_point) * input_scale
# model_output_data_3 = (interpreter.get_tensor(output_details[3]['index']) - input_zero_point) * input_scale

model_output_data_0 = interpreter.get_tensor(output_details[0]['index']) / 1
model_output_data_1 = interpreter.get_tensor(output_details[1]['index']) / 1
model_output_data_2 = interpreter.get_tensor(output_details[2]['index']) / 1
model_output_data_3 = interpreter.get_tensor(output_details[3]['index']) / 1

# model_output_data_3 = model_output_data_3 / 255
# norm = np.linalg.norm(model_output_data_3)
# model_output_data_3 = model_output_data_3/norm
# print(model_output_data_3)

# grasp_positions_out = model_output_data_1
# grasp_angles_out = np.arctan2(model_output_data_2, model_output_data_0)/2.0
# grasp_width_out = model_output_data_3 * 150.0 #becuase image is 300x300

grasp_positions_out = model_output_data_0
grasp_angles_out = np.arctan2(model_output_data_2, model_output_data_1)/2.0
grasp_width_out = model_output_data_3 #* 150.0 #becuase image is 300x300

plot_output(rgb_imgs[img, ], depth_imgs[img, ], grasp_positions_out[0, ].squeeze(), grasp_angles_out[0, ].squeeze(), bbs_all[img, ],
                                no_grasps=NO_GRASPS, grasp_width_img=grasp_width_out[0, ].squeeze())
