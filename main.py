from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tflite_runtime.interpreter import load_delegate  # coral
from tflite_runtime.interpreter import Interpreter

import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.draw import polygon
from skimage.feature import peak_local_max
import numpy as np
import h5py

interpreter = Interpreter('model_rgb_edgetpu.tflite', experimental_delegates=[load_delegate('libedgetpu.so.1.0')])  # coral
interpreter.allocate_tensors()

NO_GRASPS = 1

########################

def _bb_text_to_no(l, offset=(0, 0)):
    # Read Cornell grasping dataset.
    x, y = l.split()
    return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]


class BoundingBoxes(object):
    """
    Convenience class for storing a set of grasps represented by bounding boxes.
    """
    def __init__(self, bbs=None):
        if bbs:
            self.bbs = bbs
        else:
            self.bbs = []

    def __getitem__(self, item):
        return self.bbs[item]

    def __iter__(self):
        return self.bbs.__iter__()

    def __getattr__(self, attr):
        # Call a function on all bbs
        if hasattr(BoundingBox, attr) and callable(getattr(BoundingBox, attr)):
            return lambda *args, **kwargs: list(map(lambda bb: getattr(bb, attr)(*args, **kwargs), self.bbs))
        else:
            raise AttributeError("Couldn't find function %s in BoundingBoxes or BoundingBox" % attr)

    @classmethod
    def load_from_array(cls, arr):
        bbs = []
        for i in range(arr.shape[0]):
            bbp = arr[i, :, :].squeeze()
            if bbp.max() == 0:
                break
            else:
                bbs.append(BoundingBox(bbp))
        return cls(bbs)

    @classmethod
    def load_from_file(cls, fname):
        bbs = []
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1, p2, p3 = f.readline(), f.readline(), f.readline()
                try:
                    bb = np.array([
                        _bb_text_to_no(p0),
                        _bb_text_to_no(p1),
                        _bb_text_to_no(p2),
                        _bb_text_to_no(p3)
                    ])

                    bbs.append(BoundingBox(bb))

                except ValueError:
                    # Some files contain weird values, ignore them.
                    continue
        return cls(bbs)

    def append(self, bb):
        self.bbs.append(bb)

    def copy(self):
        new_bbs = BoundingBoxes()
        for bb in self.bbs:
            new_bbs.append(bb.copy())
        return new_bbs

    def show(self, ax=None, shape=None):
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1, 1, 1)
            ax.imshow(np.zeros(shape))
            ax.axis([0, shape[1], shape[0], 0])
            self.plot(ax)
            plt.show()
        else:
            self.plot(ax)

    def draw(self, shape, position=True, angle=True, width=True):
        if position:
            pos_out = np.zeros(shape)
        else:
            pos_out = None
        if angle:
            ang_out = np.zeros(shape)
        else:
            ang_out = None
        if width:
            width_out = np.zeros(shape)
        else:
            width_out = None

        for bb in self.bbs:
            rr, cc = bb.compact_polygon_coords(shape)
            if position:
                pos_out[rr, cc] = 1.0
            if angle:
                ang_out[rr, cc] = bb.angle
            if width:
                width_out[rr, cc] = bb.length

        return pos_out, ang_out, width_out

    def to_array(self, pad_to=0):
        a = np.stack([bb.points for bb in self.bbs])
        if pad_to:
           if pad_to > len(self.bbs):
               a = np.concatenate((a, np.zeros((pad_to - len(self.bbs), 4, 2))))
        return a.astype(np.int)

    @property
    def center(self):
        points = [bb.points for bb in self.bbs]
        return np.mean(np.vstack(points), axis=0).astype(np.int)


class BoundingBox:
    """
    Class for manipulating grasps represented as bounding boxes (i.e. 4 corners of a rectangle)
    """
    def __init__(self, points):
        self.points = points

    def __str__(self):
        return str(self.points)

    @property
    def angle(self):
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return (np.arctan2(-dy, dx) + np.pi/2) % np.pi - np.pi/2

    @property
    def as_grasp(self):
        return Grasp(self.center, self.angle, self.length, self.width)

    @property
    def center(self):
        return self.points.mean(axis=0).astype(np.int)

    @property
    def length(self):
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    @property
    def width(self):
        dy = self.points[2, 1] - self.points[1, 1]
        dx = self.points[2, 0] - self.points[1, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    def polygon_coords(self, shape=None):
        return polygon(self.points[:, 0], self.points[:, 1], shape)

    def compact_polygon_coords(self, shape=None):
        return Grasp(self.center, self.angle, self.length/3, self.width).as_bb.polygon_coords(shape)

    def iou(self, bb, angle_threshold=np.pi/6):
        if abs(self.angle - bb.angle) % np.pi > angle_threshold:
            return 0

        rr1, cc1 = self.polygon_coords()
        rr2, cc2 = polygon(bb.points[:, 0], bb.points[:, 1])

        try:
            r_max = max(rr1.max(), rr2.max()) + 1
            c_max = max(cc1.max(), cc2.max()) + 1
        except:
            return 0

        canvas = np.zeros((r_max, c_max))
        canvas[rr1, cc1] += 1
        canvas[rr2, cc2] += 1
        union = np.sum(canvas > 0)
        if union == 0:
            return 0
        intersection = np.sum(canvas == 2)
        return intersection * 1.0/union

    def copy(self):
        return BoundingBox(self.points.copy())

    def offset(self, offset):
        self.points += np.array(offset).reshape((1, 2))

    def rotate(self, angle, center):
        R = np.array(
            [
                [np.cos(-angle), np.sin(-angle)],
                [-1 * np.sin(-angle), np.cos(-angle)],
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(R, (self.points - c).T)).T + c).astype(np.int)

    def plot(self, ax, color=None):
        points = np.vstack((self.points, self.points[0]))
        ax.plot(points[:, 1], points[:, 0], color=color)

    def zoom(self, factor, center):
        T = np.array(
            [
                [1.0/factor, 0],
                [0, 1.0/factor]
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(T, (self.points - c).T)).T + c).astype(np.int)


class Grasp:
    """
    Class for manipulating grasps represented by a centre pixel, angle and width.
    """
    def __init__(self, center, angle, length=60, width=30):
        self.center = center
        self.angle = angle  # Positive angle means rotate anti-clockwise from horizontal.
        self.length = length
        self.width = width

    @property
    def as_bb(self):
        xo = np.cos(self.angle)
        yo = np.sin(self.angle)

        y1 = self.center[0] + self.length/2.0 * yo
        x1 = self.center[1] - self.length/2.0 * xo
        y2 = self.center[0] - self.length/2.0 * yo
        x2 = self.center[1] + self.length/2.0 * xo

        return BoundingBox(np.array(
            [
             [y1 - self.width/2.0 * xo, x1 - self.width/2.0 * yo],
             [y2 - self.width/2.0 * xo, x2 - self.width/2.0 * yo],
             [y2 + self.width/2.0 * xo, x2 + self.width/2.0 * yo],
             [y1 + self.width/2.0 * xo, x1 + self.width/2.0 * yo],
             ]
        ).astype(np.int))

    def max_iou(self, bbs):
        self_bb = self.as_bb
        max_iou = 0
        for bb in bbs:
            iou = self_bb.iou(bb)
            max_iou = max(max_iou, iou)
        return max_iou

    def plot(self, ax, color=None):
        self.as_bb.plot(ax, color)


def detect_grasps(point_img, ang_img, width_img=None, no_grasps=1, ang_threshold=5):
    """
    Detect Grasps in a GG-CNN output.
    """
    local_max = peak_local_max(point_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasps)

    grasps = []

    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)

        grasp_angle = ang_img[grasp_point]
        if ang_threshold > 0:
            if grasp_angle > 0:
                grasp_angle = ang_img[grasp_point[0] - ang_threshold:grasp_point[0] + ang_threshold + 1,
                                      grasp_point[1] - ang_threshold:grasp_point[1] + ang_threshold + 1].max()
            else:
                grasp_angle = ang_img[grasp_point[0] - ang_threshold:grasp_point[0] + ang_threshold + 1,
                                      grasp_point[1] - ang_threshold:grasp_point[1] + ang_threshold + 1].min()

        g = Grasp(grasp_point, grasp_angle)
        if width_img is not None:
            g.length = width_img[grasp_point]
            g.width = g.length/2.0

        grasps.append(g)

    return grasps

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

dataset_fn = '/home/pi/Desktop/ggcnn_tf/dataset_210301_0839.hdf5'
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
