import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os


class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.color)):
            self.link_pairs[i].append(tuple(np.array(self.color[i]) / 255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i]) / 255.))


# # joint_id (
# #     0 - r_ankle, 1 - r_knee, 2 - r_hip, 3 - l_hip, 4 - l_knee, 5 - l_ankle,
# #     6 - pelvis, 7 - thorax, 8 - upper_neck,
# #     9 - head_top, 10 - r_wrist, 11 - r_elbow, 12 - r_shoulder, 13 - l_shoulder, 14 - l_elbow, 15 - l_wrist
# # )

# Style
# (R,G,B)
color2 = [
                                            (252, 0, 0),
                                            (252, 0, 0),
        (255, 255, 0), (255, 255, 0), (255, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
                                             (255, 0, 255),
                                        (0, 0, 255), (252, 176, 243),
                                        (0, 0, 255), (252, 176, 243),
                                        (0, 0, 255), (252, 176, 243),
                                            # (169, 209, 142)
]

link_pairs2 = [
                            [9, 8],
                            [8, 7],
    [10, 11], [11, 12], [12, 7], [7, 13], [13, 14], [14, 15],
                            [7, 6],
                        [2, 6], [6, 3],
                        [2, 1], [3, 4],
                        [1, 0], [4, 5]
]

order = [-7, 5, 4, -1, -3, -5, -2, -4, -6, 0, 1, 6, 3, 7, 2]
color2 = [color2[o] for o in order]
link_pairs2 = [link_pairs2[o] for o in order]


point_color2 = [(240, 2, 127), (240, 2, 127), (240, 2, 127),
                (240, 2, 127), (240, 2, 127),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (252, 176, 243), (0, 176, 240), (252, 176, 243),
                (0, 176, 240), (252, 176, 243), (0, 176, 240),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142)]

style = ColorStyle(color2, link_pairs2, point_color2)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize COCO predictions')
    # general
    parser.add_argument('--image-path',
                        help='Path of MPII val images',
                        type=str,
                        default='data/mpii/images/'
                        )

    parser.add_argument('--gt-anno',
                        help='Path of COCO val annotation',
                        type=str,
                        default='data/mpii/annot/valid.json'
                        )

    parser.add_argument('--save-path',
                        help="Path to save the visualizations",
                        type=str,
                        default='visualization/mpii/')

    parser.add_argument('--prediction',
                        help="Prediction file to visualize",
                        type=str,
                        required=True)

    parser.add_argument('--style',
                        help="Style of the visualization: Chunhua style or Xiaochu style",
                        type=str,
                        default='chunhua')

    args = parser.parse_args()

    return args


def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)

    return joints_dict


def plot(data, gt_data, img_path, save_path,
         link_pairs, ring_color, save=True):
    assert data.shape[0] == len(gt_data)
    img_names = set([gt['image'] for gt in gt_data])
    # for imgId in range(len(gt_data)):
    for img_name in img_names:
        # Read Images
        # img_name = gt_data[imgId]['image']
        dataIds = [id for id, gt in enumerate(gt_data) if gt['image']==img_name]
        img_file = img_path + img_name
        data_numpy = cv2.imread(img_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        h = data_numpy.shape[0]
        w = data_numpy.shape[1]
        # ref = h

        # Plot
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax = plt.subplot(1, 1, 1)
        bk = plt.imshow(data_numpy[:, :, ::-1])
        bk.set_zorder(-1)
        print(img_name)
        scores = []
        for dataId in dataIds:
            dt_joints = data[dataId].reshape(16, -1)
            gt_joints = np.array(gt_data[dataId]['joints']).reshape(16, -1)
            joints_dict = map_joint_dict(dt_joints)
            ref = (
                          (dt_joints[:, 0].max()-dt_joints[:, 0].min())**2
                          + (dt_joints[:, 1].max()-dt_joints[:, 1].min())**2
                  )**0.5*1.5

            scores.append(ref/((dt_joints-gt_joints)**2).sum()/16*1000)

            # stick
            for k, link_pair in enumerate(link_pairs):
                lw = ref / 100.
                line = mlines.Line2D(
                    np.array([joints_dict[link_pair[0]][0],
                              joints_dict[link_pair[1]][0]]),
                    np.array([joints_dict[link_pair[0]][1],
                              joints_dict[link_pair[1]][1]]),
                    ls='-', lw=lw, alpha=1, color=link_pair[2], )
                line.set_zorder(0)
                ax.add_line(line)
            # black ring
            for k in range(dt_joints.shape[0]):
                if dt_joints[k, 0] > w or dt_joints[k, 1] > h:
                    continue
                radius = ref / 100
                circle = mpatches.Circle(tuple(dt_joints[k, :2]),
                                         radius=radius,
                                         ec='black',
                                         fc=ring_color[k],
                                         alpha=1,
                                         linewidth=1)
                circle.set_zorder(1)
                ax.add_patch(circle)

            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.axis('off')
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)

        avg_score = sum(scores)/len(scores)
        if save:
            plt.savefig(save_path + '/score_' + str(np.int(avg_score)) + '_name_' + img_name.split('.')[0] + '.png',
                        format='png', bbox_inckes='tight', dpi=100)
            # plt.savefig(save_path + 'score_' + str(np.int(avg_score)) + '/name_' + img_name.split('.')[0] + '.pdf',
            #             format='pdf', bbox_inckes='tight', dpi=100)
            # plt.savefig(save_path + '/name_' + img_name.split('.')[0] + '.pdf',
            #             format='pdf', bbox_inckes='tight', dpi=100)
        # plt.show()
        plt.close()


if __name__ == '__main__':

    args = parse_args()
    colorstyle = style

    save_path = args.save_path
    img_path = args.image_path
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except Exception:
            print('Fail to make {}'.format(save_path))

    from scipy.io import loadmat
    data = loadmat(args.prediction)['preds']

    with open(args.gt_anno) as f:
        gt_data = json.load(f)
    plot(data, gt_data, img_path, save_path, colorstyle.link_pairs, colorstyle.ring_color, save=True)
