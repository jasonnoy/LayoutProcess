import numpy as np
from LayoutProcessor import LayoutProcessor, euclidean_distance


class CoyoOcrProcessor(LayoutProcessor):
    def __init__(self, strategy='reorder', sep=" ", dist_metric="center"):
        super(CoyoOcrProcessor, self).__init__(strategy)
        self.sep = sep
        self.dist_metric = dist_metric

    def combine_boxes(self, box1, box2):
        coords_1, coords_2 = box1[0], box2[0]
        x1_tl, y1_tl = coords_1[0]
        x2_tl, y2_tl = coords_2[0]
        x1_tr, y1_tr = coords_1[1]
        x2_tr, y2_tr = coords_2[1]
        x1_br, y1_br = coords_1[2]
        x2_br, y2_br = coords_2[2]
        x1_bl, y1_bl = coords_1[3]
        x2_bl, y2_bl = coords_2[3]
        x_tl = min(x1_tl, x2_tl)
        y_tl = min(y1_tl, y2_tl)
        x_tr = max(x1_tr, x2_tr)
        y_tr = min(y1_tr, y2_tr)
        x_br = max(x1_br, x2_br)
        y_br = max(y1_br, y2_br)
        x_bl = min(x1_bl, x2_bl)
        y_bl = max(y1_bl, y2_bl)
        coords = [(x_tl, y_tl), (x_tr, y_tr), (x_br, y_br), (x_bl, y_bl)]
        text = box1[1][0] + self.sep + box2[1][0]
        score = np.mean([box1[1][1], box2[1][1]])
        res = [coords, (text, score)]
        return res

    def poly_distance(self, box1, box2):
        """
        Calculate the distance between two polygons.
        """
        # treat the polygons as rectangles
        if self.dist_metric == "rectangle":
            # use top left and bottom right corners
            x1, y1 = box1[0]
            x1b, y1b = box1[2]
            x2, y2 = box2[0]
            x2b, y2b = box2[2]

            # relative position of the second rectangle
            left = x2b < x1
            right = x1b < x2
            bottom = y2b < y1
            top = y1b < y2

            if top and left:
                return euclidean_distance(x1, y1b, x2b, y2)
            elif left and bottom:
                return euclidean_distance(x1, y1, x2b, y2b)
            elif bottom and right:
                return euclidean_distance(x1b, y1, x2, y2b)
            elif right and top:
                return euclidean_distance(x1b, y1b, x2, y2)
            elif left:
                return x1 - x2b
            elif right:
                return x2 - x1b
            elif bottom:
                return y1 - y2b
            elif top:
                return y2 - y1b
            else:  # rectangles intersect
                return 0.
        elif self.dist_metric == "center":
            # use center points
            x1_tl, y1_tl = box1[0]
            x2_tl, y2_tl = box2[0]
            x1_tr, y1_tr = box1[1]
            x2_tr, y2_tr = box2[1]
            x1_br, y1_br = box1[2]
            x2_br, y2_br = box2[2]
            x1_bl, y1_bl = box1[3]
            x2_bl, y2_bl = box2[3]
            x1 = np.mean(x1_tl, x1_tr, x1_br, x1_bl)
            y1 = np.mean(y1_tl, y1_tr, y1_br, y1_bl)
            x2 = np.mean(x2_tl, x2_tr, x2_br, x2_bl)
            y2 = np.mean(y2_tl, y2_tr, y2_br, y2_bl)
            return euclidean_distance(x1, y1, x2, y2)
        else:
            raise ValueError("Unknown polygon distance metric: {}".format(self.dist_metric))
