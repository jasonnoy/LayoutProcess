import numpy as np


def euclidean_distance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points.
    """
    return np.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))


class LayoutProcessor:
    def __init__(self, strategy="reorder"):
        self.strategy = strategy

    def poly_distance(self, box1, box2):
        """
        Calculate the distance between two rectangles.
        """
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

    def combine_boxes(self, box1, box2):
        x1, y1, x1b, y1b = box1
        x2, y2, x2b, y2b = box2
        x = min(x1, x2)
        y = min(y1, y2)
        xb = max(x1b, x2b)
        yb = max(y1b, y2b)
        return x, y, xb, yb

    def reorder_boxes(self, boxes):
        """
        Reorder the boxes by distance.
        """
        res = []
        for b_i in range(len(boxes)):
            if boxes[b_i] in res:
                continue
            res.append(boxes[b_i])
            min_dist = None
            for b_j in range(b_i + 1, len(boxes)):
                if boxes[b_j] in res:
                    continue
                dist = self.poly_distance(boxes[b_i], boxes[b_j])
                if min_dist is None or dist < min_dist[0]:
                    min_dist = (dist, b_j)
            res.append(boxes[min_dist[1]])
        return res

    def emerge_boxes(self, boxes, threshold=30):
        """
        Emerge boxes by distance.
        """
        for b_i in range(len(boxes)):
            if len(boxes[b_i]) == 0:
                continue
            for b_j in range(b_i + 1, len(boxes)):
                if len(boxes[b_j]) == 0:
                    continue
                dist = self.poly_distance(boxes[b_i], boxes[b_j])
                if dist < threshold:
                    boxes[b_i] = self.combine_boxes(boxes[b_i], boxes[b_j])
                    boxes[b_j] = []
        return boxes

    def process(self, boxes):
        if self.strategy == "reorder":
            return self.emerge_boxes(boxes)
        elif self.strategy == "emerge":
            return self.emerge_boxes(boxes)
        else:
            raise Exception("Unknown strategy: {}".format(self.strategy))

    def __call__(self, boxes):
        return self.process(boxes)
