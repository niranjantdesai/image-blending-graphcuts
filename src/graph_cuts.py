import maxflow
# import scipy.io as sio
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class GraphCuts:
    """
    Main class for image synthesis with graph cuts
    """
    def __init__(self, src, sink):
        """
        Initializes the graph
        :param src: image to be blended (foreground)
        :param sink: background image
        """
        assert (src.shape == sink.shape), "Source and sink dimensions must be the same"

        # Create the graph
        graph = maxflow.Graph[float]()
        # Add the nodes. nodeids has the identifiers of the nodes in the grid.
        node_ids = graph.add_grid_nodes((src.shape[0], src.shape[1]))

        # Add non-terminal edges
        patch_height = src.shape[0]
        patch_width = src.shape[1]
        # TODO: use alternate API which is more efficient
        for row_idx in range(patch_height):
            for col_idx in range(patch_width):
                # matching cost is the sum of squared differences between the pixel values
                wt_curr = np.square(np.linalg.norm(src[row_idx, col_idx, :] - sink[row_idx, col_idx, :]))

                # right neighbor
                if col_idx + 1 < patch_width:
                    wt_right = np.square(np.linalg.norm(src[row_idx, col_idx + 1, :] - sink[row_idx, col_idx + 1, :]))
                    weight = wt_curr + wt_right
                    graph.add_edge(node_ids[row_idx][col_idx], node_ids[row_idx][col_idx + 1], weight, weight)

                # bottom neighbor
                if row_idx + 1 < patch_height:
                    wt_bottom = np.square(np.linalg.norm(src[row_idx + 1, col_idx, :] - sink[row_idx + 1, col_idx, :]))
                    weight = wt_curr + wt_bottom
                    graph.add_edge(node_ids[row_idx][col_idx], node_ids[row_idx + 1][col_idx], weight, weight)

                # Add terminal edge capacities
                # We constrain the pixels along the patch border to come from the sink, i.e. the background image.
                # The terminal edges are already initialized for all nodes with capacity 0. We will reassign the
                # capacities only for the nodes corresponding to border pixels.
                if row_idx == 0 or row_idx == patch_height - 1 or col_idx == 0 or col_idx == patch_width - 1:
                    graph.add_tedge(node_ids[row_idx][col_idx], 0, np.inf)

        # Plot graph
        # nxg = graph.get_nx_graph()
        # self.plot_graph_2d(nxg, patch_height, patch_width)
        self.plot_graph_2d(graph, node_ids.shape)

        pass

    def plot_graph_2d(self, graph, nodes_shape, plot_weights=True, plot_terminals=True, font_size=7):
        """
        Plot the graph to be used in graph cuts
        :param graph: PyMaxflow graph
        :param nodes_shape: patch shape
        :param plot_weights: if true, edge weights are shown
        :param plot_terminals: if true, the terminal nodes are shown
        :param font_size: text font size
        """
        X, Y = np.mgrid[:nodes_shape[0], :nodes_shape[1]]
        aux = np.array([Y.ravel(), X[::-1].ravel()]).T
        positions = {i: v for i, v in enumerate(aux)}
        positions['s'] = (-1, nodes_shape[0] / 2.0 - 0.5)
        positions['t'] = (nodes_shape[1], nodes_shape[0] / 2.0 - 0.5)

        nxgraph = graph.get_nx_graph()
        print("nxgraph created")
        if not plot_terminals:
            nxgraph.remove_nodes_from(['s', 't'])

        plt.clf()
        nx.draw(nxgraph, pos=positions)

        if plot_weights:
            edge_labels = {}
            for u, v, d in nxgraph.edges(data=True):
                edge_labels[(u, v)] = d['weight']
            nx.draw_networkx_edge_labels(nxgraph,
                                         pos=positions,
                                         edge_labels=edge_labels,
                                         label_pos=0.3,
                                         font_size=font_size)

        plt.axis('equal')
        plt.show()


if __name__ == '__main__':
    # Load images
    src = cv2.imread('../data/fish-small.jpg')
    target = cv2.imread('../data/underwater-small.jpg')

    # # Load mask
    # mat = sio.loadmat('../data/mask-small.mat')
    # mat = mat['mask']

    # # Crop image
    # cropped = cv2.bitwise_and(src, src, mask=mat)
    # cv2.imshow('Cropped image', cropped)
    # cv2.waitKey(0)

    # left corners of the patches
    src_roi_pt = (120, 120)     # (x, y)
    sink_roi_pt = (180, 140)    # (x, y)
    roi_width = 215
    roi_height = 140

    src_patch = src[src_roi_pt[1]: src_roi_pt[1] + roi_height, src_roi_pt[0]: src_roi_pt[0] + roi_width, :]
    sink_patch = target[sink_roi_pt[1]: sink_roi_pt[1] + roi_height, sink_roi_pt[0]: sink_roi_pt[0] + roi_width, :]

    # cv2.imshow('Source patch', src_patch)
    # cv2.waitKey(0)
    # cv2.imshow('Sink patch', sink_patch)
    # cv2.waitKey(0)

    graphcuts = GraphCuts(src_patch, sink_patch)
    pass
