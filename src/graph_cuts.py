import maxflow
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import argparse


class GraphCuts:
    """
    Main class for image synthesis with graph cuts
    """
    def __init__(self, src, sink, mask, save_graph=False):
        """
        Initializes the graph
        :param src: image to be blended (foreground)
        :param sink: background image
        :param mask: manual mask with constrained pixels
        :param save_graph: if true, graph is saved
        """
        assert (src.shape == sink.shape), "Source and sink dimensions must be the same: " + str(src.shape) + " != " + str(sink.shape)

        # Create the graph
        graph = maxflow.Graph[float]()
        # Add the nodes. nodeids has the identifiers of the nodes in the grid.
        node_ids = graph.add_grid_nodes((src.shape[0], src.shape[1]))

        # create adjacency matrix
        self.create_adj_matrix(src, sink)
        print("Assigning weights")

        # Add non-terminal edges
        # TODO: use alternate API which is more efficient
        patch_height = src.shape[0]
        patch_width = src.shape[1]
        eps = 1e-10     # for numerical stability, avoid divide by 0
        for row_idx in range(patch_height):
            for col_idx in range(patch_width):
                # matching cost is the sum of squared differences between the pixel values

                # right neighbor
                if col_idx + 1 < patch_width:
                    weight = self.adj_matrix[row_idx, col_idx, 0]
                    norm_factor = np.square(np.linalg.norm(src_patch[row_idx, col_idx] - src_patch[row_idx, col_idx + 1])) + \
                                  np.square(np.linalg.norm(sink_patch[row_idx, col_idx] - sink_patch[row_idx, col_idx + 1]))
                    weight = weight / (norm_factor + eps)
                    graph.add_edge(node_ids[row_idx][col_idx], node_ids[row_idx][col_idx + 1], weight, weight)

                # bottom neighbor
                if row_idx + 1 < patch_height:
                    weight = self.adj_matrix[row_idx, col_idx, 1]
                    norm_factor = np.square(np.linalg.norm(src_patch[row_idx, col_idx] - src_patch[row_idx + 1, col_idx])) + \
                                  np.square(np.linalg.norm(sink_patch[row_idx, col_idx] - sink_patch[row_idx + 1, col_idx]))
                    weight = weight / (norm_factor + eps)
                    graph.add_edge(node_ids[row_idx][col_idx], node_ids[row_idx + 1][col_idx], weight, weight)

                # Add terminal edge capacities
                # We constrain the pixels along the patch border to come from the sink, i.e. the background image.
                # The terminal edges are already initialized for all nodes with capacity 0. We will reassign the
                # capacities only for the nodes corresponding to border pixels.

                if np.array_equal(mask[row_idx, col_idx, :], [0, 255, 255]):
                    graph.add_tedge(node_ids[row_idx][col_idx], 0, np.inf)
                elif np.array_equal(mask[row_idx, col_idx, :], [255, 128, 0]):
                    graph.add_tedge(node_ids[row_idx][col_idx], np.inf, 0)

        # Plot graph
        if save_graph:
            nxg = graph.get_nx_graph()
            self.plot_graph_2d(nxg, patch_height, patch_width)

        flow = graph.maxflow()
        self.sgm = graph.get_grid_segments(node_ids)

    def create_adj_matrix(self, src, sink):
        """
        Create adjacency matrix of the graph
        """
        print("Creating adjacency matrix")
        self.adj_matrix = np.zeros((src.shape[0], src.shape[1], 2))
        src_left_shifted = np.roll(src, -1, axis=1)
        sink_left_shifted = np.roll(sink, -1, axis=1)
        src_up_shifted = np.roll(src, -1, axis=0)
        sink_up_shifted = np.roll(sink, -1, axis=0)
        self.adj_matrix[:, :, 0] = np.sum(np.square(src - sink, dtype=np.float) +
                                           np.square(src_left_shifted - sink_left_shifted, dtype=np.float), axis=2)
        self.adj_matrix[:, :, 1] = np.sum(np.square(src - sink, dtype=np.float) +
                                           np.square(src_up_shifted - sink_up_shifted, dtype=np.float), axis=2)

    def plot_graph_2d(self, graph, nodes_shape, plot_weights=False, plot_terminals=True, font_size=7):
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

    def test_case(self):
        # Create the graph
        graph = maxflow.Graph[float]()
        patch_height = 4
        patch_width = 5
        # Add the nodes. nodeids has the identifiers of the nodes in the grid.
        node_ids = graph.add_grid_nodes((patch_height, patch_width))

        edges = [
            [0, 0, 0, 1, 20, 20], 
            [0, 0, 1, 0, 20, 20], 
            [1, 0, 1, 1, 20, 20], 
            [1, 0, 2, 0, 20, 20], 
            [2, 0, 2, 1, 20, 20], 
            [2, 0, 3, 0, 20, 20], 
            [3, 0, 3, 1, 20, 20], 

            [0, 1, 0, 2, 20, 20], 
            [0, 1, 1, 1, 20, 20], 
            [1, 1, 1, 2, 1, 1], 
            [1, 1, 2, 1, 20, 20], 
            [2, 1, 2, 2, 1, 1], 
            [2, 1, 3, 1, 20, 20], 
            [3, 1, 3, 2, 20, 20], 

            [0, 2, 0, 3, 20, 20], 
            [0, 2, 1, 2, 1, 1], 
            [1, 2, 1, 3, 20, 20], 
            [1, 2, 2, 2, 20, 20], 
            [2, 2, 2, 3, 20, 20], 
            [2, 2, 3, 2, 1, 1], 
            [3, 2, 3, 3, 20, 20], 

            [0, 3, 0, 4, 20, 20], 
            [0, 3, 1, 3, 1, 1], 
            [1, 3, 1, 4, 1, 1], 
            [1, 3, 2, 3, 20, 20], 
            [2, 3, 2, 4, 1, 1], 
            [2, 3, 3, 3, 1, 1], 
            [3, 3, 3, 4, 20, 20], 

            [0, 4, 1, 4, 20, 20], 
            [1, 4, 2, 4, 20, 20], 
            [2, 4, 3, 4, 20, 20], 
        ]

        src_edges = [
            [0, 0], 
            [0, 1], 
            [0, 2], 
            [0, 3], 
            [0, 4], 
            [1, 0], 
            [2, 0], 
            [3, 0], 
            [3, 1], 
            [3, 2], 
            [3, 3], 
            [3, 4], 
            [2, 4], 
            [1, 4]
        ]

        sink_edges = [[2, 3]]

        for edge in edges:
            src_row = edge[0]
            src_col = edge[1]
            dst_row = edge[2]
            dst_col = edge[3]
            weight1 = edge[4]
            weight2 = edge[5]
            
            graph.add_edge(node_ids[src_row][src_col], node_ids[dst_row][dst_col], weight1, weight2)
        
        for edge in sink_edges:
            node_row = edge[0]
            node_col = edge[1]
            
            graph.add_tedge(node_ids[node_row][node_col], 0, np.inf)

        for edge in src_edges:
            node_row = edge[0]
            node_col = edge[1]
            
            graph.add_tedge(node_ids[node_row][node_col], np.inf, 0)

        self.plot_graph_2d(graph, node_ids.shape, True)

        flow = graph.maxflow()
        sgm = graph.get_grid_segments(node_ids)
        print(sgm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='image_dir', required=True, help='Image directory')
    args = parser.parse_args()

    image_dir = args.image_dir
    src = cv2.imread(os.path.join(image_dir, 'src.jpg'))
    target = cv2.imread(os.path.join(image_dir, 'target.jpg'))

    mask = cv2.imread(os.path.join(image_dir, 'our_mask.png'))
    # left corners of the patches
    src_roi_pt = (0, 0)     # (x, y)
    sink_roi_pt = (0, 0)    # (x, y)
    roi_width = src.shape[1]
    roi_height = src.shape[0]

    src_patch = src[src_roi_pt[1]: src_roi_pt[1] + roi_height, src_roi_pt[0]: src_roi_pt[0] + roi_width, :]
    sink_patch = target[sink_roi_pt[1]: sink_roi_pt[1] + roi_height, sink_roi_pt[0]: sink_roi_pt[0] + roi_width, :]

    graphcuts = GraphCuts(src_patch, sink_patch, mask)
    # graphcuts.test_case()

    target[graphcuts.sgm == True] = src[graphcuts.sgm == True]
    cv2.imwrite(os.path.join(image_dir, "result.png"), target)
