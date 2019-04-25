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
        assert (src.shape == sink.shape), "Source and sink dimensions must be the same: " + str(src.shape) + " != " + str(sink.shape)

        # Create the graph
        graph = maxflow.Graph[float]()
        # Add the nodes. nodeids has the identifiers of the nodes in the grid.
        node_ids = graph.add_grid_nodes((src.shape[0], src.shape[1]))
        # norm_factor = 10000

        # create adjacency matrix
        self.create_adj_matrix(src, sink)

        # Add non-terminal edges
        # TODO: use alternate API which is more efficient
        patch_height = src.shape[0]
        patch_width = src.shape[1]
        norm_factor = np.amax(self.adj_matrix)
        for row_idx in range(patch_height):
            for col_idx in range(patch_width):
                # matching cost is the sum of squared differences between the pixel values
                wt_curr = np.square(np.linalg.norm(src[row_idx, col_idx, :] - sink[row_idx, col_idx, :]))

                # right neighbor
                if col_idx + 1 < patch_width:
                    weight = self.adj_matrix[row_idx * patch_width + col_idx, 0]
                    weight = - weight/norm_factor
                    graph.add_edge(node_ids[row_idx][col_idx], node_ids[row_idx][col_idx + 1], weight, weight)

                # bottom neighbor
                if row_idx + 1 < patch_height:
                    weight = self.adj_matrix[row_idx * patch_width + col_idx, 1]
                    weight = - weight/norm_factor
                    graph.add_edge(node_ids[row_idx][col_idx], node_ids[row_idx + 1][col_idx], weight, weight)

                # Add terminal edge capacities
                # We constrain the pixels along the patch border to come from the sink, i.e. the background image.
                # The terminal edges are already initialized for all nodes with capacity 0. We will reassign the
                # capacities only for the nodes corresponding to border pixels.
                if row_idx == 0 or row_idx == patch_height - 1 or col_idx == 0 or col_idx == patch_width - 1:
                    graph.add_tedge(node_ids[row_idx][col_idx], np.inf, 0)
        graph.add_tedge(node_ids[patch_height//2][patch_width//2], 0, np.inf)

        # Plot graph
        # nxg = graph.get_nx_graph()
        # self.plot_graph_2d(nxg, patch_height, patch_width)

        flow = graph.maxflow()
        self.sgm = graph.get_grid_segments(node_ids)
        print(self.sgm)
        print(np.sum(self.sgm))

        # self.plot_graph_2d(graph, node_ids.shape, True)

        pass

    def create_adj_matrix(self, src, sink):
        """
        Create adjacency matrix of the graph
        """
        height = src.shape[0]
        width = src.shape[1]
        num_pixels = height * width
        self.adj_matrix = np.zeros((num_pixels, 2))
        for row_idx in range(height):
            for col_idx in range(width):
                wt_curr = np.square(np.linalg.norm(src[row_idx, col_idx, :] - sink[row_idx, col_idx, :]))

                # right neighbor
                if col_idx + 1 < width:
                    wt_right = np.square(np.linalg.norm(src[row_idx, col_idx + 1, :] - sink[row_idx, col_idx + 1, :]))
                    weight = wt_curr + wt_right
                    self.adj_matrix[row_idx * width + col_idx, 0] = weight
                    # self.adj_matrix[row_idx * width + col_idx + 1, row_idx * width + col_idx] = weight

                # bottom neighbor
                if row_idx + 1 < height:
                    wt_bottom = np.square(np.linalg.norm(src[row_idx + 1, col_idx, :] - sink[row_idx + 1, col_idx, :]))
                    weight = wt_curr + wt_bottom
                    self.adj_matrix[row_idx * width + col_idx, 1] = weight
                    # self.adj_matrix[(row_idx + 1) * width + col_idx, row_idx * width + col_idx] = weight

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
    # Load images
    # src = cv2.imread('../images/fish-small.jpg')
    # target = cv2.imread('../images/underwater-small.jpg')
    # # left corners of the patches
    # src_roi_pt = (150, 125)     # (x, y)
    # sink_roi_pt = (100, 100)    # (x, y)
    # roi_width = 150
    # roi_height = 120

    src = cv2.imread('../images/3.png')
    target = cv2.imread('../images/4.png')
    # left corners of the patches
    src_roi_pt = (0, 50)     # (x, y)
    sink_roi_pt = (0, 10)    # (x, y)
    roi_width = 590
    roi_height = 400

    src_patch = src[src_roi_pt[1]: src_roi_pt[1] + roi_height, src_roi_pt[0]: src_roi_pt[0] + roi_width, :]
    sink_patch = target[sink_roi_pt[1]: sink_roi_pt[1] + roi_height, sink_roi_pt[0]: sink_roi_pt[0] + roi_width, :]

    # cv2.imshow('Source patch', src_patch)
    # cv2.waitKey(0)
    # cv2.imshow('Sink patch', sink_patch)
    # cv2.waitKey(0)

    graphcuts = GraphCuts(src_patch, sink_patch)
    # graphcuts.test_case()

    sink_patch[graphcuts.sgm == True] = src_patch[graphcuts.sgm == True]
    # cv2.imwrite("result.png", sink_patch)
    cv2.imshow('Output', sink_patch)
    cv2.waitKey(0)
    pass
