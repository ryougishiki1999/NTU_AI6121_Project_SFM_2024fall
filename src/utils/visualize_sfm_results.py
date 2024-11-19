from turtle import mode
import matplotlib.pyplot as plt
from mayavi import mlab


class SFMResultVisualizer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SFMResultVisualizer, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def _plot_points3D(self, points3D):
        self.ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c='blue', s=0.1)
        
    def _plot_camera_points3D(self, camera_points3D):
        pass

    def run(self, points3D, camera_points3D):
        # self._plot_points3D(points3D)
        # self._plot_camera_points3D(camera_points3D)
        #plt.show()
        mlab.points3d(points3D[:, 0], points3D[:, 1], points3D[:, 2], mode='point')
        mlab.show()
