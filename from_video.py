import numpy as np
import cv2
from scene3D import SceneReconstruction3D

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


def main():
    K = np.array([[2759.48 / 4, 0, 1520.69 / 4, 0, 2764.16 / 4,
                   1006.81 / 4, 0, 0, 1]]).reshape(3, 3)
    d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)
    scene = SceneReconstruction3D(K, d)

    old_img = None

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cam = cv2.VideoCapture(0)
    frame_count = 0
    while True:
        ret, frame = cam.read()
        frame_count += 1

        if old_img is None:
            old_img = frame
            continue

        # img1 = cv2.pyrDown(old_img)
        # img1 = cv2.pyrDown(img1)
        img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # img2 = cv2.pyrDown(frame)
        # img2 = cv2.pyrDown(img2)
        img2 = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)

        scene.load_image_pair(img1, img2)

        opt_flow_img = scene.plot_optic_flow()

        cv2.imshow("imgFlow", opt_flow_img)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            cam.release()
            break

        if not frame_count % 5:
            frame_count = 0
            old_img = frame

        Xs, Zs, Ys = scene.plot_point_cloud()

        ax.scatter(Xs, Ys, Zs, c=Ys, cmap=cm.hsv, marker='o')
        plt.pause(1)
        ax.clear()
    plt.show()


if __name__ == '__main__':
    main()
