import tensorflow as tf
import cv2
import time
import argparse

import posenet

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()


def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        #平均フレーム数の算出に使用
        start = time.time()
        frame_count = 0
        rates = []
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            keypoint_coords *= output_scale
            for pi in range(len(pose_scores)):
                if pose_scores[pi] == 0.:
                    break
                #print("FrameNumber",frame_count)
                #print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                emp = []
                # 各部位座標の取り出し
                for point in keypoint_coords[pi, :, :]:
                    emp.append(point) #y,x座標
                #for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                   # print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
                #1フレームの座標情報リスト(y,x)×17
                emp = np.array(emp)
                print(emp)
                print()
                #距離比率計算
                # 0鼻 1左目 2右目 3左耳 4右耳 5左肩 6右肩 7左肘 8右肘
                # 9左手首 10右手首 11左腰 12右腰 13左膝 14右膝
                dis0 = np.linalg.norm(emp[0]-emp[1])
                dis1 = np.linalg.norm(emp[2]-emp[1])
                dis2 = np.linalg.norm(emp[4]-emp[3])
                rate = np.array([dis0, dis1, dis2]) / dis0

                #各距離比率の配列([[dis0][dis1]....])
                rates.append(list(rate))

            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            cv2.imshow('posenet', overlay_image) #ウィンドウ表示(ウィンドウ名、出力画像)
            frame_count += 1 #フレーム数をカウント(平均FPSの算出)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): #Qキーで終了
                break
        rates = np.array(rates).T
        print(rates)

        for i in rates:
            plt.plot(np.arange(len(i)), i)
        plt.xlim(0, len(rates[0]))
        plt.show()

        print('Average FPS: ', frame_count / (time.time() - start)) #平均FPSを表示


if __name__ == "__main__":
    main()