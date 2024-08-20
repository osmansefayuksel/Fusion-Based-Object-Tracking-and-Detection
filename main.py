import argparse
import cv2
import imutils
import numpy as np
import time
from imutils.video import VideoStream, FPS
from mmdet.apis import init_detector, inference_detector
from scipy.spatial import distance as dist
from utils import compute_iou, load_annotations, compute_center_location_error, initialize_tracker, compute_visual_similarity, get_visual_features

model_paths = {
    "yolox": {
        "config": "C:/Users/osy/Desktop/OsmanSefa_Yuksel_rapor_2/Fusion/configs/yolox_cfg.py",
        "checkpoint": "C:/Users/osy/Desktop/OsmanSefa_Yuksel_rapor_2/Fusion/checkpoints/yolox.pth"
    },
    "faster": {
        "config": "C:/Users/osy/Desktop/OsmanSefa_Yuksel_rapor_2/Fusion/configs/faster_rcnn_cfg.py",
        "checkpoint": "C:/Users/osy/Desktop/OsmanSefa_Yuksel_rapor_2/Fusion/checkpoints/faster_rcnn.pth"
    },
    "cascade": {
        "config": "C:/Users/osy/Desktop/OsmanSefa_Yuksel_rapor_2/Fusion/configs/cascade_rcnn_cfg.py",
        "checkpoint": "C:/Users/osy/Desktop/OsmanSefa_Yuksel_rapor_2/Fusion/checkpoints/cascade_rcnn.pth"
    }
}

tracker_output = "C:/Users/osy/Desktop/OsmanSefa_Yuksel_rapor_2/Fusion/results/only_tracker/tracker.mp4"
fusion_output = "C:/Users/osy/Desktop/OsmanSefa_Yuksel_rapor_2/Fusion/results/fusion/fusion.mp4"

def tracker_only(video_path, annotations_path, tracker_type, output_path):
    annotations = load_annotations(annotations_path)
    vs = cv2.VideoCapture(video_path)
    time.sleep(1.0)
    fps = None
    initBB = None
    frame_count = 0
    ious = []
    center_errors = []
    video_fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    tracker = initialize_tracker(tracker_type)

    while True:
        frame = vs.read()[1]
        if frame is None:
            break

        if frame_count < len(annotations):
            gt_bbox = annotations[frame_count]

        if initBB is not None:
            success, box = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                pred_bbox = (x, y, x + w, y + h)
                iou = compute_iou(gt_bbox, pred_bbox)
                ious.append(iou)
                center_error = compute_center_location_error(gt_bbox, pred_bbox)
                center_errors.append(center_error)
            else:
                ious.append(0)
                center_errors.append(0)
        else:
            initBB = (int(gt_bbox[0]), int(gt_bbox[1]), int(gt_bbox[2] - gt_bbox[0]), int(gt_bbox[3] - gt_bbox[1]))
            tracker.init(frame, initBB)
            if fps is None:
                fps = FPS().start()
            else:
                fps.start()
            print(f'initBB: {initBB}')
            iou = compute_iou(gt_bbox, (initBB[0], initBB[1], initBB[0] + initBB[2], initBB[1] + initBB[3]))
            ious.append(iou)
            center_error = compute_center_location_error(gt_bbox, (initBB[0], initBB[1], initBB[0] + initBB[2], initBB[1] + initBB[3]))
            center_errors.append(center_error)

        frame_count += 1
        out.write(frame)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(int(1000 / video_fps)) & 0xFF
        if key == ord("q"):
            break

    vs.release()
    out.release()
    cv2.destroyAllWindows()
    
    average_iou = np.mean(ious)
    average_center_error = np.mean(center_errors)

    return ious, average_iou, center_errors, average_center_error

def tracker_with_detector(video_path, annotations_path, detector, tracker_type, output_path):
    if detector not in model_paths:
        print(f"Unsupported detector: {detector}")
        return
    
    config_file = model_paths[detector]["config"]
    checkpoint_file = model_paths[detector]["checkpoint"]

    annotations = load_annotations(annotations_path)
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    vs = cv2.VideoCapture(video_path)
    time.sleep(1.0)
    fps = None
    initBB = None
    frame_count = 0
    ious = []
    center_errors = []
    video_fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    tracker = initialize_tracker(tracker_type)

    while True:
        frame = vs.read()[1]
        if frame is None:
            break

        if frame_count < len(annotations):
            gt_bbox = annotations[frame_count]

        if initBB is not None:
            success, box = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                pred_bbox = (x, y, x + w, y + h)
                pred_feat = get_visual_features(frame, pred_bbox)
                iou = compute_iou(gt_bbox, pred_bbox)
                ious.append(iou)
                center_error = compute_center_location_error(gt_bbox, pred_bbox)
                center_errors.append(center_error)
            else:
                result = inference_detector(model, frame)
                pred_instances = result.pred_instances
                if len(pred_instances.bboxes) > 0:
                    det_bbox = pred_instances.bboxes[0].cpu().numpy()
                    det_feat = get_visual_features(frame, det_bbox)
                    similarity = compute_visual_similarity(pred_feat, det_feat)
                    print(f'[INFO] Similarity: {similarity:.2f}')

                    if similarity < 3.5:
                        initBB = (int(det_bbox[0]), int(det_bbox[1]), int(det_bbox[2] - det_bbox[0]), int(det_bbox[3] - det_bbox[1]))
                        tracker = initialize_tracker(tracker_type)
                        tracker.init(frame, initBB)
                        if fps is None:
                            fps = FPS().start()
                        else:
                            fps.start()
                        
                        print("[INFO] Tracker was restarted by Detector.")
                        pred_bbox = (initBB[0], initBB[1], initBB[0] + initBB[2], initBB[1] + initBB[3])
                        iou = compute_iou(gt_bbox, pred_bbox)
                        ious.append(iou)
                        center_error = compute_center_location_error(gt_bbox, pred_bbox)
                        center_errors.append(center_error)
                    else:
                        ious.append(0)
                        center_errors.append(0)

            if frame_count % 30 == 0:
                result = inference_detector(model, frame)
                pred_instances = result.pred_instances
                if len(pred_instances.bboxes) > 0:
                    det_bbox = pred_instances.bboxes[0].cpu().numpy()
                    det_feat = get_visual_features(frame, det_bbox)
                    det_center = ((det_bbox[0] + det_bbox[2]) / 2, (det_bbox[1] + det_bbox[3]) / 2)
                    cv2.rectangle(frame, (int(det_bbox[0]), int(det_bbox[1])), (int(det_bbox[2]), int(det_bbox[3])), (0, 0, 255), 2)
                    track_center = (x + w / 2, y + h / 2)

                    distance = dist.euclidean(det_center, track_center)
                    print(f'[INFO] Euclidian Distance: {distance:.2f}')
                    similarity = compute_visual_similarity(pred_feat, det_feat)

                    if distance > 0.50 and similarity < 3.5:
                        initBB = (int(det_bbox[0]), int(det_bbox[1]), int(det_bbox[2] - det_bbox[0]), int(det_bbox[3] - det_bbox[1]))
                        tracker = initialize_tracker(tracker_type)
                        tracker.init(frame, initBB)

                        print("[INFO] Tracker was restarted by Detector.")
                        if fps is None:
                            fps = FPS().start()
                        else:
                            fps.start()
                        pred_bbox = (initBB[0], initBB[1], initBB[0] + initBB[2], initBB[1] + initBB[3])
                        iou = compute_iou(gt_bbox, pred_bbox)
                        ious.append(iou)
                        center_error = compute_center_location_error(gt_bbox, pred_bbox)
                        center_errors.append(center_error)
                else:
                    ious.append(0)
                    center_errors.append(0)
        else:
            result = inference_detector(model, frame)
            pred_instances = result.pred_instances
            if len(pred_instances.bboxes) > 0:
                det_bbox = pred_instances.bboxes[0].cpu().numpy()
                initBB = (int(det_bbox[0]), int(det_bbox[1]), int(det_bbox[2] - det_bbox[0]), int(det_bbox[3] - det_bbox[1]))
                tracker.init(frame, initBB)
                if fps is None:
                    fps = FPS().start()
                else:
                    fps.start()
                
                print("[INFO] Tracker was started by Detector.")
                pred_bbox = (initBB[0], initBB[1], initBB[0] + initBB[2], initBB[1] + initBB[3])
                iou = compute_iou(gt_bbox, pred_bbox)
                ious.append(iou)
                center_error = compute_center_location_error(gt_bbox, pred_bbox)
                center_errors.append(center_error)

        frame_count += 1
        out.write(frame)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(int(1000 / video_fps)) & 0xFF
        if key == ord("q"):
            break

    vs.release()
    out.release()
    cv2.destroyAllWindows()

    average_iou = np.mean(ious)
    average_center_error = np.mean(center_errors)

    return ious, average_iou, center_errors, average_center_error

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, required=True, help="path to input video file")
    ap.add_argument("-a", "--annotations", type=str, required=True, help="path to annotations file")
    ap.add_argument("-t", "--tracker", type=str, default="csrt", help="OpenCV object tracker type")
    ap.add_argument("-d", "--detector", type=str, required=True, choices=["yolox", "faster", "cascade"], help="Detector type")
    ap.add_argument("-o1", "--output1", type=str, default=tracker_output, help="path to output video file for tracker only")
    ap.add_argument("-o2", "--output2", type=str, default=fusion_output, help="path to output video file for tracker with detector")
    args = vars(ap.parse_args())

    tracker_ious, tracker_avg_iou, tracker_center_errors, tracker_avg_center = tracker_only(args["video"], args["annotations"], args["tracker"], args["output1"])
    fusion_ious, fusion_avg_iou, fusion_center_errors, fusion_avg_center = tracker_with_detector(args["video"], args["annotations"], args["detector"], args["tracker"], args["output2"])

    print(f'\nTracker: {args["tracker"]}')
    print(f'Detector: {args["detector"]}')

    print(f'\nResults:')
    print(f'Mean IoU (Tracker): {tracker_avg_iou:.4f}')
    print(f'Mean IoU (Tracker + Detector): {fusion_avg_iou:.4f}')
    print(f'Average Central Position Error (Tracker): {tracker_avg_center:.4f}')
    print(f'Average Central Position Error (Tracker + Detector): {fusion_avg_center:.4f}')

    print(f'Successful Frame Rate (Tracker): {len([iou for iou in tracker_ious if iou > 0]) / len(tracker_ious):.4f}')
    print(f'Successful Frame Rate (Tracker + Detector): {len([iou for iou in fusion_ious if iou > 0]) / len(fusion_ious):.4f}')
