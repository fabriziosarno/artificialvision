import argparse
from video_processing import *


def main():  

    id_counter = 1
    mapper = {}

    # Create an argument parser with descriptions for command-line arguments
    parser = argparse.ArgumentParser(description='Process video frames with YOLO and ViLT models.')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file (mp4).')
    parser.add_argument('--configuration', type=str, required=True, help='Path to the ROI configuration file (txt).')
    parser.add_argument('--results', type=str, required=True, help='Path to the output JSON format file (txt).')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load YOLO model
    yolo_model_path = 'yolo_models/yolov8s.pt'
    yolo_model = load_yolo(yolo_model_path)
    yolo_model.to("cuda")

    # variable to choose which method to use for the par
    vilt = True
    if vilt:
        # Load ViLT model
        vilt_model_path = 'dandelin/vilt-b32-finetuned-vqa'
        vilt_model = load_vilt(vilt_model_path)
        vilt_model.to("cuda")
        par_model = vilt_model
    else:
        # Load MTNN model
        mtnn_model_path = 'multitask_model/mtnn_best_model.pth'
        mtnn_model = load_mtnn(mtnn_model_path)
        mtnn_model.to("cuda")
        par_model = mtnn_model


    # Open the video file
    video_path = args.video
    cap = open_video(video_path)

    # Create an ROI manager and read ROIs from the JSON file
    roi_manager = ROI(args.configuration, video_path)

    # Initialize tracking data dictionary
    tracking_data = {}

    # Get video fps
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Process frames using YOLO, and PAR model
    process_frames(yolo_model, par_model, cap, roi_manager, tracking_data, fps, mapper, id_counter)

    # Save tracking statistics to an output JSON file
    save_tracking_statistics(tracking_data, args.results, fps, mapper)

if __name__ == "__main__":
    main()