import cv2, time, json, argparse, torch
from ultralytics import YOLO
from roi import ROI
from vilt import ViLTPAR
from PIL import Image
from MTNN import MultiTaskPAR

import os

def load_yolo(yolo_model_path):
    """
    Load a YOLO model.

    Parameters:
    - yolo_model_path (str): Path to the YOLO model.

    Returns:
    - YOLO: YOLO model instance.
    """
    return YOLO(yolo_model_path)

def load_vilt(vilt_model_path):
    """
    Load a ViLT model.

    Parameters:
    - vilt_model_path (str): Path to the ViLT model.

    Returns:
    - ViLTPAR: ViLT model instance.
    """
    return ViLTPAR(vilt_model_path)

def load_mtnn(mtnn_model_path):
    """
    Load a MultiTaskPAR model.

    Parameters:
    - mtnn_model_path (str): Path to the MultiTaskPAR model.

    Returns:
    - MultiTaskPAR: MultiTaskPAR model instance.
    """
    return MultiTaskPAR(mtnn_model_path)

def open_video(video_path):
    """
    Open a video file using OpenCV.

    Parameters:
    - video_path (str): Path to the video file.

    Returns:
    - cv2.VideoCapture: VideoCapture object.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        exit()
    return cap


def process_frames(yolo_model, par_model, cap, rois, tracking_data, fps, mapper, id_counterer):
    """
    Process frames from the video, update tracking data, and display the annotated frames.

    Parameters:
    - yolo_model (YOLO): YOLO model for object detection and tracking.
    - par_model (ViltPAR or MultiTaskPAR): model for pedestrian attribute extraction.
    - cap (cv2.VideoCapture): VideoCapture object for reading video frames.
    - rois (ROI): Instance of the ROI class containing region of interest information.
    - tracking_data (dict): Dictionary to store tracking data.
    - fps (int): Frames per second of the video.
    - mapper: data structure used to map used IDs
    - id_counterer: counter of IDs

    Returns:
    - None
    """
    # Number of frames to wait before updating tracking information
    frames_to_wait = fps * 2.5
    frame_counter = frames_to_wait
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Adapting OpenCV video window
    cv2.namedWindow("YOLOv8 Tracking + PAR", cv2.WINDOW_KEEPRATIO)

    while True:
        # Read the next frame from the video
        success, frame = cap.read()
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Break the loop if no more frames
        if not success:
            break

        # Use YOLO model for object detection and tracking
        results = yolo_model.track(frame, persist=True, classes=0, verbose=False, tracker="yolo_trackers/custom.yaml")

        # Compute bounding box informations
        bbinfo, id_counterer = calculate_bbox_info(results, mapper, id_counterer)

        # Decide whether to perform attribute extraction in the current frame
        if frame_counter >= frames_to_wait or current_frame == tot_frames:
            flag_par = True
            frame_counter = 0
        else:
            flag_par = False
            frame_counter += 1

        # Update tracking data based on the current frame
        update_data(frame, bbinfo, tracking_data, rois, par_model, flag_par)

        # Display the annotated frame with bounding boxes and ROIs
        annotated_frame = plot_bboxes(bbinfo, tracking_data, frame, mapper)
        rois.add_roi_to_image(annotated_frame)
        cv2.imshow("YOLOv8 Tracking + PAR", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

def update_data(frame, bbinfo, tracking_data, rois, par_model, flag_par):
    """
    Update tracking data based on object positions in ROIs and PAR.

    Parameters:
    - frame (numpy.ndarray): The current frame.
    - bbinfo (list): List containing information about detected objects, including object ID, center coordinates, and angles.
    - tracking_data (dict): Dictionary to store tracking data.
    - rois (ROI): Instance of the ROI class containing region of interest information.
    - par_model: An instance of the ViltPAR or MultiTaskPAR for attribute extraction.
    - flag_par (bool): Flag to determine whether attribute extraction should be performed.
    """

    for info in bbinfo:
        # Extraction of box attributes
        obj_id = info[0]
        centers = info[1]
        # coordinates of the four corners of each box
        angles = info[2]

        # Check if the center of the box is in one of the two ROIs
        is_in_roi1, is_in_roi2 = rois.point_in_rois((centers[0], centers[1]))


        if obj_id not in tracking_data:
            # Initialize tracking data for the object ID
            tracking_data[obj_id] = {
                "gender": None,
                "hat": None,
                "bag": None,
                "upper_color": None,
                "lower_color": None,
                "roi1_passages": 0,
                "roi1_persistence_time": 0,
                "roi1_flag": False,
                "roi2_passages": 0,
                "roi2_persistence_time": 0,
                "roi2_flag": False
            }

        if flag_par:

            #Extraction of the crop for each person
            cropped_frame = crop_objects(frame, obj_id, angles)
            attributes = par_model.extract_attributes(cropped_frame)

            # PAR attributes update
            tracking_data[obj_id]['gender'] = attributes[0]
            tracking_data[obj_id]['hat'] = attributes[1]
            tracking_data[obj_id]['bag'] = attributes[2]
            tracking_data[obj_id]['upper_color'] = attributes[3]
            tracking_data[obj_id]['lower_color'] = attributes[4]
        # Roi attributes update
        if is_in_roi1:
            update_roi_statistic(tracking_data, obj_id, "roi1")
        else:
            tracking_data[obj_id]['roi1_flag'] = False
        if is_in_roi2:
            update_roi_statistic(tracking_data, obj_id, "roi2")
        else:
            tracking_data[obj_id]['roi2_flag'] = False


def update_roi_statistic(tracking_data, obj_id, roi):
    """
    Update tracking data dictionary with the number of passages and persistence time.

    Parameters:
    - tracking_data (dict): Dictionary to store tracking data.
    - obj_id (str): Object ID.
    - roi (str): Region of interest identifier ("roi1" or "roi2").
    """
    # Define strings for dictionary keys
    str1 = "_passages"
    str2 = "_persistence_time"
    flag = "_flag"

    # Check if the object has entered the ROI
    if not tracking_data[obj_id][roi + flag]:
        # Increment the number of passages
        tracking_data[obj_id][roi + str1] += 1
        # Set the flag to indicate the object has entered
        tracking_data[obj_id][roi + flag] = True

    # Increment the persistence time
    tracking_data[obj_id][roi + str2] += 1



def calculate_bbox_info(results, mapper, id_counter):
    """
    Calculate the information of bounding boxes given a results object.

    Parameters:
    - results (ultralytics.YOLO): YOLO results object containing information about detected objects.
    - mapper: data structure used to map used IDs
    - id_counterer: counter of IDs

    Returns:
    - list: List of tuples, each containing the track ID and coordinates of the center and the corners of a bounding box ("id_x", (cx, cy), (x1, y1, x2, y2)).
    - id_counter: final sequential ID number
    """
    bbinfo = []

    # Iterate through each detection in the YOLO results
    for result in results:
        # Extract bounding box information from the results
        boxes = result.boxes.cpu().numpy()

        # Check if both xyxys and ids are non-empty
        if boxes.xyxy is None or boxes.id is None:
            continue

        xyxys = boxes.xyxy
        ids = boxes.id  # Use result.id to get track IDs

        # Process each bounding box in the current detection
        for xyxy, track_id in zip(xyxys, ids):
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            # Calculate the center of the bounding box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Convert track_id to a string in the format "id_x"
            track_id_str = f"id_{int(track_id)}"

            # Aggiornamento counter sequenziale
            if track_id_str not in mapper:
                mapper[track_id_str] =  id_counter
                id_counter += 1 
            # Append a tuple with track ID, bb center, and coordinates to the bbinfo list
            bbinfo.append((track_id_str, (cx, cy), (x1, y1, x2, y2)))

    return bbinfo, id_counter



def plot_bboxes(bbinfo, tracking_data, frame, mapper):
    """
    Plot bounding boxes and associated information on the input frame.

    Parameters:
    - bbinfo (list): List containing information about bounding boxes.
    - tracking_data (dict): Dictionary containing tracking information.
    - frame (numpy.ndarray): Input frame on which bounding boxes will be drawn.
    - mapper: data structure used to map used IDs

    Returns:
    - numpy.ndarray: Frame with drawn bounding boxes and labels.
    """

    hat = None
    bag = None
    upper_color = None
    lower_color = None

    labeling_color = None

    for info in bbinfo:

        obj_id = info[0]
        angles = info[2]

        x1 = angles[0]
        y1 = angles[1]
        x2 = angles[2]
        y2 = angles[3]

        tracking_info = tracking_data.get(obj_id, {})

        # Draw general info box
        h, w, c = frame.shape
        w = round((w / 100) * 19)
        h = round((h / 50) * 9)

        frame = cv2.rectangle(frame, (0, 0), (w, h), (255, 255, 255), -1)
        general_info = get_general_info(bbinfo, tracking_data)
        cv2.putText(frame, f'People in ROI: {general_info[0]}', (7, 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(frame, f'Total persons: {general_info[1]}', (7, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(frame, f'Passages in ROI 1: {general_info[2]}', (7, 130), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(frame, f'Passages in ROI 2: {general_info[3]}', (7, 180), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2)

        # Draw the bounding box in red
        if tracking_info['roi1_flag'] == True and tracking_info['roi2_flag'] == False:
            labeling_color = (255, 0, 0)
        elif tracking_info['roi1_flag'] == False and tracking_info['roi2_flag'] == True:
            labeling_color = (0, 255, 0)
        elif tracking_info['roi1_flag'] == False and tracking_info['roi2_flag'] == False:
            labeling_color = (0, 0, 255)
        
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), labeling_color, 3)

        # Add label with ID above the box (ID as an integer)
        id = str(mapper[obj_id])
        id_label_position = (x1 + 5, y1 + 25)
        frame = cv2.rectangle(frame, (x1, y1), (x1 + 40, y1 + 40), (255, 255, 255), -1)
        cv2.putText(frame, id, id_label_position, cv2.FONT_HERSHEY_DUPLEX, 0.8, labeling_color, 2)

        # Add label with PAR attributes below the box
        bound_w = x2 - x1
        bound_h = y2 - y1

        rect_w = 250
        rect_h = 100

        rect_x = round(x1 + bound_w / 2 - rect_w / 2)
        rect_y = y2

        frame = cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (255, 255, 255), -1)

        if tracking_info.get('gender', 0) == 'male':
            gender = 'M'
        elif tracking_info.get('gender', 0) == 'female':
            gender = 'F'
        else:
            gender = 'ND'

        gender_label_position = (rect_x + 7, rect_y + 30)
        cv2.putText(frame, f"Gender: {gender}", gender_label_position, cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)

        if tracking_info.get('hat', 0) == 'yes' or tracking_info.get('hat', 0) == 'true':
            hat = 'Hat'
        elif tracking_info.get('hat', 0) == 'no' or tracking_info.get('hat', 0) == 'false':
            hat = 'No Hat'

        if tracking_info.get('bag', 0) == 'yes' or tracking_info.get('bag', 0) == 'true':
            bag = 'Bag'
        elif tracking_info.get('bag', 0) == 'no' or tracking_info.get('bag', 0) == 'false':
            bag = 'No Bag'

        hat_bag_label_position = (rect_x + 7, rect_y + 60)
        cv2.putText(frame, f"{hat} {bag}", hat_bag_label_position, cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)

        if tracking_info.get("upper_color", 0) == 'tan':
            upper_color = 'Brown'
        elif tracking_info.get("upper_color", 0) == 'black and white':
            upper_color = 'Black'
        else:
            upper_color = tracking_info.get("upper_color", 0)

        if tracking_info.get("lower_color", 0) == 'tan':
            lower_color = 'Brown'
        elif tracking_info.get("lower_color", 0) == 'black and white':
            lower_color = 'Black'
        else:
            lower_color = tracking_info.get("lower_color", 0)

        color_label_position = (rect_x + 7, rect_y + 90)
        cv2.putText(frame, f"U-L: {upper_color}-{lower_color}", color_label_position, cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)

    return frame

def get_general_info(bbinfo, tracking_data):

    """
    Helps to extract all the general info to show during the video processing.

    Parameters
    - bbinfo (list): List containing information about bounding boxes.
    - tracking_data (dict): Dictionary containing tracking information.

    Returns
    - list of extracted information
    """

    people_in_roi = 0
    tot_people = 0
    roi1_passages = 0
    roi2_passages = 0 

    for id in tracking_data:
        tot_people = len(bbinfo)        
        info = tracking_data.get(id, {})

        if info['roi1_flag'] or info['roi2_flag']:
            people_in_roi += 1
        
        roi1_passages += info['roi1_passages']
        roi2_passages += info['roi2_passages']

    return [people_in_roi, tot_people, roi1_passages, roi2_passages]
            

def crop_objects(frame, id, angles):
    """
    Utility function that crops the frame to obtain images of people detected in the scene.

    Parameters:
    - frame (numpy.ndarray): Current scene frame.
    - angles (tuple): Bounding box coordinates (x1, y1, x2, y2).

    Returns:
    - Image.Image: Cropped image containing detected people.
    """
    # Extract bounding box coordinates
    x1, y1, x2, y2 = angles

    # Crop the section related to the bounded box
    cropped_image = frame[y1:y2, x1:x2].copy()

    # Convert the cropped array to an Image object
    cropped_image = Image.fromarray(cropped_image)

    return cropped_image

def save_tracking_statistics(tracking_data, output_file, fps, mapper):
    """
    Save tracking statistics for each object in the tracking data to a JSON file.

    Parameters:
    - tracking_data (dict): Dictionary containing tracking data.
    - output_file (str): The name of the output JSON file.
    - fps (float): Frames per second of the video.
    - mapper: data structure used to map used IDs
    """
    output_list = []
    hat = None
    bag = None
    upper_color = None
    lower_color = None

    for obj_id, data in tracking_data.items():

        # Hat, bag, upper color and lower color final value management

        if data.get("hat", False) == 'yes' or data.get("hat", False) == 'true':
            hat = True
        elif data.get("hat", False) == 'no' or data.get("hat", False) == 'false':
            hat = False

        if data.get("bag", False) == 'yes' or data.get("bag", False) == 'true':
            bag = True
        elif data.get("bag", False) == 'no' or data.get("bag", False) == 'false':
            bag = False

        if data.get("upper_color", False) == 'tan':
            upper_color = 'brown'
        elif data.get("upper_color", False) == 'black and white':
            upper_color = 'black'
        else:
            upper_color = data.get("upper_color", False)
            
        if data.get("lower_color", False) == 'tan':
            lower_color = 'brown'
        elif data.get("lower_color", False) == 'black and white':
            lower_color = 'black'
        else:
            lower_color = data.get("lower_color", False)


        entry = {
            "id": mapper[obj_id],
            "gender": data.get("gender", "unknown"),
            "hat": hat,
            "bag": bag,
            "upper_color": upper_color,
            "lower_color": lower_color,
            "roi1_passages": data.get("roi1_passages", 0),
            "roi1_persistence_time": round(data.get("roi1_persistence_time", 0) / fps, 2),
            "roi2_passages": data.get("roi2_passages", 0),
            "roi2_persistence_time": round(data.get("roi2_persistence_time", 0) / fps, 2)
        }
        output_list.append(entry)

    output_data = {"people": output_list}

    with open(output_file, 'w') as json_file:
        json.dump(output_data, json_file, indent=2)


