# Smart Surveillance - Building a Vision System for Pedestrian Tracking Attribute Recognition

## Structure
All the design and implementation material can be found inside the 'Project' folder:
- **multitask_model**: contains the .pth file for the used Multi-Task Neural Network
- **test_data**: contains the videos used for testing the final model and the groundtruth files
- **yolo_models**: contains the used YOLOv8 models
- **yolo_trackers**: contains the custom tracker used inside the YOLOv8 network, with some parameters which have been slightly modified
- **config.txt**: the ROIs initial configuration file
- **MTNN.py**: python script file for the Multi-Task Neural Network class implementing the Multi-Task model
- **MTNN_Training.ipynb**: Jupyter notebook containing all the code used for the Multi-Task Neural Network training
- **roi.py**: python script file which uses the 'config.txt' file to display the two ROIs on the video
- **video_processing.py**: python script file which implements all the reasoning for tracking with YOLOv8 and PAR with ViLT
- **group06.py**: python main script used to run the entire project code
- **vilt.py**: python script file that implements the ViLT pretrained model for Visual Question Answering applied to Pedestrian Attribute Recognition

Inside the main function in 'tracking.py' file there's a boolean flag called 'vilt'. You can set this flag to 'True' if you want to run Pedestrian Attribute Recognition using ViLT model, otherwise to 'False' if you want to run it with the Multi-Task model

## Github repository
Here's the [Github repository link](https://github.com/antoineb1/Artificial_vision), where you can find code written and used for the project implementation.

## Credits
The project has been designed and realized by Artificial Vision Group 06

- Members and references:
  - Antoine Barbet - [a.barbet1@studenti.unisa.it](mailto:a.barbet1@studenti.unisa.it)
  - Marco Milone - [m.milone15@studenti.unisa.it](mailto:m.milone15@studenti.unisa.it)
  - Martina Napoli - [m.napoli81@studenti.unisa.it](mailto:m.napoli81@studenti.unisa.it)
  - Fabrizio Sarno - [f.sarno14@studenti.unisa.it](mailto:f.sarno14@studenti.unisa.it)

- Data
  For the Multi-Task model training, [MIVIA](https://mivia.unisa.it/) dataset has been used.

- Institution
  [University of Salerno](https://web.unisa.it/en/university), DIEM, MIVIA
