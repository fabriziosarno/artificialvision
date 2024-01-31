from transformers import ViltProcessor, ViltForQuestionAnswering
import torch

class ViLTPAR:
    def __init__(self, vilt_model):
        """
        Initializes the ViLTPAR object.

        Parameters:
        - vilt_model (str): Path to the ViLT pre-trained model.
        """
        # Initialize ViLT processor and question-answering model
        self.processor = ViltProcessor.from_pretrained(vilt_model)
        self.model = ViltForQuestionAnswering.from_pretrained(vilt_model)

        # Set predefined questions for each attribute
        self.gender_question = "Is the person male or female?"
        self.hat_question = "Does the person have a hat?"
        self.bag_question = "Does the person have a bag?"
        self.upper_clothing_question = "What color is the upper clothing?"
        self.lower_clothing_question = "What color is the lower clothing?"

    def to(self, mode):
        """
        Switch to CUDA execution if available.

        Parameters:
        - mode: execution mode (CUDA or CPU)
        """
        if mode == "cuda":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
        else:
            self.model = self.model.to("cpu")
        return

    def extract_attributes(self, image):
        """
        Extracts attributes (gender, hat, bag, upper clothing color, lower clothing color) from an image.

        Parameters:
        - image: Image input for attribute extraction.

        Returns:
        - list: List containing extracted features for each attribute.
        """
        answers = []
        feature_list = []

        try:
            # Process each predefined question and obtain answers
            questions = [
                self.processor(image, self.gender_question, return_tensors='pt').to(self.model.device),  # gender
                self.processor(image, self.hat_question, return_tensors='pt').to(self.model.device),     # hat
                self.processor(image, self.bag_question, return_tensors='pt').to(self.model.device),     # bag
                self.processor(image, self.upper_clothing_question, return_tensors='pt').to(self.model.device),  # upper color
                self.processor(image, self.lower_clothing_question, return_tensors='pt').to(self.model.device)  # lower color
            ]

            answers.append(questions)

            # Process answers and append to feature_list
            for answer in answers:
                for i, a in enumerate(answer):
                    outputs = self.model(**a)
                    logits = outputs.logits
                    idx = logits.argmax(-1).item()
                    feature_list.append(str(self.model.config.id2label[idx]).lower())

        except Exception as e:
            print("Error occurred --- ", e)

        return feature_list
