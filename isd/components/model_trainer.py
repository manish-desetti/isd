import os
import sys
import shutil
import zipfile
import yaml
from six.moves import urllib
from isd.utils.main_utils import read_yaml_file
from isd.logger import logging
from isd.exception import isdException
from isd.constant.training_pipeline import *
from isd.entity.config_entity import ModelTrainerConfig
from isd.entity.artifacts_entity import ModelTrainerArtifact

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def unzip_file(self, zip_path, extract_to):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        except Exception as e:
            raise isdException(e, sys)

    def remove_path(self, path):
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            raise isdException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            # Unzipping data
            logging.info("Unzipping data")
            self.unzip_file(DATA_INGESTION_S3_DATA_NAME, os.getcwd())
            self.remove_path(DATA_INGESTION_S3_DATA_NAME)

            # Prepare image path in txt file
            train_img_path = os.path.join(os.getcwd(), "images", "train")
            val_img_path = os.path.join(os.getcwd(), "images", "val")

            # Ensure directories exist
            if not os.path.exists(train_img_path):
                raise FileNotFoundError(f"The directory {train_img_path} does not exist")
            if not os.path.exists(val_img_path):
                raise FileNotFoundError(f"The directory {val_img_path} does not exist")

            # Training images
            with open('train.txt', "w") as f:
                img_list = os.listdir(train_img_path)
                for img in img_list:
                    f.write(os.path.join(train_img_path, img) + '\n')
            logging.info("Done Training images")

            # Validation images
            with open('val.txt', "w") as f:
                img_list = os.listdir(val_img_path)
                for img in img_list:
                    f.write(os.path.join(val_img_path, img) + '\n')
            logging.info("Done Validation images")

            # Download COCO starting checkpoint
            url = self.model_trainer_config.weight_name
            file_name = os.path.basename(url)
            urllib.request.urlretrieve(url, os.path.join("yolov7", file_name))

            # Training
            os.system(f"cd yolov7 && python train.py --batch {self.model_trainer_config.batch_size} --cfg cfg/training/custom_yolov7.yaml --epochs {self.model_trainer_config.no_epochs} --data data/custom.yaml --weights 'yolov7.pt'")

            # Copy best model
            shutil.copy("yolov7/runs/train/exp/weights/best.pt", "yolov7/best.pt")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            shutil.copy("yolov7/runs/train/exp/weights/best.pt", self.model_trainer_config.model_trainer_dir)

            # Clean up
            self.remove_path("yolov7/runs")
            self.remove_path("images")
            self.remove_path("labels")
            self.remove_path("classes.names")
            self.remove_path("train.txt")
            self.remove_path("val.txt")
            self.remove_path("train.cache")
            self.remove_path("val.cache")

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path="yolov7/best.pt")

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise isdException(e, sys)
