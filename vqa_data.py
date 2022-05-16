import sys
from paths import BASEDIR, VQA_API_DIR

sys.path.insert(0, VQA_API_DIR.as_posix())
from vqa import VQA

import torch
from torch.utils.data import Dataset
from skimage import io

import json


class VQA2(Dataset):
    def __init__(
        self,
        versionType: str,
        taskType: str,
        dataType: str,
        dataSubType: str,
        split: str,
    ):
        super(Dataset, self).__init__()

        self.split = split

        quesFile = (
            BASEDIR
            / "Questions"
            / "{}{}_{}_{}_questions.json".format(
                versionType,
                taskType,
                dataType,
                dataSubType,
            )
        )

        self.questions = json.load(open(quesFile, "r"))["questions"]
        self.img_dir = BASEDIR / "Images" / f"{dataType}" / f"{dataSubType}"

        self.img_partial_name = f"COCO_{dataSubType}"

        if self.split in ["train", "val"]:
            annFile = (
                BASEDIR
                / "Annotations"
                / "{}{}_{}_annotations.json".format(
                    versionType,
                    dataType,
                    dataSubType,
                )
            )
            self.annotations = json.load(open(annFile, "r"))["annotations"]
        else:
            self.annotations = None

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        input_info = self.questions[idx]

        img_id = input_info["image_id"]
        question = input_info["question"]
        question_id = input_info["question_id"]

        if self.split in ["train", "val"]:
            annotation = self.annotations[idx]
            answer = annotation["multiple_choice_answer"]

            assert annotation["image_id"] == img_id, "image IDs don't match"
            assert annotation["question_id"] == question_id, "question IDs don't match"

        if self.split == "test":
            answer = []

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_fname = f"{self.img_partial_name}_{str(img_id).zfill(12)}.jpg"
        image = io.imread(self.img_dir / img_fname, pilmode="RGB")

        return image, question, answer, question_id
