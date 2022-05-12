import sys
from paths import BASEDIR, VQA_API_DIR
sys.path.insert(0, VQA_API_DIR.as_posix())
from vqa import VQA
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from skimage import io, transform

class VqaDataset(Dataset):
     def __init__(
        self,
        versionType = "v2_",
        taskType = "OpenEnded",
        dataType = "mscoco",
        dataSubType = "val2014",
     ):
        super(Dataset, self).__init__()

        annFile = BASEDIR / "Annotations" / "{}{}_{}_annotations.json".format(
            versionType,
            dataType,
            dataSubType,
        )

        quesFile = BASEDIR / "Questions" / "{}{}_{}_{}_questions.json".format(
            versionType,
            taskType,
            dataType,
            dataSubType,
        )

        self.img_dir = BASEDIR / "Images" / f"{dataType}" / f"{dataSubType}"

        self.vqa = VQA(annFile, quesFile)
        self.ann_ids = self.vqa.qqa.keys()
        self.img_partial_name = f"COCO_{dataSubType}"

        
     def __len__(self):
        return len(self.vqa.qqa.keys())
        
     def __getitem__(self, idx):
        input_info = self.vqa.questions['questions'][idx]

        img_id = input_info['image_id']
        question = input_info['question']
        question_id = input_info['question_id']
        
        annotation = self.vqa.dataset['annotations'][idx]
        answer = annotation["multiple_choice_answer"]

      #   if question_id == 86000:
      #      import ipdb; ipdb.set_trace()

        assert annotation["image_id"] == img_id, "image IDs don't match"
        assert annotation["question_id"] == question_id, "question IDs don't match"

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_fname = f"{self.img_partial_name}_{str(img_id).zfill(12)}.jpg"
      #   image = Image.open(self.img_dir / img_fname)
        image = io.imread(self.img_dir / img_fname, pilmode='RGB')
        if len(image.shape) != 3:
           import ipdb; ipdb.set_trace()

        return image, question, answer, question_id
