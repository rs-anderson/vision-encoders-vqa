from collections import defaultdict
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
from vqa_data import VQA2
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from paths import BASEDIR
import wandb


def default_collate(batch):
    """
    Override `default_collate` https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader

    Reference:
    def default_collate(batch) at https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
    https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
    https://github.com/pytorch/pytorch/issues/1512

    We need our own collate function that wraps things up (imge, mask, label).

    In this setup,  batch is a list of tuples (the result of calling: img, mask, label = Dataset[i].
    The output of this function is four elements:
        . data: a pytorch tensor of size (batch_size, c, h, w) of float32 . Each sample is a tensor of shape (c, h_,
        w_) that represents a cropped patch from an image (or the entire image) where: c is the depth of the patches (
        since they are RGB, so c=3),  h is the height of the patch, and w_ is the its width.
        . mask: a list of pytorch tensors of size (batch_size, 1, h, w) full of 1 and 0. The mask of the ENTIRE image (no
        cropping is performed). Images does not have the same size, and the same thing goes for the masks. Therefore,
        we can't put the masks in one tensor.
        . target: a vector (pytorch tensor) of length batch_size of type torch.LongTensor containing the image-level
        labels.
    :param batch: list of tuples (img, mask, label)
    :return: 3 elements: tensor data, list of tensors of masks, tensor of labels.
    """
    image = [item[0] for item in batch]
    question = [item[1] for item in batch]
    answer = [item[2] for item in batch]
    question_id = [item[3] for item in batch]

    return image, question, question_id, answer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

wandb.init(project="VQA2.0-baseline")
config = wandb.config

dataset_config = dict(
    versionType = "v2_",
    taskType = "OpenEnded",
    dataType = "mscoco",
    dataSubType = "val2014",
    split = "val"
)

ans_class_idxs_all = []
question_ids_all = []

print("Loading data...")
vqa_dataset = VQA2(**dataset_config)
dataloader = DataLoader(vqa_dataset, batch_size=128, shuffle=False, num_workers=8, collate_fn=default_collate)

print("Loading models...")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device)

wandb.watch(model)

print("Starting inference...")
for i, (images, questions, question_ids, answers) in enumerate(tqdm(dataloader)):
    
    # prepare inputs
    encoding = processor(images, questions, return_tensors="pt", padding=True).to(device)

    # forward pass
    outputs = model(**encoding)
    ans_class_idxs_batch = outputs.logits.argmax(-1)
    ans_class_idxs_all.append(ans_class_idxs_batch)
    question_ids_all.append(question_ids)

results = defaultdict(list)

question_ids_all = [question_id for question_id_list in question_ids_all for question_id in question_id_list]
ans_class_idxs_all = [ans_class_idx for ans_class_idxs_list in ans_class_idxs_all for ans_class_idx in ans_class_idxs_list.tolist()]

results = [
    {
        "answer": str(model.config.id2label[class_id]),
        "question_id": q_id
    }
    for class_id, q_id in zip(ans_class_idxs_all, question_ids_all)
]

json_results = json.dumps(results)
with open(BASEDIR / "Results" / f"{dataset_config['dataSubType']}.json", "w") as results_file:
    json.dump(results, results_file)