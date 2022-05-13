import wget


training_annotations_url = 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip'
validation_annotations_url = 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip'

training_questions_url = 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip'
validation_questions_url = 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip'
test_questions_url = 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip'

training_images_url = 'http://images.cocodataset.org/zips/train2014.zip'
validation_images_url = 'http://images.cocodataset.org/zips/val2014.zip'
test_images_url = 'http://images.cocodataset.org/zips/test2015.zip'

urls = [
    training_annotations_url,
    validation_annotations_url,
    training_questions_url,
    validation_questions_url,
    test_questions_url,
    training_images_url,
    validation_images_url,
    test_images_url
]

for url in urls:
    wget.download(url)