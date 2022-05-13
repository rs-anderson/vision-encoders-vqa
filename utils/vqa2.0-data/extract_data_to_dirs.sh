#!/bin/bash

mkdir {Annotations, Images, Questions, Images/mscoco}


# images
unzip train2014.zip -d Images/mscoco
rm train2014.zip

unzip val2014.zip -d Images/mscoco
rm val2014.zip

unzip test2015.zip -d Images/mscoco
rm test2015.zip


# questions
unzip v2_Questions_Train_mscoco.zip -d Questions
rm v2_Questions_Train_mscoco.zip

unzip v2_Questions_Val_mscoco.zip -d Questions
rm v2_Questions_Val_mscoco.zip

unzip v2_Questions_Test_mscoco.zip -d Questions
rm v2_Questions_Test_mscoco.zip


# annotations
unzip v2_Annotations_Train_mscoco.zip -d Annotations
rm v2_Annotations_Train_mscoco.zip

unzip v2_Annotations_Val_mscoco.zip -d Annotations
rm v2_Annotations_Val_mscoco.zip