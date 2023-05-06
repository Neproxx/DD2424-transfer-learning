# DD2424-Transfer-Learning

Group project in the course DD2424 at KTH in which we explore representation learning with domain-specific unlabeled data to boost performance on a downstream task where the model is fine-tuned on labeled from the same domain data.

## Base Model

A variant of the ConvNeXt architecture (see [ConvNeXt](https://github.com/facebookresearch/ConvNeXt#results-and-pre-trained-models)) [2].

## Dataset Information

In this project, we use the VGGFace2 dataset [1] that contains images of faces. We preprocessed the data by cropping and resizing the images to a resolution of 224x224. For each of the 9131 identities in the train and test set, we selected up to 60 images each. For most of the identities, more than 60 images were available. The compressed, preprocessed data can be downloaded from this [link](https://drive.google.com/drive/folders/1pzyStORojl4jei89uto6lXZEMTuzoKSQ?usp=sharing) and should be placed in the "./data" folder. A pytorch Dataset class for easy use of the dataset is implemented in the file "dataset.py" and a usage example with the [ConvNeXt-tiny](https://huggingface.co/facebook/convnext-tiny-224) model is demonstrated in the file "./examples/dataset_use.py". The dataset offers four different versions that can be selected for different tasks:

1. "original": The original dataset with the original images and labels.
2. "rotated": The dataset with images rotated by either 0, 90, 180 or 270 degrees. The labels represent the rotation of the image in onehot encoding.
3. "grayscale": The dataset with grayscale images. As labels, it contains the original but flattened images.
4. "rotated_grayscale": A combination of the two above. The images are rotated and grayscale. The labels of each approaches are concatenated with the four values for the rotation coming first.

## References

[1] Cao, Qiong, et al. "VGGFace2: A dataset for recognising faces across pose and age." 2018 13th IEEE international conference on automatic face & gesture recognition (FG 2018). IEEE, 2018.

[2] Liu, Zhuang, et al. "A convnet for the 2020s." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
