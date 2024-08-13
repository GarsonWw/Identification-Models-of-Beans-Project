# Images-identification-models
# Duties of beans classification and partition


In English：
💡 **To Reproduce the Original and Improved Models**
💡 **The reproduction of the original model can be referenced in the README.md file located in the corresponding model's folder**

1. If using your own dataset, arrange it according to the structure in the `beans_data` folder (i.e., each class corresponds to a folder). Also, set the `num_classes` in the training and prediction scripts to match the number of classes in your dataset.
2. In the `train.py` script, set the `--data-path` to the absolute path of the unzipped `beans_photos` folder.
3. Download the pre-trained weights. Each model in `model.py` provides a download link for the pre-trained weights. Download the weights corresponding to the model you are using.
4. In the `train.py` script, set the `--weights` parameter to the path of the downloaded pre-trained weights.
5. Once you've set the dataset path (`--data-path`) and the pre-trained weights path (`--weights`), you can start training using the `train.py` script (during training, a `class_indices.json` file will be automatically generated).
6. In the `predict.py` script, import the same model used in the training script, and set the `model_weight_path` to the path of the trained model weights (by default, they are saved in the `weights` folder).
7. In the `predict.py` script, set the `img_path` to the absolute path of the image you want to predict.
8. Once you've set the `model_weight_path` and the `img_path`, you can use the `predict.py` script to make predictions.
9. After training, you can write a `train_log.txt` script, similar to `swintransformer-garson-master\train_log.txt`, to record the training logs for experimental analysis.
💡 **After Training:** The `weight` folder stores the results of the model training. You can place images in the `img_test` folder for prediction.


简体中文：
💡 **原模型和改进模型均可参考以下方式复现**
💡 **原模型的复现参考文件位于对应模型文件下的README.md文件**

1.如果使用自己的数据集，请按照beans_data文件结构进行摆放(即一个类别对应一个文件夹)，并且将训练以及预测脚本中的num_classes设置成你自己数据的类别数
2.在train.py脚本中将--data-path设置成解压后的beans_photos文件夹绝对路径
3.下载预训练权重，在model.py文件中每个模型都有提供预训练权重的下载地址，根据自己使用的模型下载对应预训练权重
4.在train.py脚本中将--weights参数设成下载好的预训练权重路径
5.设置好数据集的路径--data-path以及预训练权重的路径--weights就能使用train.py脚本开始训练了(训练过程中会自动生成class_indices.json文件)
6.在predict.py脚本中导入和训练脚本中同样的模型，并将model_weight_path设置成训练好的模型权重路径(默认保存在weights文件夹下)
7.在predict.py脚本中将img_path设置成你自己需要预测的图片绝对路径
8.设置好权重路径model_weight_path和预测的图片路径img_path就能使用predict.py脚本进行预测了
9.训练结束后可以自编写train_log.txt脚本如swintransformer-garson-master\train_log.txt所示，记录训练日志进行实验分析
💡 **训练后：weight文件夹存放模型训练结果  可将图片放入img_test文件进行predict。**

