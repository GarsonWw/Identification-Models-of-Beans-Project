## 该文件夹是用来存放训练数据的目录
### 使用步骤如下：
* （1）在data_set文件夹下创建新文件夹"beans_data"
* （2）在beans_data文件夹下创建新文件夹"beans_photo"存放分类文件夹
* （4）执行"split_data.py"脚本自动将数据集划分成训练集train和验证集val    

```
├── beans_data   
       ├── beans_photos（解压的数据集文件夹）
                ├── pod（豆荚有籽）  
                └── no pod（豆荚无籽） 
       ├── train（生成的训练集）  
       └── val（生成的验证集） 
```
