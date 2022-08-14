# Self-Supervised Vision Transformers with DINO

## Introduction 

While the volume of data collected for vision based tasks has increased exponentially in recent times, annotating all unstructured datasets is practically impossible. 

`DINO` which is based `self supervised learning`, does not require large amounts of labelled data to achieve state of the art results on segmentation tasks, unlike traditional supervised methods.

![](https://github.com/TanyaChutani/DINO_Tf2.x/blob/f59fd89e52d25f1f8fa4915d4febd9991d664749/assets/dino.png)

To be specific, DINO is `self distillation` with `NO labels`, wherein 2 models (teacher and student) are used. While they have the same model architecture, the teacher model is trained using an exponentially weighted average of the student model's parameters.

This technique was introduced in the research paper by Facebook AI titled `"Emerging Properties in Self-Supervised Vision Transformers"`.

Visualization of the generated attention maps highlight that DINO can learn class specific features automatically, which help us generate accurate segmentation maps without the need of labelled data in vision based tasks.

## Commands to download the data
```
gdown --id 1Lw_XPTbkoHUtWpG4U9ByYIwwmLlufvyj
unzip PASCALVOC2007.zip
```

## Train model
```
git clone https://github.com/TanyaChutani/DINO_Tf2.x.git
pip install -r requirements.txt
python main.py
```

## Results
![](https://github.com/TanyaChutani/DINO_Tf2.x/blob/48ec43e4bb98dad4088047df4f8af25abb4211f1/result/000011.png)
![](https://github.com/TanyaChutani/DINO_Tf2.x/blob/48ec43e4bb98dad4088047df4f8af25abb4211f1/result/000027.png)
![](https://github.com/TanyaChutani/DINO_Tf2.x/blob/48ec43e4bb98dad4088047df4f8af25abb4211f1/result/000478.png)
![](https://github.com/TanyaChutani/DINO_Tf2.x/blob/48ec43e4bb98dad4088047df4f8af25abb4211f1/result/000495.png)
![](https://github.com/TanyaChutani/DINO_Tf2.x/blob/48ec43e4bb98dad4088047df4f8af25abb4211f1/result/000655.png)
![](https://github.com/TanyaChutani/DINO_Tf2.x/blob/48ec43e4bb98dad4088047df4f8af25abb4211f1/result/000883.png)
![](https://github.com/TanyaChutani/DINO_Tf2.x/blob/48ec43e4bb98dad4088047df4f8af25abb4211f1/result/001422.png)
![](https://github.com/TanyaChutani/DINO_Tf2.x/blob/48ec43e4bb98dad4088047df4f8af25abb4211f1/result/001698.png)
![](https://github.com/TanyaChutani/DINO_Tf2.x/blob/1bbd765201339c53b2d84929f829def2bb4c0da1/result/001764.png)
![](https://github.com/TanyaChutani/DINO_Tf2.x/blob/48ec43e4bb98dad4088047df4f8af25abb4211f1/result/001823.png)
![](https://github.com/TanyaChutani/DINO_Tf2.x/blob/48ec43e4bb98dad4088047df4f8af25abb4211f1/result/001848.png)
![](https://github.com/TanyaChutani/DINO_Tf2.x/blob/48ec43e4bb98dad4088047df4f8af25abb4211f1/result/001850.png)
![](https://github.com/TanyaChutani/DINO_Tf2.x/blob/48ec43e4bb98dad4088047df4f8af25abb4211f1/result/001959.png)

