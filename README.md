# Team HelloWorld System Code Guideline

## 1.Data Preparation

**All of the dev&test data need to resample to 16K.**

You should modify dev&test data paths in csv files, including:

- speechbrain-develop/recipes/VoxCeleb/SpeakerRec/results/MSV_22/I_MSV/1986/csv_files/*.csv
- speechbrain-develop/recipes/VoxCeleb/SpeakerRec/results/MSV_22/I_MSV/test/csv_files/*.csv
- speechbrain-develop/recipes/VoxCeleb/SpeakerRec/results/MSV_22/I_MSV/private_test/csv_files/*.csv

## 2. Training Process

You can run the following script to train the model, and you can modify the hyperparameters in the `hparams/train_MSV.yaml` and `hparams/train_MSV_weight_regular.yaml`.

````python
python train_MSV.py hparams/train_MSV.yaml
python train_MSV_weight_regular.py hparams/train_MSV_weight_regular.yaml
````

## 3.Evaluation Process

You can run the following script to train the model, and you can modify the hyperparameters in the `hparams/verification_MSV.yaml`. Note that you must modify the value of `pretrain_path` to the path of the trained model. Finally, you can get scores at `output_folder`.

```
python speaker_verification_MSV.py hparams/verification_MSV.yaml
```