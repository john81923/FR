# **F**ace **R**ecognition Framework (KFR-Framework)

- Original NIR　dataset ```/mnt/sdc/craig/Guizhou_NIR_dataset_kfr```
- trainval NIR　dataset ```/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_0304_trainval```
- Original CASIA NIR　dataset ```/mnt/sdc/craig/CASIA_NIR/NIR```
- Testing CASIA NIR　dataset ```/mnt/sdc/craig/CASIA_NIR/NIR_0304```

## Goal

The purpose of this code is to add an extra head to the RGB-Pretrained FR model for increasing the accuracy on NIR images

## Getting Started

### Preparation
  - Generate trainval/test through FD-SSD and landmark pre-processing
    - ``` python extract_img.py ```
    - ``` python extract_img_casia.py```
  - Split trainval
    - ``` python util/split_data ```
  - Generate same/diff pairs .txt
    - ``` python pair_gen/gen_pairs_in_two_file_nir/casia.py ```
  - Set up the config. in conf/config.py
    - train directory:  ```IMAGE_ROOT  ```
    - model directory:  ```MODEL_STRUCTURE  ```
    - pair directory:  ``` PAIR_FOLDER_NAME  ```
    - val directory:  ```VALID_LIST ```

### Training
  - Need to adjust the log-dir for the checkpoint under  ``` bin/ktrain.py ```
  - ``` python -m kfr.bin.ktrain ```


### Validation/Testing

#### Preparation for Validation

  1. Your dataset shall be a root folder with different person label separated as:


  ```
  +-- testing_folder
  |   +-- person_1
  |   +-- person_2
  |   +-- ......
  ```
  2. These folders shall be 112 by 112 images after the process of  ``` FD-SSD and warp by landmark ``` (if not, please use python extract_img.py in ```Getting Started -> Preparation ```)


  3.  Generate same/diff pairs .txt,
    - ``` python pair_gen/gen_pairs_in_two_file_casia.py ```
    - you may set the totla pair with the parameter ```total_pairs_num ```
  4. open ``` ./inference/inference.py ```
    - For ```  parser.add_argument('--checkpoint) ```, you may set the default path to the checkpoint at this path (the pre-trained RGB model): [Checkpoint](http://192.168.200.1:8088/craig_hsin/kneron_face_recognition/tree/master/model/checkpoint)
    - For ``` parser.add_argument('--nir-head) ```, please download checkpoint from [nir_head](https://kneron.sharepoint.com/sites/Kneron-ALG/Shared%20Documents/Forms/AllItems.aspx?newTargetListUrl=%2Fsites%2FKneron%2DALG%2FShared%20Documents&viewpath=%2Fsites%2FKneron%2DALG%2FShared%20Documents%2FForms%2FAllItems%2Easpx&viewid=4763a651%2D204a%2D42d5%2D84a4%2D778efab0b1a6&id=%2Fsites%2FKneron%2DALG%2FShared%20Documents%2FModels%2Fface%5Frecognition%2FNIR)
    - For ``` same_pairs_path/diff_pairs_path  ```, please use the files generated by step 3
    - For VALID_LIST, please set to the root folder for your testing directory
    - There are two methods for loading the backbone (default case 2)
      - Case1: Load backbone from originalrgb's backbone

        ```
          model = torch.load(args.checkpoint)
          backbone = ResNet50Backbone(model)
          backbone.eval()

        ```
      - Case2: Load backbone and head from our nir_head checkpoint      
      ```
          checkpoint_state = torch.load(args.nir_head)
          backbone = checkpoint_state['backbone']
          print('load backbone')
      ```
  5. Run ``` python -m kfr.inference.inference ```
