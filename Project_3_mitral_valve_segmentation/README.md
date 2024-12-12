# Mitral Valve Segmentation Project

## Workflow

This project focuses on the segmentation of the mitral valve using advanced machine learning techniques. The workflow is divided into several key stages:


1. **Train data preprocessing**
    - Run `reshape.py` to extract and reshape the train data
    - Expert frames will be cropped or padded to a size $700\times 700$
    - Amateur frames will be upsized to $512\times 512$ and then padded to $700\times 700$
    - The reshaped frames and masks is saved in `combined_train_data.pkl`

2. **Denoising, contrast enhancement and mask smoothing**
    - Run final_processing.py to apply denoising, contrast enhancement and mask smoothing 
      to the test data
    - Choose whether to load previously trained DRUNet model checkpoints for denoising
    - The final processed data will be saved as `final_train_data.pkl`

3. **UNet model**
    - The UNet model used for the segmentation task is contained in `model.py`

5. **Model training**
    - Run `training.py` to train the UNet model on the train data
    - Select whether to load the previously trained model checkpoints

6. **Preparing test data**
    - Run `prepare_test.py` to prepare test data for prediction
    - The prepared test data will be saved as `final_test_data` containing 
      test_names and test_frames

7. **Make submission**
    - Run `make_submission` to make the `submission.csv` file