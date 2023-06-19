# ep-segmentation
Segmentation of epithelial cells from Hematoxylin and Eosin stained slides using cytokeratin as ground truth.

## Create dataset and train model:
Create dataset from tma:
```
python /path/to/create_data_new.py 
```
Create dataset from wsi: 
```
python /path/to/create_data_wsi.py 
```
Train model (remember to change dataset name and pars argument values. Toggle/untoggle deep supervision/multiscale input/grad 
accumulation when creating model) Use only tma, only wsi or both datasets:
```
python /path/to/train.py 
```

## Train network from terminal: 

Create a screen session: 
```
screen -S session-name
```
Reenter existing screen session: 
```
screen -r session-name
```
Activate virtual environment: 
```
source environment-name/bin/activate
```
Start training: 
```
python /path/to/script.py
```
If you want to change arguments in script that has argparse (from default) then f.ex do:
```
python /path/to/script.py --batch_size 16 --learning_rate 0.001
```
Exit screen session: 
```
ctr ad
```
Check if in screen session: 
```
ctr at
```
## Evaluate model:
Create tma-level dataset for evaluation: 
```
python /path/to/create_tma_pairs.py
```
Evaluate model with: 
```
python /path/to/eval_tma_cylinders.py
```

## Run tf models in FastPathology: 
Convert model to onnx for FastPathology
```
pip install tf2onnx
python -m tf2onnx.convert --saved-model output/models/model_060223_122342_unet_bs_32/ --output output/converted_models/model_060223_122342_unet_bs_32.onnx
```
### Run model in FastPathology:
Add models from disk: Press "add models from disk" and find correct model and open

If pipeline already exists, press "Edit pipeline" and change model name to current model

## Import segmentation from FastPathology to QuPath:


## Troubleshoot: 
### QuPath: 
Error when exporting annotations to geojson with QuPath script: 
Make sure "Include default imports" under "Run" in Script Editor is toggled.
