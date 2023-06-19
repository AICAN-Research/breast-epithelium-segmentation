# ep-segmentation
Segmentation of epithelial cells from Hematoxylin and Eosin stained slides using cytokeratin and pathologists annotations
as ground truth.

This repository includes the source code used for the model presented in (manuscript when published). 
The final model presented in the paper has been integrated in [FastPathology](https://github.com/AICAN-Research/FAST-Pathology).

### To use the code base you need:
**Disclamer:** The source code includes hard-coded solutions. To use on a new dataset, the code requires some
modifications, and the new dataset needs to be created, in which case you would need:

1. HE and CK images of tissue slides (.vsi)
3. Manual annotations of: 
   4. Benign and in situ lesions, annotated in HE images (.ome.tif)
   5. Cylinders (for tma slides) or areas (for wsi slides) to remove, annotated in CK images (.ome.tif)
   6. Triplet info (for tma slides), annotated in CK images (ome.tif)
4. QuPath v 3.2
5. [FastPathology](https://github.com/AICAN-Research/FAST-Pathology) (for inference)
6. Libraries in requirements.txt

## Create dataset:
To create the datasets you need five (for wsi) or six (for tma) images of each slide: he images (.vsi), ck images (.vsi), thresholded dab-channel (.tiff),
manual annotations of benign/in situ lesions (.ome.tif), annotations of areas to remove (.ome.tif), 
and triplet info (.ome.tif).

QuPath: 

All groovy scripts are run in QuPath. Open QuPath-project. Go to Automate -> Project scripts -> Script name. In script
editor go to Run -> Run for project. Select images to run script on.

### Create epithelial mask from ck images:
Create QuPath project and add ck images.

Threshold dab-channel in QuPath (uses pixel classifier dab_seg2.json):

```
dab_seg.groovy
```
Export dab-channel annotations to geojson:
```
geojson_exporter.groovy
```
Convert geojson to tiff:
```
convert_to_tiff.py
```
### Convert annotations to ome-tif:
Create QuPath projects for the different tasks (1-3). Add images and annotations. 

Convert manual annotations of benign/in situ lesions (1), cores to remove (2), and triplet info (3). to ome-tiff.
Remember to change annotation name depending on annotation category.
```
ome_tif_exporter.groovy
```
### Create datasets:

Create dataset from tma:
```
python /path/to/create_data_new.py 
```
Create dataset from wsi: 
```
python /path/to/create_data_wsi.py 
```
### Train model:
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
```
import_from_fastpathology.groovy
```

## Troubleshoot: 
### QuPath: 
Error when exporting annotations to geojson with QuPath script: 
Make sure "Include default imports" under "Run" in Script Editor is toggled.

## Acknowledgements
Code for AGU-Net from:
<pre>
@misc{bouget2021meningioma,
  title={Meningioma segmentation in T1-weighted MRI leveraging global context and attention mechanisms},
  author={David Bouget and André Pedersen and Sayied Abdol Mohieb Hosainey and Ole Solheim and Ingerid Reinertsen},
  year={2021},
  eprint={2101.07715},
  archivePrefix={arXiv},
  primaryClass={eess.IV}
}
</pre>
and 
<pre>
@article{10.3389/fmed.2022.971873,
  author={Pedersen, André and Smistad, Erik and Rise, Tor V. and Dale, Vibeke G. and Pettersen, Henrik S. and Nordmo, Tor-Arne S. and Bouget, David and Reinertsen, Ingerid and Valla, Marit},
  title={H2G-Net: A multi-resolution refinement approach for segmentation of breast cancer region in gigapixel histopathological images},
  journal={Frontiers in Medicine},
  volume={9},
  year={2022},
  url={https://www.frontiersin.org/articles/10.3389/fmed.2022.971873},
  doi={10.3389/fmed.2022.971873},
  issn={2296-858X}
}
</pre>