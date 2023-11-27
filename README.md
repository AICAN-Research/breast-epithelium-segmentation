# [Breast Epithelium Segmentation](https://github.com/AICAN-Research/breast-epithelium-segmentation#breast-epithelium-segmentation)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2311.13261-firebrick?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2311.13261)
[![Demo](https://img.shields.io/badge/demo-FastPathology-blue?logo=arxiv&logoColor=blue)](https://github.com/AICAN-Research/FAST-Pathology)

This repository includes the source code related to the preprint [_"Immunohistochemistry guided segmentation of benign epithelial cells, in situ lesions, and invasive epithelial cells in breast cancer slides"_](https://arxiv.org/abs/2311.13261).

## [Summary](https://github.com/AICAN-Research/breast-epithelium-segmentation#summary)

> An immunohistochemistry (IHC) restaining technique was used to facilitate the annotation process of both whole slide images (WSIs) and tissue microarrays (TMAs). An algorithm was then used to post-process the cytokeratin (CK) images to produce binary segmentations and form HE-CK pairs. From these proposals, a pathologist distinguished between invasive and epithelial cells and _in situ_ lesions. A convolutional neural network was then trained to perform semantic segmentation. The model was made more robust through tailored data augmentation techniques, utilizing a multi-scale network architecture, and introducing patches from WSIs in addition to the TMA pairs. The final model was then made available in [FastPathology](https://ieeexplore.ieee.org/document/9399433).

![bilde_github](https://github.com/AICAN-Research/breast-epithelium-segmentation/assets/89521132/e7a13473-a7c5-43c1-83ad-e219c7ec9ec7)

## [Testing model](https://github.com/AICAN-Research/breast-epithelium-segmentation#testing-model)

<details open>
<summary>

### [With FastPathology](https://github.com/AICAN-Research/breast-epithelium-segmentation#with-fastpathology)</summary>

The trained model is available in the [FastPathology](https://github.com/AICAN-Research/FAST-Pathology) software.
Select "Download models & pipelines" from the main menu and look for the "Breast Epithelium Segmentation" and press download.
Then you can load your own WSI data and apply the model without doing any programming.

</details>

<details open>
<summary>

### [From command line with pyFAST](https://github.com/AICAN-Research/breast-epithelium-segmentation#from-command-line-with-pyfast)</summary>

You can run the model on your own WSI data from the command line after installing [FAST for Python (aka pyfast)](https://fast.eriksmistad.no/install.html).

```bash
pip install pyfast

runPipeline --datahub breast-epithelium-segmentation --file /path/to/WSI
```

</details>

<details>
<summary>

### [From Python with pyFAST](https://github.com/AICAN-Research/breast-epithelium-segmentation#from-python-with-pyfast)</summary>
First install [FAST for Python (aka pyfast)](https://fast.eriksmistad.no/install.html).
```bash
pip install pyfast
```
> **Note:** There are some requirements to be installed for Ubuntu Linux and macOS (see [here](https://fast.eriksmistad.no/install-ubuntu-linux.html) and [here](https://fast.eriksmistad.no/install-mac.html), respectively). Windows should work out of the box.

Then from Python you can do:
```python
import fast

pipeline = fast.Pipeline.fromDataHub('breast-epithelium-segmentation', {'file': '/path/to/your/WSI'})
pipeline.run()
```
This will visualize the output of the model. 

You can also export the segmentation to a pyramidal TIFF like so:
```python
import fast

pipeline = fast.Pipeline.fromDataHub('breast-epithelium-segmentation', {'file': '/path/to/your/WSI'})
pipeline.parse(visualization=False)
output = pipeline.getPipelineOutputData('segmentation')
fast.TIFFImagePyramidExporter.create('segmentation.tiff')\
	.connect(output)\
	.run()
```

See the [documentation for more info on how to work with WSI data with pyFAST](https://fast.eriksmistad.no/python-tutorial-wsi.html).

</details>

## [Training preliminaries](https://github.com/AICAN-Research/breast-epithelium-segmentation#training-preliminaries)

**Disclaimer:** The source code includes hard-coded solutions. To train the model on a new dataset, the code requires
modifications, and the new dataset needs to be created, in which case you would need:

1. HE and CK images of tissue slides (`.vsi`).
2. Manual annotations of: 
   - Benign and *in situ* lesions, annotated in HE images (`.ome.tif`).
   - Cylinders (for tissue micro array (TMA) slides) or areas (for whole slide image (WSI) slides) to remove, annotated in CK images (`.ome.tif`).
   - Triplet info (for TMA slides), annotated in CK images (`.ome.tif`).
3. [QuPath v3.2](https://github.com/qupath/qupath) for annotations and masks.

Then you can clone the repo, create a virtual environment, and install dependencies by running:

```
git clone https://github.com/AICAN-Research/breast-epithelium-segmentation.git
cd breast-epithelium-segmentation/
python -m venv venv/
source venv/bin/activate
pip install -r requirements.txt
```

## [Create dataset](https://github.com/AICAN-Research/breast-epithelium-segmentation#create-dataset)

To create the datasets you need five (for WSI) or six (for TMA) images of each slide: HE images (`.vsi`), CK images (`.vsi`), thresholded DAB-channel (`.tiff`), manual annotations of benign/*in situ* lesions (`.ome.tif`), annotations of areas to remove (`.ome.tif`), and triplet info (`.ome.tif`).
   
### [QuPath](https://github.com/AICAN-Research/breast-epithelium-segmentation#qupath)
   
All groovy scripts are run in QuPath. To run a script on a batch of WSIs:

* Open QuPath-project.
* Select Project scripts.
* Select Script name.
* In script editor, click `Run for Project`.
* Select images to run script on.
* Click `Run` to start running the script on the selected WSIs.
   
The code for dataset creation from TMAs assumes a folder structure like below:
```
└── path/to/data/
   └── cohorts/
      ├── cohort_1/
      ├── cohort_2/
      ├── [...]
      └── cohort_n/
```

<details>
<summary>

### [Create epithelial mask from CK images](https://github.com/AICAN-Research/breast-epithelium-segmentation#create-epithelial-mask-from-ck-images)</summary>

* Create QuPath project and add CK images.

* Threshold DAB-channel in QuPath (uses pixel classifier `dab_seg2.json`):
```
dab_seg.groovy
```

* Export DAB-channel annotations to GeoJSON:
```
geojson_exporter.groovy
```

* Convert GeoJSON to TIFF:
```
convert_to_tiff.py
```
</details>

<details>
<summary>

### [Convert annotations to OME-TIFF](https://github.com/AICAN-Research/breast-epithelium-segmentation#convert-annotations-to-ome-tiff)</summary>

* Create QuPath projects for the different tasks (1-3).
* Add images and create annotations. 
* Convert manual annotations of benign/*in situ* lesions (1), cores to remove (2), and triplet info (3) to OME-TIFF.
* Run this script through the QuPath script editor: `ome_tif_exporter.groovy`

_**NOTE:**_ Remember to change annotation name depending on annotation category.
</details>

<details>
<summary>

### [Create datasets](https://github.com/AICAN-Research/breast-epithelium-segmentation#create-datasets)</summary>

Split data into train, validation, and test sets:

* For TMA:
```
python /path/to/divide_data.py 
```

* For WSI:
```
python /path/to/divide_data_wsi.py 
```

Create train/val dataset:

* From TMA:
```
python /path/to/create_data_tma.py 
```

* From WSI:
```
python /path/to/create_data_wsi.py 
```
</details>

<details open>
<summary>

## [Train model](https://github.com/AICAN-Research/breast-epithelium-segmentation#train-model)</summary> 

Remember to change dataset name and pairs argument values. Toggle/untoggle deep supervision/multiscale input/grad 
accumulation when creating model. Use only TMA, only WSI, or both datasets.

Assuming that the virtual environment is setup and all dependencies are installed, you can start training using the default configuration by running:
```
python /path/to/script.py
```

It is possible to adjust some of the hyperparameters during experiments through the command line. To see the default configuration and what is possible to set, run:
```
python /path/to/script.py -h
```

An example changing the `batch size` and `learning rate` can be seen below:
```
python /path/to/script.py --batch_size 16 --learning_rate 0.001
```
</details>

<details>
<summary>

## [Evaluate model](https://github.com/AICAN-Research/breast-epithelium-segmentation#evaluate-model)</summary> 

Create TMA-level dataset for evaluation: 
```
python /path/to/create_tma_pairs.py
```

Evaluate model on cylinder-level with:
```
python /path/to/eval_quantitatively.py
```

_**NOTE:**_ Make sure that the correct model and dataset are used.

Evaluate model on histological subtype/grade with:
```
python /path/to/eval_histologic_subtype.py
```

_**NOTE:**_ Make sure that the correct model and dataset are used. For this evaluation you would need an external data file containing the histological subtype/grade.

</details>

<details>
<summary>

## [Deploy custom model in FastPathology](https://github.com/AICAN-Research/breast-epithelium-segmentation#train-model)</summary> 

Given that you have trained your own model, you may want to use [FastPathology](https://github.com/AICAN-Research/FAST-Pathology) to enable the model to be used through a simple graphical user interface (GUI).

1. Convert pretrained model to the ONNX format:
```
pip install tf2onnx
python -m tf2onnx.convert --saved-model /path/to/saved_model/ --output /path/to/converted/model.onnx --opset 13
```

2. To add models from disk, open FastPathology and click `"Add models from disk"` on the bottom left. Then find the model stored in the appropriate format (e.g., `.onnx`) and click `open` to start importing it.

3. You can then import the FAST Pipeline file (`multiclass_ep_seg_agunet.fpl`) made available under `pipelines/` in this repository, by clicking `Import pipeline` from the FastPathology user interface and doing the same steps as for model importing. 

4. In order to make the FPL file compatible with your custom model, you will need to change the model name in the FPL file. You can do this by choosing the pipeline from the `Process` widget, clicking `"Edit pipeline` and changing the model name you chose in step 1 when converting it (see `NeuralNetwork` process object in the FPL).

</details>

<details>
<summary>

## [Import segmentation from FastPathology to QuPath](https://github.com/AICAN-Research/breast-epithelium-segmentation#Import-segmentation-from-FastPathology-to-QuPath)</summary> 

This can be performed by using this groovy script:
```
import_from_fastpathology.groovy
```

See the script header for more details on how to use it.

</details>

<details>
<summary>

## [Troubleshooting](https://github.com/AICAN-Research/breast-epithelium-segmentation#troubleshooting)</summary> 

### [QuPath export error](https://github.com/AICAN-Research/breast-epithelium-segmentation#qupath-export-error)
	
**Q:** Error when exporting annotations to GeoJSON with QuPath script

**A:** Make sure `Include default imports` under `Run` in Script Editor is enabled.

</details>

## [How to cite](https://github.com/AICAN-Research/breast-epithelium-segmentation#how-to-cite)

Please, cite our research article if you found this repository useful:
```
@misc{hoibo2023immunohistochemistry,
    title={Immunohistochemistry guided segmentation of benign epithelial cells, in situ lesions, and invasive epithelial cells in breast cancer slides}, 
    author={Maren Høibø and André Pedersen and Vibeke Grotnes Dale and Sissel Marie Berget and Borgny Ytterhus and Cecilia Lindskog and Elisabeth Wik and Lars A. Akslen and Ingerid Reinertsen and Erik Smistad and Marit Valla},
    year={2023},
    eprint={2311.13261},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```

## [Acknowledgements](https://github.com/AICAN-Research/breast-epithelium-segmentation#acknowledgements)

Thank you to @petebankhead and the [QuPath](https://qupath.github.io/) team for their invaluable support in implementing the groovy scripts.

AGU-Net implementation for histopathological image segmentation:
```
@article{pedersen2022h2gnet,
   title={{H2G-Net: A multi-resolution refinement approach for segmentation of breast cancer region in gigapixel histopathological images}},
   author={Pedersen, André and Smistad, Erik and Rise, Tor V. and Dale, Vibeke G. and Pettersen, Henrik S. and Nordmo, Tor-Arne S. and Bouget, David and Reinertsen, Ingerid and Valla, Marit},
   journal={Frontiers in Medicine},
   volume={9},
   year={2022},
   url={https://www.frontiersin.org/articles/10.3389/fmed.2022.971873},
   doi={10.3389/fmed.2022.971873},
   issn={2296-858X}
}
```

Which was adapted from the 3D AGU-Net architecture proposed in:
```
@article{bouget2021agunet,
   title={{Meningioma Segmentation in T1-Weighted MRI Leveraging Global Context and Attention Mechanisms}},
   author={Bouget, David and Pedersen, André and Hosainey, Sayied Abdol Mohieb and Solheim, Ole and Reinertsen, Ingerid},
   journal={Frontiers in Radiology},
   volume={1},
   year={2021},
   url={https://www.frontiersin.org/articles/10.3389/fradi.2021.711514},
   doi={10.3389/fradi.2021.711514},
   issn={2673-8740},
}

```
