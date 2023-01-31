# ep-segmentation
Segmentation of epithelial cells from Hematoxylin and Eosin stained slides using cytokeratin as ground truth.

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
## Troubleshoot: 
### QuPath: 
Error when exporting annotations to geojson with QuPath script: 
Make sure "Include default imports" under "Run" in Script Editor is toggled.
