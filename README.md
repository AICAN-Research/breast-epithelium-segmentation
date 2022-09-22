# ep-segmentation
Segmentation of epithelial cells from Hematoxylin and Eosin stained slides using cytokeratin as ground truth.

## Train network from terminal: 

Create a screen session: 
```
screen -S "name of session"
```
Activate virtual environment: 
```
source "name of environment"/bin/activate
```
Start training: 
```
python "name of script.py"
```
If you want to change arguments in script that has argparse (from default) then f.ex do:
```
python "name of script.py" --batch_size 16 --learning_rate 0.001
```
Exit screen session: 
```
ctr ad
```
