## Neural Operator method cost accuracy trade-off
### Training neural operator
```bash
    sbatch fno_train_parallel.py
```

### Cost accuracy trade-off
Estimate the cost (floating point flops and CPU/GPU time) and accuracy (relative error) with different downsampled meshes, save the data
cost_accuracy_neural_operator() in fno_darcy_solver.py

Plot solutions with different downsampled meshes, plot cost accuracy trade-off curves 
cost_accuracy_plot() in multigrid_darcy_solver.py

