# TensorBoard Metrics

Training can export metrics that are compatible with TensorBoard. To enable
logging, set the `tensorboard_path` field inside the `export` section of the
root configuration. The trainer writes TensorBoard event files into the
specified directory while training runs.

```protobuf
export {
  path: "checkpoints"
  tensorboard_path: "logs/train"
}
```

## Recorded data

The trainer records the following statistics:

* **Per step**
  * Learning rate that was applied for the step.
  * Weighted loss value returned by the loss function.
  * Unweighted loss components for value, policy, and moves left heads.
  * Global gradient norm after clipping.
* **Per epoch**
  * Histogram of all model weights together with mean, standard deviation,
    minimum, and maximum scalars.
  * Configuration scalars: batch size, steps per network, and chunks per
    network.

## Viewing the dashboard

Launch TensorBoard and point it to the directory configured above:

```bash
tensorboard --logdir logs/train
```

The dashboard contains grouped scalars for configuration parameters, losses,
and gradient norms, and a histogram of the weight distribution after each
training epoch.
