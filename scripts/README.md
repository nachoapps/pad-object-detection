# Execute these commands to get started

Make sure your PYTHONPATH is configured per the object_detection docs (should
include tensorflow/research tensorflow/research/slim).

## Creating model/validation records

For the orb data:

```
python scripts/create_pascal_tf_record_generic.py \
--data_dir=images \
--annotations_dir=annotations \
--output_path=training/inputs/orb_data \
--label_map_path=training/inputs/orb_data/orb_label_map.pbtxt
```

For the full label set:

```
python scripts/create_pascal_tf_record_generic.py \
--data_dir=images \
--annotations_dir=annotations \
--output_path=training/inputs/pad_data \
--label_map_path=training/inputs/pad_data/pad_label_map.pbtxt
```

## Compiling the protos in research/object_detection

If you're on Windows, globs don't work, try this command:

```
# from tensorflow/models/research/
for %P in (object_detection\protos\*.proto) do "C:\tmp\protoc-3.5.1-win32\bin\protoc.exe" %P --python_out=.
```

On Linux, follow the command specified in the object_detection docs:

```
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```

## Training

This step takes at least a couple of hours on my PC to get good results. If
you're impatient, you can kill the job after an hour, run evaluation to
check the results (or if your PC is beefy enough, run evaluation concurrently).

SSD Inception:

```
TF_RESEARCH=/path/to/tensorflow/models/research

python $TF_RESEARCH/object_detection/train.py \
--logtostderr \
--pipeline_config_path=model_configs/configured/ssd_inception_v2_coco.config \
--train_dir=training/checkpoints/ssd_inception
```

Faster RCNN:

```
TF_RESEARCH=/path/to/tensorflow/models/research

python $TF_RESEARCH/object_detection/train.py \
--logtostderr \
--pipeline_config_path=model_configs/configured/faster_rcnn_resnet101_coco.config \
--train_dir=training/checkpoints/faster_rcnn
```

## Evaluation

```
TF_RESEARCH=/path/to/tensorflow/models/research

python $TF_RESEARCH/object_detection/eval.py \
--logtostderr \
--pipeline_config_path=model_configs/configured/ssd_inception_v2_coco.config \
--checkpoint_dir=training/checkpoints/ssd_inception \
--eval_dir=training/checkpoints/ssd_inception


# In another window
tensorboard --logdir=training/checkpoints/ssd_inception
```

## Exporting the frozen graph

Replace '12030' in the command below with the latest checkpoint you
generated.

```
TF_RESEARCH=/path/to/tensorflow/models/research

python $TF_RESEARCH/object_detection/export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path=model_configs/configured/ssd_inception_v2_coco.config \
--trained_checkpoint_prefix=training/checkpoints/ssd_inception/model.ckpt-12030 \
--output_directory=training/exported/ssd_inception
```

## Running the resulting graph on some images

This will pop up an annotated image and also print some stats to the terminal.

```
python scripts/run_inference.py \
--model_checkpoint=training/exported/ssd_inception/frozen_inference_graph.pb \ 
--label_map_path=training/inputs/orb_data/orb_label_map.pbtxt \
--input_image_path=images/bombs_2.png
```

## Other notes

### GPU OOM

If you want to run training and eval simultaneously, but your GPU memory keeps
getting exhausted, set this in your eval console to force it onto the CPU
instead:

```
export CUDA_VISIBLE_DEVICES=-1
```

### Visualization limits

By default only 20 boxes are drawn, so your eval images in tensorboard might
look like they're not performing well.

You can update evaluator.py:174 to print more detections like so:

```
eval_util.visualize_detection_results(
   <snip other stuff leave it alone>
    show_groundtruth=eval_config.visualization_export_dir,
    max_num_predictions=100)
```

If you're using the jupyter notebook `object_detection_ssd_inception.ipynb`
remember to do the same thing using `max_boxes_to_draw=100`.
