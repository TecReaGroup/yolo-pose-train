"""Verify the packaged model and measure endpoint error in original image pixels."""
import argparse
import csv
import json
import logging
import os
from pathlib import Path

from pen_pose.logging import configure_logging
from pen_pose.path import DATASET_DIR, MODEL_DIR, PROJECT_ROOT, RUN_DIR, TEMP_DIR

os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('YOLO_CONFIG_DIR', str(TEMP_DIR/'ultralytics'))

import numpy as np
from ultralytics import YOLO

from pen_pose.training.device import select_device

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Measure endpoint accuracy for a packaged model and validation split."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATASET_DIR)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    device = select_device(args.device)
    model = YOLO(str(MODEL_DIR/'yolo26-pose.pt'))
    assert model.model.object_class == model.ckpt['object_class'] == 'pen'
    assert model.model.object_point == model.ckpt['object_point'] == ['tip', 'tail']
    assert model.names == {0: 'pen'}
    assert list(model.model.kpt_shape) == [2, 3]
    dataset = args.dataset.resolve()
    LOGGER.info('Evaluating model=%s dataset=%s', MODEL_DIR/'yolo26-pose.pt', dataset)
    images = [str((dataset/p).resolve()) for p in (dataset/'val.txt').read_text().splitlines()]
    errors = [[], []]
    normalized = [[], []]
    missing = []
    per_image = []
    for prediction in model.predict(images, imgsz=640, device=device, batch=8, conf=0.25, stream=True, verbose=False):
        name = Path(prediction.path).stem
        target = np.loadtxt(dataset/'labels'/f'{name}.txt', ndmin=2)
        assert target.shape == (1,11), 'This evaluation expects one annotated pen per image.'
        if not len(prediction.boxes):
            missing.append(name)
            continue
        index = int(prediction.boxes.conf.argmax())
        predicted = prediction.keypoints.xy[index].cpu().numpy()
        h,w = prediction.orig_shape
        truth = target[0,5:].reshape(2,3)
        xy = truth[:,:2]*[w,h]
        distance = np.linalg.norm(predicted-xy,axis=1)
        pen_length = np.linalg.norm(xy[0]-xy[1])
        for i in range(2):
            if truth[i,2] > 0:
                errors[i].append(float(distance[i]))
                normalized[i].append(float(distance[i]/pen_length))
        per_image.append({'image':name,'tip_error_px':float(distance[0]),'tail_error_px':float(distance[1])})
    metrics = {}
    for i,name in enumerate(['tip','tail']):
        values = np.array(errors[i])
        assert len(values), f'No {name} predictions'
        metrics[name] = {
            'mean_error_px':float(values.mean()), 'median_error_px':float(np.median(values)),
            'p95_error_px':float(np.percentile(values,95)),
            'within_10px':float((values<=10).sum()/len(images)),
            'within_20px':float((values<=20).sum()/len(images)),
            'within_5_percent_pen_length':float((np.array(normalized[i])<=.05).sum()/len(images)),
        }
    training_run = Path(model.ckpt['metadata']['training_run'].replace('\\', '/'))
    # Historical checkpoints retain their original run location.
    run_dir = RUN_DIR.joinpath(*training_run.parts[1:]) if training_run.parts[0] == 'runs' else PROJECT_ROOT / training_run
    with (run_dir/'results.csv').open() as f:
        epochs = list(csv.DictReader(f))
    best = max(epochs,key=lambda r:float(r['metrics/mAP50-95(B)'])+float(r['metrics/mAP50-95(P)']))
    report = {
        'validation_images':len(images),'missing_detections':missing,
        'completed_epochs':int(epochs[-1]['epoch']), 'best_epoch':int(best['epoch']),
        'training_seconds':float(epochs[-1]['time']), 'metadata_verified':True,
        'keypoint_errors':metrics,
        'note':'Original-image pixel distances to visual manual labels; highest-confidence pen per image. Missing detections count as failures in threshold rates. Both videos share the same scene.',
        'per_image':per_image,
    }
    (MODEL_DIR/'endpoint_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    LOGGER.info('Evaluation completed: %s', json.dumps({k:v for k,v in report.items() if k != 'per_image'}))


if __name__ == '__main__':
    main()
