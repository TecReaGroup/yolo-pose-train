"""Train YOLO26 Pose and package the best checkpoint with pen metadata."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from pen_pose.logging import configure_logging
from pen_pose.path import DATASET_DIR, MODEL_DIR, PROJECT_ROOT, RUN_DIR, TEMP_DIR, WEIGHT_DIR

os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('YOLO_CONFIG_DIR', str(TEMP_DIR/'ultralytics'))

import torch
import ultralytics
from ultralytics import YOLO
import yaml

from pen_pose.training.device import select_device

LOGGER = logging.getLogger(__name__)


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Persist a training status or metadata document as UTF-8."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def dataset_digest(dataset: Path) -> str:
    """Fingerprint image, annotation, and split contents."""
    digest = hashlib.sha256()
    for folder in ['images', 'labels']:
        for path in sorted((dataset / folder).glob('*')):
            if path.is_file():
                digest.update(path.relative_to(dataset).as_posix().encode())
                digest.update(path.read_bytes())
    for name in ['train.txt', 'val.txt']:
        digest.update((dataset / name).read_bytes())
    return digest.hexdigest()


def package_model(best: Path, output: Path, dataset: Path, run_dir: Path) -> YOLO:
    """Package a local checkpoint with verified pen-specific metadata."""
    metadata = {
        'schema_version': 1,
        'model_family': 'yolo26-pose',
        'model_variant': 'yolo26n-pose',
        'task': 'pose',
        'object_class': 'pen',
        'object_point': ['tip', 'tail'],
        'names': {0: 'pen'},
        'kpt_names': {0: ['tip', 'tail']},
        'kpt_shape': [2, 3],
        'flip_idx': [0, 1],
        'keypoint_definitions': {
            'tip': 'Writing tip at the front of the metal nib',
            'tail': 'Center of the blue cap end opposite the writing tip',
        },
        'label_visibility': {'0': 'not labeled', '1': 'occluded / estimated', '2': 'visible'},
        'prediction_keypoint_fields': ['x', 'y', 'confidence'],
        'prediction_note': 'Keypoint confidence is not a COCO visibility value.',
        'input_size': [640, 640],
        'training_images': 160,
        'validation_images': 40,
        'dataset_sha256': dataset_digest(dataset),
        'dataset_note': 'Visual estimates from two videos of the same scene; validation uses held-out time blocks.',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'ultralytics_version': ultralytics.__version__,
        'torch_version': str(torch.__version__),
        'source_checkpoint': best.relative_to(PROJECT_ROOT).as_posix(),
        'training_run': run_dir.relative_to(PROJECT_ROOT).as_posix(),
    }
    # The checkpoint was generated locally by this training run.
    checkpoint = torch.load(best, map_location='cpu', weights_only=False)
    checkpoint['metadata'] = metadata
    checkpoint['object_class'] = metadata['object_class']
    checkpoint['object_point'] = metadata['object_point']
    for key in ['model', 'ema']:
        network = checkpoint.get(key)
        if network is not None:
            network.object_class = metadata['object_class']
            network.object_point = metadata['object_point']
            network.metadata = metadata
            network.names = {0: 'pen'}
            network.kpt_names = {0: ['tip', 'tail']}
    torch.save(checkpoint, output)
    write_json(output.with_suffix('.metadata.json'), metadata)
    loaded = YOLO(str(output))
    assert loaded.task == 'pose'
    assert loaded.names == {0: 'pen'}
    assert loaded.model.object_class == 'pen'
    assert loaded.model.object_point == ['tip', 'tail']
    assert list(loaded.model.kpt_shape) == [2, 3]
    assert loaded.ckpt['object_point'] == ['tip', 'tail']
    assert loaded.ckpt['metadata'] == metadata
    return loaded


def main() -> None:
    """Train, validate, and package a pen pose checkpoint."""
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=DATASET_DIR)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    args.device = select_device(args.device)
    dataset = args.dataset.resolve()
    LOGGER.info('Training dataset=%s epochs=%s batch=%s', dataset, args.epochs, args.batch)
    cfg = yaml.safe_load((dataset/'data.yaml').read_text(encoding='utf-8'))
    assert cfg['names'] == {0: 'pen'}
    assert cfg['kpt_shape'] == [2, 3]
    assert cfg['kpt_names'] == {0: ['tip', 'tail']}
    assert cfg['flip_idx'] == [0, 1]
    output_dir = MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir/'yolo26-pose.pt'
    if output.exists():
        raise FileExistsError(f'Preserving existing model: {output}. Rename it before starting a new run.')
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    # Resolve custom datasets independently of stale paths in source YAML.
    cfg['path'] = dataset.as_posix()
    dataset_config = TEMP_DIR/'training_dataset.yaml'
    dataset_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    status_path = TEMP_DIR/'training_status.json'
    write_json(status_path, {'state': 'initializing', 'epochs': args.epochs})
    try:
        model = YOLO(str(WEIGHT_DIR/'yolo26n-pose.pt'))

        def progress(trainer: Any) -> None:
            """Persist the latest completed epoch and its metrics."""
            metrics = {k: float(v) for k,v in trainer.metrics.items()}
            LOGGER.info('Epoch %s/%s: %s', trainer.epoch+1, trainer.epochs, metrics)
            write_json(status_path, {
                'state': 'training', 'epoch': trainer.epoch+1, 'epochs': trainer.epochs,
                'run_dir': str(trainer.save_dir), 'metrics': metrics,
            })

        model.add_callback('on_fit_epoch_end', progress)
        model.train(
            data=str(dataset_config), epochs=args.epochs, imgsz=640,
            batch=args.batch, device=args.device, workers=2, patience=25,
            project=str(RUN_DIR/'pose'), name='pen_yolo26n', exist_ok=False,
            pretrained=True, optimizer='AdamW', lr0=0.001, lrf=0.01,
            cos_lr=True, warmup_epochs=3, weight_decay=0.0005,
            degrees=25, translate=0.1, scale=0.35, fliplr=0.5, flipud=0.0,
            mosaic=0.3, close_mosaic=10, mixup=0.0,
            seed=42, deterministic=True, amp=True, plots=True, save=True,
            cache=False, verbose=False,
        )
        run_dir = Path(model.trainer.save_dir)
        best = Path(model.trainer.best)
        loaded = package_model(best, output, dataset, run_dir)
        metrics = loaded.val(data=str(dataset_config), imgsz=640, batch=args.batch,
                             device=args.device, workers=2, project=str(RUN_DIR/'pose'),
                             name='pen_final_validation', plots=True, verbose=False)
        samples = [dataset/'images'/name for name in ['v1_050.jpg','v1_080.jpg','v2_025.jpg','v2_065.jpg']]
        predictions = loaded.predict(source=[str(p) for p in samples], imgsz=640,
                                     device=args.device, conf=0.25, verbose=False)
        prediction_summary = []
        for prediction in predictions:
            assert prediction.keypoints is not None and tuple(prediction.keypoints.data.shape[1:]) == (2, 3)
            assert len(prediction.boxes) > 0, f'No pen detected in {prediction.path}'
            prediction.save(filename=str(output_dir/f'prediction_{Path(prediction.path).stem}.jpg'))
            prediction_summary.append({'image': Path(prediction.path).name,
                                       'detections': len(prediction.boxes),
                                       'keypoint_shape': list(prediction.keypoints.data.shape)})
        report = {
            'state': 'completed', 'model': str(output), 'run_dir': str(run_dir),
            'validation': {k:float(v) for k,v in metrics.results_dict.items()},
            'metadata_verified': True, 'predictions': prediction_summary,
        }
        write_json(output_dir/'training_report.json', report)
        write_json(status_path, report)
        LOGGER.info('Training completed: %s', json.dumps(report))
    except Exception as error:
        write_json(status_path, {'state': 'failed', 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
