"""Convert the input dataset and pixel annotations to 720p."""
import json
import logging
from pathlib import Path
import shutil

import cv2

from pen_pose.logging import configure_logging

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / 'data/input'
BACKUP_DIR = ROOT / 'temp/input_before_720p'
STAGING_DIR = ROOT / 'temp/input_720p_stage'
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
LOGGER = logging.getLogger('pen_pose.dataset.resize_input')


def main() -> None:
    """Stage resized files, back up originals, and verify the conversion."""
    configure_logging()
    annotation_path = INPUT_DIR / 'annotations_pixels.json'
    annotations = json.loads(annotation_path.read_text(encoding='utf-8'))
    width, height = annotations['width'], annotations['height']
    image_paths = sorted((INPUT_DIR / 'images').glob('*.jpg'))
    preview_paths = sorted((INPUT_DIR / 'preview').glob('*.jpg'))
    expected_names = {entry['image'] for entry in annotations['annotations']}
    if not image_paths or {p.name for p in image_paths} != expected_names:
        raise ValueError('Images do not match pixel annotations')
    if {p.stem for p in (INPUT_DIR / 'labels').glob('*.txt')} != {p.stem for p in image_paths}:
        raise ValueError('Images do not match YOLO labels')
    for path in image_paths + preview_paths:
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (height, width):
            raise ValueError(f'Unexpected image dimensions: {path}')
    if (width, height) == (TARGET_WIDTH, TARGET_HEIGHT):
        LOGGER.info('Input dataset is already 1280x720; no changes needed')
        return
    if width * TARGET_HEIGHT != height * TARGET_WIDTH:
        raise ValueError('Expected 16:9 images; refusing to distort annotations')
    if BACKUP_DIR.exists() or STAGING_DIR.exists():
        raise FileExistsError('Conversion backup or staging directory already exists; inspect before retrying')
    shutil.copytree(INPUT_DIR, BACKUP_DIR)
    LOGGER.info('Original dataset backed up to %s', BACKUP_DIR)
    for path in image_paths + preview_paths:
        target = STAGING_DIR / path.relative_to(INPUT_DIR)
        target.parent.mkdir(parents=True, exist_ok=True)
        resized = cv2.resize(cv2.imread(str(path)), (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(target), resized, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f'Could not write {target}')
    sx, sy = TARGET_WIDTH / width, TARGET_HEIGHT / height
    for entry in annotations['annotations']:
        entry['bbox_xyxy'] = [value * scale for value, scale in zip(entry['bbox_xyxy'], [sx, sy, sx, sy])]
        for name in ['tip', 'tail']:
            entry[name][0] *= sx
            entry[name][1] *= sy
    annotations['width'], annotations['height'] = TARGET_WIDTH, TARGET_HEIGHT
    (STAGING_DIR / annotation_path.name).write_text(json.dumps(annotations, indent=2), encoding='utf-8')
    validation_path = INPUT_DIR / 'validation.json'
    validation = json.loads(validation_path.read_text(encoding='utf-8'))
    validation.update(width=TARGET_WIDTH, height=TARGET_HEIGHT)
    (STAGING_DIR / validation_path.name).write_text(json.dumps(validation, indent=2), encoding='utf-8')
    for path in STAGING_DIR.rglob('*.jpg'):
        if cv2.imread(str(path)).shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
            raise ValueError(f'Incorrect resized dimensions: {path}')
    for path in sorted(STAGING_DIR.rglob('*')):
        if path.is_file():
            path.replace(INPUT_DIR / path.relative_to(STAGING_DIR))
    shutil.rmtree(STAGING_DIR)
    for path in image_paths + preview_paths:
        if cv2.imread(str(path)).shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
            raise ValueError(f'Final dimensions do not match: {path}')
    for path in (INPUT_DIR / 'labels').glob('*.txt'):
        if path.read_bytes() != (BACKUP_DIR / 'labels' / path.name).read_bytes():
            raise ValueError(f'Normalized label changed unexpectedly: {path}')
    saved = json.loads(annotation_path.read_text(encoding='utf-8'))
    if saved != annotations:
        raise ValueError('Saved pixel annotations do not match converted coordinates')
    LOGGER.info('Converted and verified %d images and %d previews to 1280x720; pixel coordinates scaled; YOLO labels unchanged', len(image_paths), len(preview_paths))


if __name__ == '__main__':
    main()
