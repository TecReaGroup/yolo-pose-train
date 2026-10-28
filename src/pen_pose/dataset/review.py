"""Generate cropped and full-frame annotation review sheets."""
import logging

import cv2
import numpy as np

from pen_pose.logging import configure_logging
from pen_pose.path import DATASET_DIR, TEMP_DIR

LOGGER = logging.getLogger(__name__)
REVIEW_DIR = TEMP_DIR / 'review'


def main() -> None:
    """Build review sheets from the project's annotated video frames."""
    configure_logging()
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted((DATASET_DIR / 'images').glob('*.jpg'))
    if not images:
        raise FileNotFoundError(f'No dataset images found in {DATASET_DIR}')
    LOGGER.info('Generating review sheets for %s images', len(images))
    for start in range(0, len(images), 20):
        tiles = []
        for image_path in images[start:start+20]:
            image = cv2.imread(str(image_path))[180:850, 650:1550]
            image = cv2.resize(image, (450, 335))
            cv2.putText(image, image_path.stem, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255), 2)
            tiles.append(image)
        while len(tiles) < 20:
            tiles.append(np.zeros((335,450,3), np.uint8))
        sheet = np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,20,4)])
        cv2.imwrite(str(REVIEW_DIR / f'page_{start//20:02d}.jpg'), sheet)

    selected_names = [f'v1_{i:03d}' for i in [67,75,76,77,78,79,84,85,86,90,91,92]]
    selected_names += [f'v2_{i:03d}' for i in [49,50,51,*range(64,84)]]
    for start in range(0, len(selected_names), 8):
        tiles = []
        for name in selected_names[start:start+8]:
            image = cv2.resize(cv2.imread(str(DATASET_DIR / 'images' / f'{name}.jpg')), (640,360))
            cv2.putText(image, name, (8,24), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255), 2)
            tiles.append(image)
        while len(tiles) < 8:
            tiles.append(np.zeros((360,640,3),np.uint8))
        cv2.imwrite(str(REVIEW_DIR / f'full_{start//8:02d}.jpg'), np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,8,2)]))
    LOGGER.info('Review sheets saved to %s', REVIEW_DIR)


if __name__ == '__main__':
    main()
