"""Extract annotation frames from the source videos."""
import logging
import cv2
import numpy as np

from pen_pose.path import DATASET_DIR, TEMP_DIR, VIDEO_DIR
from pen_pose.logging import configure_logging

OUT = DATASET_DIR
LOGGER = logging.getLogger(__name__)

def main() -> None:
    """Extract evenly spaced frames and a contact sheet."""
    configure_logging()
    (OUT / 'images').mkdir(parents=True, exist_ok=True)
    previews = []
    for vi, video in enumerate(sorted(VIDEO_DIR.glob('*.mp4'))):
        cap = cv2.VideoCapture(str(video))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        LOGGER.info('Extracting %s: frames=%s fps=%s width=%s height=%s', video.name, count, fps, cap.get(3), cap.get(4))
        indices = np.linspace(0, count - 1, 100).round().astype(int)
        for i, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError((video, frame_idx))
            name = f'v{vi+1}_{i:03d}'
            cv2.imwrite(str(OUT / 'images' / f'{name}.jpg'), frame)
            if i in [0, 20, 40, 60, 80, 99]:
                thumb = cv2.resize(frame, (480, 270))
                cv2.putText(thumb, name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                previews.append(thumb)
        cap.release()
    sheet = np.vstack([np.hstack(previews[i:i+3]) for i in range(0, len(previews), 3)])
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(TEMP_DIR / 'contact_sheet.jpg'), sheet)
    LOGGER.info('Dataset frames saved to %s', OUT)

if __name__ == '__main__':
    main()
