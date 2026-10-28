"""Export manual pen annotations to YOLO pose labels."""
import csv
import json
import logging
import cv2
import numpy as np

from pen_pose.path import ANNOTATION_DIR, DATASET_DIR, TEMP_DIR, VIDEO_DIR
from pen_pose.logging import configure_logging

OUT = DATASET_DIR
LOGGER = logging.getLogger(__name__)

def main() -> None:
    """Export normalized labels, dataset splits, and annotation previews."""
    configure_logging()
    LOGGER.info('Exporting annotations to %s', OUT)
    records = {}
    for i in range(62):
        records[f'v1_{i:03d}'] = dict(tip=[1034, 336], tail=[982, 745], visibility=[2, 1 if i == 17 else 2], bbox=[954, 330, 1048, 751])
    for filename, scale, offset in [('manual_points.csv', 2, np.array([650,180])), ('manual_full.csv', 3, np.array([0,0]))]:
        with (ANNOTATION_DIR / filename).open() as f:
            for row in csv.DictReader(f):
                tip = np.array([float(row['tip_x']),float(row['tip_y'])]) * scale + offset
                tail = np.array([float(row['tail_x']),float(row['tail_y'])]) * scale + offset
                radius = float(row['pad']) * scale
                direction = tail - tip
                normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
                # Enclose the shaft/cap width; the metal tip tapers to a point.
                corners = np.array([tip, tip + direction * .15 + normal * radius * .7,
                                    tip + direction * .15 - normal * radius * .7,
                                    tail + normal * radius, tail - normal * radius])
                low = corners.min(axis=0) - 5
                high = corners.max(axis=0) + 5
                records[row['image']] = dict(tip=tip.tolist(), tail=tail.tolist(), visibility=[int(row['tip_v']),int(row['tail_v'])], bbox=[*low.tolist(),*high.tolist()])
    images = sorted((OUT / 'images').glob('*.jpg'))
    assert len(images) == 200
    assert {f.stem for f in images} == set(records), ({f.stem for f in images} - set(records))
    (OUT / 'labels').mkdir(parents=True, exist_ok=True)
    for folder in ['preview', 'annotated_sheet']:
        (TEMP_DIR / folder).mkdir(parents=True, exist_ok=True)
    tiles = []
    for f in images:
        r = records[f.stem]
        x1,y1,x2,y2 = r['bbox']
        tx,ty = r['tip']; ex,ey = r['tail']
        tv,ev = r['visibility']
        values = [0,(x1+x2)/3840,(y1+y2)/2160,(x2-x1)/1920,(y2-y1)/1080,tx/1920,ty/1080,tv,ex/1920,ey/1080,ev]
        assert all(0 <= values[i] <= 1 for i in [1,2,3,4,5,6,8,9])
        assert x1 <= tx <= x2 and y1 <= ty <= y2 and x1 <= ex <= x2 and y1 <= ey <= y2
        (OUT / 'labels' / f'{f.stem}.txt').write_text(' '.join(str(int(v)) if i in [0,7,10] else f'{v:.6f}' for i,v in enumerate(values))+'\n')
        im = cv2.imread(str(f))
        cv2.rectangle(im, (round(x1),round(y1)), (round(x2),round(y2)), (0,220,0), 3)
        for point, name, color, vis in [(r['tip'],'tip',(0,0,255),tv),(r['tail'],'tail',(255,100,0),ev)]:
            p = tuple(round(v) for v in point)
            cv2.circle(im,p,7,color,-1 if vis == 2 else 2)
            cv2.putText(im,f'{name} v={vis}',(p[0]+12,p[1]),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
        cv2.imwrite(str(TEMP_DIR / 'preview' / f.name),im)
        # Object-centered review crop includes enough context to check the box.
        cx,cy = (x1+x2)/2,(y1+y2)/2
        size = max(x2-x1,y2-y1)+110
        left = max(0,min(1920-int(size),int(cx-size/2)))
        top = max(0,min(1080-int(size),int(cy-size/2)))
        tile = cv2.resize(im[top:top+int(size),left:left+int(size)],(300,300))
        cv2.putText(tile,f.stem,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
        tiles.append(tile)
    for i in range(0,200,20):
        sheet = np.vstack([np.hstack(tiles[j:j+5]) for j in range(i,i+20,5)])
        cv2.imwrite(str(TEMP_DIR/'annotated_sheet'/f'{i//20:02d}.jpg'),sheet)
    (OUT/'annotations_pixels.json').write_text(json.dumps(records,indent=2))
    # Hold out contiguous time blocks, rather than alternating neighboring frames.
    val = {f'v1_{i:03d}' for i in [*range(45,60),*range(80,85)]}
    val |= {f'v2_{i:03d}' for i in [*range(20,30),*range(60,70)]}
    for split in ['train','val']:
        chosen = [f'./images/{f.name}' for f in images if (f.stem in val) == (split == 'val')]
        (OUT/f'{split}.txt').write_text('\n'.join(chosen)+'\n')
    (OUT/'data.yaml').write_text(f'path: {OUT.as_posix()}\ntrain: train.txt\nval: val.txt\nnames:\n  0: pen\nkpt_shape: [2, 3]\nflip_idx: [0, 1]\nkpt_names:\n  0: [tip, tail]\n')
    manifest = []
    for vi,video in enumerate(sorted(VIDEO_DIR.glob('*.mp4')),1):
        cap=cv2.VideoCapture(str(video)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS); cap.release()
        for i,idx in enumerate(np.linspace(0,n-1,100).round().astype(int)):
            manifest.append([f'v{vi}_{i:03d}.jpg',video.name,int(idx),round(int(idx)/fps,6)])
    with (OUT/'frames.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['image','source_video','frame_index','seconds']); w.writerows(manifest)
    counts = {str(v):sum(r['visibility'].count(v) for r in records.values()) for v in [0,1,2]}
    report=dict(images=200,labels=200,objects=200,keypoints=400,visibility=counts,train=160,val=40)
    (OUT/'validation.json').write_text(json.dumps(report,indent=2))
    LOGGER.info('Annotation export completed: %s', json.dumps(report))

if __name__ == '__main__':
    main()
