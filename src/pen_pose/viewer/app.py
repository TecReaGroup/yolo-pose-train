"""A small, read-only YOLO Pose dataset viewer."""
import argparse
import logging
import math
import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QRectF, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFrame, QGraphicsItem,
    QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMainWindow, QPushButton, QScrollArea, QSplitter,
    QVBoxLayout, QWidget,
)

from pen_pose.path import DATASET_DIR, TEMP_DIR
from pen_pose.logging import configure_logging

LOGGER = logging.getLogger(__name__)
COLORS = ['#ff786c', '#70b7ff']
STYLE = """
QWidget { background: #15191f; color: #e7ecf2; font-size: 13px; }
QMainWindow { background: #101318; }
QFrame#panel { background: #1c2129; border: 1px solid #2c333e; border-radius: 12px; }
QLabel { background: transparent; }
QLabel#title { font-size: 23px; font-weight: 700; }
QLabel#muted { color: #929eae; }
QLabel#section { color: #929eae; font-size: 11px; font-weight: 600; }
QPushButton, QComboBox, QLineEdit { background: #242b35; border: 1px solid #343e4b; border-radius: 7px; padding: 8px 12px; }
QPushButton:hover { background: #303b49; border-color: #607286; }
QPushButton:disabled { color: #58616e; }
QPushButton:checked { background: #21483f; color: #8be1c2; border-color: #418e75; }
QLineEdit:focus { border-color: #70c7aa; }
QListWidget { background: transparent; border: none; outline: none; }
QListWidget::item { padding: 10px; margin: 2px; border-radius: 6px; }
QListWidget::item:selected { background: #28473f; color: #a0ebd1; }
QListWidget::item:hover { background: #2a323d; }
QCheckBox { background: transparent; spacing: 8px; padding: 5px 0; }
QScrollArea { border: none; }
QScrollBar:vertical { background: #1c2129; width: 8px; }
QScrollBar::handle:vertical { background: #46515e; border-radius: 4px; min-height: 25px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QSplitter::handle { background: #15191f; width: 10px; }
"""


def read_labels(path: Path) -> list[list[float]]:
    """Read and validate single-class, two-keypoint YOLO annotations."""
    if not path.exists():
        raise ValueError('缺少标签文件')
    rows = []
    for n, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        v = list(map(float, line.split()))
        if (len(v) != 11 or not all(math.isfinite(x) for x in v)
                or v[0] != 0 or any(not 0 <= v[i] <= 1 for i in [1,2,3,4,5,6,8,9])
                or v[3] <= 0 or v[4] <= 0 or v[7] not in [0,1,2] or v[10] not in [0,1,2]):
            raise ValueError(f'第 {n} 行不是有效的 pen / 双关键点标签')
        rows.append(v)
    return rows


class Canvas(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor('#101318'))
        self.setStyleSheet('border: none;')
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.image_rect = QRectF()
        self.target_rect = QRectF()
        self.auto_fit = True
        self.box_items = []
        self.point_items = []

    def fit_image(self):
        self.auto_fit = True
        if not self.image_rect.isEmpty():
            self.fitInView(self.image_rect.adjusted(-30,-30,30,30), Qt.KeepAspectRatio)

    def focus_pen(self):
        if not self.target_rect.isEmpty():
            self.auto_fit = False
            self.fitInView(self.target_rect.adjusted(-100,-100,100,100), Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.auto_fit:
            self.fit_image()

    def wheelEvent(self, event):
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        if .05 <= self.transform().m11() * factor <= 15:
            self.auto_fit = False
            self.scale(factor, factor)
        event.accept()

    def load(self, pixmap, rows):
        self.box_items = []
        self.point_items = []
        self.scene().clear()
        self.resetTransform()
        self.image_rect = QRectF(pixmap.rect())
        self.target_rect = QRectF()
        self.scene().setSceneRect(self.image_rect)
        if pixmap.isNull():
            return
        self.scene().addPixmap(pixmap)
        w, h = pixmap.width(), pixmap.height()
        for row in rows:
            _, cx, cy, bw, bh, *points = row
            rect = QRectF((cx-bw/2)*w, (cy-bh/2)*h, bw*w, bh*h)
            self.target_rect = rect if self.target_rect.isEmpty() else self.target_rect.united(rect)
            pen = QPen(QColor('#7ee2b8'), 2)
            pen.setCosmetic(True)
            self.box_items.append(self.scene().addRect(rect, pen))
            for i, name in enumerate(['tip', 'tail']):
                x, y, visibility = points[i*3:i*3+3]
                if visibility == 0:
                    continue
                color = QColor(COLORS[i])
                pen = QPen(color, 2)
                dot = self.scene().addEllipse(-5,-5,10,10,pen,
                    color if visibility == 2 else Qt.NoBrush)
                dot.setPos(x*w, y*h)
                dot.setFlag(QGraphicsItem.ItemIgnoresTransformations)
                text = self.scene().addSimpleText(f'  {name}' + (' · 遮挡' if visibility == 1 else ''))
                text.setBrush(color)
                text.setFont(QFont('Microsoft YaHei UI', 10, QFont.Bold))
                text.setPos(x*w+10, y*h-25)
                text.setFlag(QGraphicsItem.ItemIgnoresTransformations)
                self.point_items.extend([dot, text])
        self.fit_image()


class Viewer(QMainWindow):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.files = sorted(p for p in (dataset/'images').glob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png'])
        self.filtered = []
        self.rows = []
        self.setWindowTitle('Pen Studio · 标注预览')
        self.resize(1440, 900)
        self.setMinimumSize(1050, 650)
        body = QWidget()
        self.setCentralWidget(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24,20,24,16)
        layout.setSpacing(16)
        header = QHBoxLayout()
        title = QLabel('Pen Studio'); title.setObjectName('title')
        header.addWidget(title)
        subtitle = QLabel('标注预览  /  YOLO Pose'); subtitle.setObjectName('muted')
        header.addWidget(subtitle); header.addStretch()
        header.addWidget(QLabel(f'{len(self.files)} 张图片   ·   tip / tail'))
        layout.addLayout(header)
        splitter = QSplitter()
        layout.addWidget(splitter, 1)
        left, ll = self.panel()
        left.setMinimumWidth(195)
        ll.addWidget(self.label('图片库', 'section'))
        self.search = QLineEdit(); self.search.setPlaceholderText('搜索文件名…')
        ll.addWidget(self.search)
        self.source = QComboBox(); self.source.addItems(['全部视频', '视频 1', '视频 2'])
        ll.addWidget(self.source)
        self.list = QListWidget(); ll.addWidget(self.list,1)
        self.count = self.label('', 'muted'); ll.addWidget(self.count)
        splitter.addWidget(left)
        center, cl = self.panel()
        bar = QHBoxLayout()
        self.filename = QLabel('选择图片'); bar.addWidget(self.filename); bar.addStretch()
        self.fit_button = self.button('适应画布', lambda: self.canvas.fit_image())
        self.focus_button = self.button('聚焦笔', lambda: self.canvas.focus_pen())
        bar.addWidget(self.fit_button); bar.addWidget(self.focus_button)
        cl.addLayout(bar)
        self.canvas = Canvas(); cl.addWidget(self.canvas,1)
        nav = QHBoxLayout()
        self.previous = self.button('← 上一张', lambda: self.step(-1))
        self.next = self.button('下一张 →', lambda: self.step(1))
        self.position = self.label('0 / 0', 'muted')
        nav.addWidget(self.previous); nav.addStretch(); nav.addWidget(self.position)
        nav.addStretch(); nav.addWidget(self.next); cl.addLayout(nav)
        splitter.addWidget(center)
        right, rl = self.panel(); right.setMinimumWidth(205)
        rl.addWidget(self.label('显示图层', 'section'))
        self.boxes = QCheckBox('笔的边界框'); self.boxes.setChecked(True)
        self.points = QCheckBox('tip / tail 关键点'); self.points.setChecked(True)
        rl.addWidget(self.boxes); rl.addWidget(self.points)
        rl.addSpacing(20); rl.addWidget(self.label('当前标注', 'section'))
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        self.details = QLabel(); self.details.setWordWrap(True)
        self.details.setAlignment(Qt.AlignTop)
        self.details.setTextFormat(Qt.PlainText)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        scroll.setWidget(self.details); rl.addWidget(scroll,1)
        rl.addWidget(self.label('● tip  笔尖', 'tip'))
        rl.addWidget(self.label('● tail  笔尾', 'tail'))
        rl.addWidget(self.label('实心：可见    空心：遮挡', 'muted'))
        splitter.addWidget(right)
        splitter.setSizes([215,950,225]); splitter.setStretchFactor(1,1)
        self.status = self.label('滚轮缩放 · 拖动画布 · ← → 翻页 · F 适应画布 · P 聚焦笔', 'muted')
        layout.addWidget(self.status)
        self.search.textChanged.connect(self.filter_files)
        self.source.currentIndexChanged.connect(self.filter_files)
        self.list.currentRowChanged.connect(self.select)
        self.boxes.toggled.connect(self.update_layers)
        self.points.toggled.connect(self.update_layers)
        for key, callback in [('Left',lambda:self.step(-1)),('Right',lambda:self.step(1)),
                              ('F',self.canvas.fit_image),('P',self.canvas.focus_pen)]:
            shortcut = QShortcut(QKeySequence(key),self)
            shortcut.activated.connect(lambda fn=callback: fn() if not self.search.hasFocus() else None)
        self.filter_files()
        QTimer.singleShot(0,self.canvas.fit_image)

    @staticmethod
    def label(text, name):
        label = QLabel(text); label.setObjectName(name)
        if name in ['tip','tail']:
            label.setStyleSheet(f'color: {COLORS[0 if name == "tip" else 1]};')
        return label

    @staticmethod
    def button(text, callback):
        button = QPushButton(text); button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(callback)
        return button

    @staticmethod
    def panel():
        frame = QFrame(); frame.setObjectName('panel')
        layout = QVBoxLayout(frame); layout.setContentsMargins(14,14,14,14)
        layout.setSpacing(10)
        return frame, layout

    def filter_files(self):
        current = self.list.currentItem().text() if self.list.currentItem() else None
        text = self.search.text().strip().lower()
        source = self.source.currentIndex()
        self.filtered = [p for p in self.files if text in p.name.lower() and (source == 0 or p.stem.startswith(f'v{source}_'))]
        self.list.blockSignals(True)
        self.list.clear(); self.list.addItems([p.name for p in self.filtered])
        index = next((i for i,p in enumerate(self.filtered) if p.name == current),0)
        self.list.setCurrentRow(index if self.filtered else -1)
        self.list.blockSignals(False)
        self.count.setText(f'{len(self.filtered)} / {len(self.files)} 张图片')
        self.select(self.list.currentRow())

    def step(self, delta):
        if self.filtered:
            self.list.setCurrentRow(max(0,min(len(self.filtered)-1,self.list.currentRow()+delta)))

    def select(self, index):
        self.previous.setEnabled(index > 0)
        self.next.setEnabled(0 <= index < len(self.filtered)-1)
        self.rows = []
        self.position.setText(f'{index+1 if index >= 0 else 0} / {len(self.filtered)}')
        if index < 0:
            self.canvas.load(QPixmap(),[])
            self.filename.setText('没有匹配的图片')
            self.details.setText('请调整搜索条件。' if self.files else f'未找到图片，请检查：\n{self.dataset / "images"}')
            return
        path = self.filtered[index]
        self.filename.setText(path.name)
        pixmap = QPixmap(str(path))
        try:
            if pixmap.isNull():
                raise ValueError('图片无法读取')
            self.rows = read_labels(self.dataset/'labels'/f'{path.stem}.txt')
            details = [f'{pixmap.width()} × {pixmap.height()} px',f'{len(self.rows)} 个 pen 实例']
            for n,r in enumerate(self.rows,1):
                details.append(f'\n实例 {n} · pen\n框尺寸  {r[3]*pixmap.width():.0f} × {r[4]*pixmap.height():.0f} px')
                for offset,name in [(5,'tip'),(8,'tail')]:
                    state = ['未标注','遮挡估计','可见'][int(r[offset+2])]
                    details.append(f'\n{name} · {state}\n({r[offset]*pixmap.width():.1f}, {r[offset+1]*pixmap.height():.1f})')
            self.details.setText('\n'.join(details))
        except (ValueError,OSError) as error:
            LOGGER.warning('Cannot load annotation for %s: %s', path, error)
            self.details.setText(f'无法加载标注\n\n{error}')
        self.canvas.load(pixmap,self.rows)
        self.update_layers()

    def update_layers(self):
        for item in self.canvas.box_items:
            item.setVisible(self.boxes.isChecked())
        for item in self.canvas.point_items:
            item.setVisible(self.points.isChecked())


def main() -> int:
    """Launch the read-only dataset viewer."""
    configure_logging()
    parser = argparse.ArgumentParser(description='Pen Studio 标注预览')
    parser.add_argument('--dataset',type=Path,default=DATASET_DIR)
    parser.add_argument('--smoke-test',action='store_true',help='离屏检查全部图片和交互后退出')
    args = parser.parse_args()
    LOGGER.info('Opening dataset viewer: %s', args.dataset.resolve())
    app = QApplication(sys.argv[:1] + (['-platform','offscreen'] if args.smoke_test and os.environ.get('PEN_NATIVE_CHECK') != '1' else []))
    app.setStyle('Fusion'); app.setFont(QFont('Microsoft YaHei UI',10))
    app.setStyleSheet(STYLE)
    window = Viewer(args.dataset.resolve()); window.show()
    if args.smoke_test:
        app.processEvents()
        assert window.files, '没有可供验证的图片'
        for i in range(len(window.filtered)):
            window.list.setCurrentRow(i)
            assert window.rows, window.filename.text() + ': 标签加载失败'
        window.source.setCurrentIndex(2)
        assert all(p.stem.startswith('v2_') for p in window.filtered)
        window.search.setText('no-such-image'); assert not window.filtered
        window.search.clear(); window.source.setCurrentIndex(0)
        window.list.setCurrentRow(17)
        window.boxes.setChecked(False)
        assert all(not x.isVisible() for x in window.canvas.box_items)
        window.boxes.setChecked(True)
        window.points.setChecked(False)
        assert all(not x.isVisible() for x in window.canvas.point_items)
        window.points.setChecked(True)
        window.step(1); assert window.list.currentRow() == 18
        window.canvas.focus_pen(); assert not window.canvas.auto_fit
        window.canvas.fit_image(); assert window.canvas.auto_fit
        window.list.setCurrentRow(80); app.processEvents()
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        assert window.grab().save(str(TEMP_DIR/'app-preview.png'))
        print(f'PASS: {len(window.files)} images, labels, filters, navigation, overlays and focus')
        window.close()
        return 0
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
