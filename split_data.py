import shutil
from pathlib import Path
import random
import os

SRC_DIR = 'data/flower_photos'
DST_DIR = 'data'
VAL_RATE = 0.2
SEED = 42
IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp",
             ".JPG", ".JPEG", ".PNG", ".BMP"}

def split(src_dir, dst_dir, val_rate):
    src = Path(src_dir)
    dst = Path(dst_dir)

    assert src.is_dir(), f"{src} does not exist"

    train_dir = dst / 'train'
    val_dir = dst / 'val'

    if train_dir.is_dir():
        shutil.rmtree(train_dir)
    if val_dir.is_dir():
        shutil.rmtree(val_dir)

    classes = sorted([d.name for d in src.iterdir() if d.is_dir()])
    print(f'类别 : {classes}')

    rng = random.Random(SEED)
    total_train, total_val = 0, 0

    for cls in classes:
        cls_src = src / cls
        imgs = [f for f in os.listdir(cls_src) if os.path.splitext(f)[-1] in IMG_EXTS]
        imgs.sort()
        rng.shuffle(imgs)

        n_val = int(len(imgs) * val_rate)
        val_imgs = imgs[:n_val]
        train_imgs = imgs[n_val:]

        (train_dir / cls).mkdir(parents=True, exist_ok=True)
        (val_dir / cls).mkdir(parents=True, exist_ok=True)

        for f in train_imgs:
            shutil.copy2(cls_src / f, train_dir / cls / f)
        for f in val_imgs:
            shutil.copy2(cls_src / f, val_dir / cls / f)

        total_train += len(train_imgs)
        total_val += len(val_imgs)

        print(f"{cls:15s} 总数 {len(imgs):4d} | "
              f"train {len(train_imgs):4d} | val {len(val_imgs):4d}")

    print("-" * 50)
    print(f"合计: train {total_train}, val {total_val}")
    print(f"输出目录: {dst.resolve()}")

if __name__ == '__main__':
    split(SRC_DIR, DST_DIR, VAL_RATE)


