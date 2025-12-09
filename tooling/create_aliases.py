import argparse
import os
import shutil


def copy_alias(root, src, dst):
    src_path = os.path.join(root, src)
    if not os.path.exists(src_path):
        print(f"Source not found: {src_path}")
        return 2

    # Normalize dst to use OS separators
    dst_rel = dst.replace('/', os.sep).replace('\\', os.sep)
    dst_path = os.path.join(root, dst_rel)

    count = 0
    for r, dirs, files in os.walk(src_path):
        rel = os.path.relpath(r, src_path)
        if rel == '.':
            target_dir = dst_path
        else:
            target_dir = os.path.join(dst_path, rel)

        os.makedirs(target_dir, exist_ok=True)

        for f in files:
            sfile = os.path.join(r, f)
            dfile = os.path.join(target_dir, f)
            if os.path.exists(dfile):
                print(f"Skipping existing: {dfile}")
                continue
            shutil.copy2(sfile, dfile)
            print(f"Copied: {sfile} -> {dfile}")
            count += 1

    print(f"Done. Files copied: {count}")
    return 0


def main():
    p = argparse.ArgumentParser(description="Create filesystem aliases by copying folders under data/raw")
    p.add_argument('--root', default='data/raw', help='data root')
    p.add_argument('--src', required=True, help='source folder name under data root (e.g. XBTUSD)')
    p.add_argument('--dst', required=True, help='destination folder name under data root (e.g. XBT/USD)')
    args = p.parse_args()

    return copy_alias(args.root, args.src, args.dst)


if __name__ == '__main__':
    raise SystemExit(main())
