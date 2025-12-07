import argparse
import os
import hashlib


def sha256sum(path, block_size=65536):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            h.update(block)
    return h.hexdigest()


def convert(src_root, dst_root):
    if not os.path.exists(src_root):
        print(f"Source root does not exist: {src_root}")
        return 2

    files_converted = 0
    for dirpath, dirnames, filenames in os.walk(src_root):
        rel = os.path.relpath(dirpath, src_root)
        if rel == '.':
            target_dir = dst_root
        else:
            target_dir = os.path.join(dst_root, rel)

        os.makedirs(target_dir, exist_ok=True)

        for name in filenames:
            src_file = os.path.join(dirpath, name)
            dst_file = os.path.join(target_dir, name)

            # If dst exists, verify identical contents
            if os.path.exists(dst_file):
                try:
                    s_stat = os.stat(src_file)
                    d_stat = os.stat(dst_file)
                    if s_stat.st_size != d_stat.st_size:
                        print(f"Size differs, skipping: {dst_file}")
                        continue
                    s_sum = sha256sum(src_file)
                    d_sum = sha256sum(dst_file)
                    if s_sum != d_sum:
                        print(f"Checksum differs, skipping: {dst_file}")
                        continue
                except Exception as e:
                    print(f"Error checking files: {e}")
                    continue

                # Remove dst and create hardlink
                try:
                    os.remove(dst_file)
                    os.link(src_file, dst_file)
                    print(f"Replaced with hardlink: {dst_file} -> {src_file}")
                    files_converted += 1
                except Exception as e:
                    print(f"Failed to create hardlink for {dst_file}: {e}")
                    continue
            else:
                # dst doesn't exist; create hardlink directly
                try:
                    os.link(src_file, dst_file)
                    print(f"Hardlinked: {dst_file} -> {src_file}")
                    files_converted += 1
                except Exception as e:
                    print(f"Failed to hardlink {dst_file}: {e}")

    print(f"Done. Files converted to hardlinks: {files_converted}")
    return 0


def main():
    p = argparse.ArgumentParser(description='Convert copied alias files into hardlinks')
    p.add_argument('--src', required=True, help='source root (e.g. data/raw/XBTUSD)')
    p.add_argument('--dst', required=True, help='destination root (e.g. data/raw/XBT/USD)')
    args = p.parse_args()
    return convert(args.src, args.dst)


if __name__ == '__main__':
    raise SystemExit(main())
