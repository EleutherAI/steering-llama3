import os
import sys
import shutil

def copy_if_larger(src, dst):
    if os.path.exists(dst):
        if os.path.getsize(src) > os.path.getsize(dst):
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)

def recursive_copy(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)

        if os.path.isdir(src_path):
            recursive_copy(src_path, dst_path)
        else:
            copy_if_larger(src_path, dst_path)

def copy_tarball(src, dst_dir):
    # split dst_dir into parent and child
    parent, child = os.path.split(dst_dir)
    # make a temporary directory
    tmp_dir = os.path.join(parent, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    # extract the tarball
    # check extension and use appropriate command
    if src.endswith(".tar.gz") or src.endswith(".tgz"):
        os.system(f"tar -xzf {src} -C {tmp_dir}")
    elif src.endswith(".tar"):
        os.system(f"tar -xf {src} -C {tmp_dir}")
    else:
        raise ValueError(f"Unknown extension: {src}")

    # check that the child directory exists
    if not os.path.exists(os.path.join(tmp_dir, child)):
        raise ValueError(f"Target directory not found in tarball: {child}")

    # copy the contents of the temporary directory
    recursive_copy(os.path.join(tmp_dir, child), dst_dir)

    # remove the temporary directory
    shutil.rmtree(tmp_dir)

def smart_copy(src, dst_dir):
    if os.path.exists(dst_dir) and not os.path.isdir(dst_dir):
        raise ValueError(f"Destination is not a directory: {dst_dir}")

    if os.path.isdir(src):
        recursive_copy(src, dst_dir)
    elif src.endswith(".tar.gz") or src.endswith(".tgz") or src.endswith(".tar"):
        copy_tarball(src, dst_dir)
    else:
        raise ValueError(f"Source is not directory or tarball: {src}")


if __name__ == "__main__":
    # usage: python smartcopy.py src dst_dir
    if len(sys.argv) != 3:
        print("Usage: python smartcopy.py src dst_dir")
        sys.exit(1)
    src = sys.argv[1]
    dst_dir = sys.argv[2]

    smart_copy(src, dst_dir)