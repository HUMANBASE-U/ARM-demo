import argparse
import fnmatch
import os
import posixpath
from pathlib import Path
from typing import Iterable, List

import paramiko


EXCLUDE_PATTERNS = [
    ".git/*",
    ".venv/*",
    "__pycache__/*",
    "data/*",
    "outputs/*",
    "checkpoints/*",
]


def _match_any(path: str, patterns: Iterable[str]) -> bool:
    path = path.replace("\\", "/")
    for p in patterns:
        if fnmatch.fnmatch(path, p):
            return True
    return False


def iter_local_files(root: str) -> List[str]:
    root_path = Path(root)
    files = []
    for p in root_path.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root_path).as_posix()
        if _match_any(rel, EXCLUDE_PATTERNS):
            continue
        files.append(rel)
    return files


def sftp_mkdir_p(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = remote_dir.strip("/").split("/")
    cur = ""
    for part in parts:
        cur = f"{cur}/{part}"
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="trinity")
    parser.add_argument("--username", type=str, default="ra-ugrad")
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--local_root", type=str, default=".")
    parser.add_argument("--remote_root", type=str, default="~/Ajax/ARM-demo")
    args = parser.parse_args()

    password = args.password or os.environ.get("LAB_SERVER_PASSWORD")
    if not password:
        raise ValueError("Provide --password or set LAB_SERVER_PASSWORD.")
    remote_root = args.remote_root.replace("~", f"/home/{args.username}")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.username, password=password, timeout=20)
    try:
        sftp = client.open_sftp()
        try:
            files = iter_local_files(args.local_root)
            for rel in files:
                local_path = os.path.join(args.local_root, rel)
                remote_path = posixpath.join(remote_root, rel.replace("\\", "/"))
                sftp_mkdir_p(sftp, posixpath.dirname(remote_path))
                sftp.put(local_path, remote_path)
            print(f"Uploaded {len(files)} files to {remote_root}")
        finally:
            sftp.close()
    finally:
        client.close()


if __name__ == "__main__":
    main()
