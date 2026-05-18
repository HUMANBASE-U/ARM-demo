import argparse
import fnmatch
import os
import posixpath
import stat
import sys
import time
from pathlib import Path
from typing import Iterable, List

import paramiko

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.io import ensure_dir  # noqa: E402


EXCLUDE_PATTERNS = [
    ".git/*",
    ".venv/*",
    "__pycache__/*",
    "data/*",
    "outputs/*",
    "checkpoints/*",
]


def safe_print(text: str) -> None:
    # Avoid Windows console encoding crashes from remote unicode progress bars.
    printable = text.encode("ascii", errors="ignore").decode("ascii", errors="ignore")
    print(printable, end="")


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


def upload_project(sftp: paramiko.SFTPClient, local_root: str, remote_root: str) -> None:
    files = iter_local_files(local_root)
    for rel in files:
        local_path = os.path.join(local_root, rel)
        remote_path = posixpath.join(remote_root, rel.replace("\\", "/"))
        remote_parent = posixpath.dirname(remote_path)
        sftp_mkdir_p(sftp, remote_parent)
        sftp.put(local_path, remote_path)
    print(f"Uploaded {len(files)} files to {remote_root}")


def run_remote_command(client: paramiko.SSHClient, command: str, timeout_s: int, heartbeat_s: int = 20) -> None:
    print(f"\n[REMOTE] {command}")
    stdin, stdout, stderr = client.exec_command(command, get_pty=True)
    channel = stdout.channel
    start = time.time()
    last_heartbeat = start
    while True:
        if channel.recv_ready():
            out = channel.recv(4096).decode("utf-8", errors="replace")
            safe_print(out)
        if channel.recv_stderr_ready():
            err = channel.recv_stderr(4096).decode("utf-8", errors="replace")
            safe_print(err)
        if channel.exit_status_ready():
            code = channel.recv_exit_status()
            if code != 0:
                raise RuntimeError(f"Remote command failed with exit code {code}: {command}")
            break
        now = time.time()
        if now - last_heartbeat >= heartbeat_s:
            elapsed = int(now - start)
            print(f"[REMOTE HEARTBEAT] still running, elapsed={elapsed}s, timeout={timeout_s}s")
            last_heartbeat = now
        if time.time() - start > timeout_s:
            channel.close()
            raise TimeoutError(f"Remote command timed out after {timeout_s}s: {command}")
        time.sleep(0.2)


def sftp_is_dir(sftp: paramiko.SFTPClient, path: str) -> bool:
    try:
        return stat.S_ISDIR(sftp.stat(path).st_mode)
    except FileNotFoundError:
        return False


def download_latest_rollout_video(
    sftp: paramiko.SFTPClient, remote_rollout_dir: str, local_dir: str
) -> str:
    files = []
    for attr in sftp.listdir_attr(remote_rollout_dir):
        if attr.filename.endswith(".mp4"):
            files.append((attr.filename, attr.st_mtime))
    if not files:
        raise FileNotFoundError(f"No rollout .mp4 found in {remote_rollout_dir}")
    files.sort(key=lambda x: x[1], reverse=True)
    latest_name = files[0][0]
    remote_path = posixpath.join(remote_rollout_dir, latest_name)
    ensure_dir(local_dir)
    local_path = os.path.join(local_dir, latest_name)
    sftp.get(remote_path, local_path)
    return local_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="trinity")
    parser.add_argument("--username", type=str, default="ra-ugrad")
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--local_root", type=str, default=".")
    parser.add_argument("--remote_root", type=str, default="~/Ajax/ARM-demo")
    parser.add_argument("--data_dir", type=str, default="data/raw_quick")
    parser.add_argument("--num_episodes", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--save_name", type=str, default="remote_quick")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--skip_sync", action="store_true")
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--skip_collect", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--install_timeout_s", type=int, default=2400)
    parser.add_argument("--collect_timeout_s", type=int, default=3600)
    parser.add_argument("--train_timeout_s", type=int, default=14400)
    parser.add_argument("--rollout_timeout_s", type=int, default=1800)
    parser.add_argument("--heartbeat_s", type=int, default=20)
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
            if not sftp_is_dir(sftp, remote_root):
                run_remote_command(client, f"mkdir -p {remote_root}", timeout_s=20)
            if not args.skip_sync:
                upload_project(sftp, args.local_root, remote_root)
        finally:
            sftp.close()

        if not args.skip_install:
            run_remote_command(
                client,
                f"cd {remote_root} && python -m pip install -r requirements.txt",
                timeout_s=args.install_timeout_s,
                heartbeat_s=args.heartbeat_s,
            )
        if not args.skip_collect:
            run_remote_command(
                client,
                " && ".join(
                    [
                        f"cd {remote_root}",
                        (
                            "python scripts/collect_data.py "
                            f"--output_dir {args.data_dir} "
                            f"--num_episodes {args.num_episodes} "
                            f"--max_steps {args.max_steps} "
                            f"--image_size {args.image_size}"
                        ),
                    ]
                ),
                timeout_s=args.collect_timeout_s,
                heartbeat_s=args.heartbeat_s,
            )
        if not args.skip_train:
            run_remote_command(
                client,
                " && ".join(
                    [
                        f"cd {remote_root}",
                        (
                            "python scripts/train.py "
                            "--config configs/default.yaml "
                            f"--data_dir {args.data_dir} "
                            f"--epochs {args.epochs} "
                            f"--save_name {args.save_name}"
                        ),
                    ]
                ),
                timeout_s=args.train_timeout_s,
                heartbeat_s=args.heartbeat_s,
            )
        run_remote_command(
            client,
            " && ".join(
                [
                    f"cd {remote_root}",
                    (
                        "python scripts/visualize_rollout.py "
                        f"--checkpoint checkpoints/{args.save_name}_best.pt "
                        f"--data_dir {args.data_dir} "
                        f"--horizon {args.horizon}"
                    ),
                ]
            ),
            timeout_s=args.rollout_timeout_s,
            heartbeat_s=args.heartbeat_s,
        )

        sftp = client.open_sftp()
        try:
            local_video = download_latest_rollout_video(
                sftp=sftp,
                remote_rollout_dir=posixpath.join(remote_root, "outputs/rollout_videos"),
                local_dir=os.path.join(args.local_root, "outputs", "rollout_videos_server"),
            )
        finally:
            sftp.close()
        print(f"\nDownloaded rollout video to: {local_video}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
