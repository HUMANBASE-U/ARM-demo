import argparse
import os
import posixpath
import stat

import paramiko


def is_dir(sftp: paramiko.SFTPClient, remote_path: str) -> bool:
    return stat.S_ISDIR(sftp.stat(remote_path).st_mode)


def download_recursive(sftp: paramiko.SFTPClient, remote_path: str, local_path: str) -> None:
    if is_dir(sftp, remote_path):
        os.makedirs(local_path, exist_ok=True)
        for entry in sftp.listdir_attr(remote_path):
            child_remote = posixpath.join(remote_path, entry.filename)
            child_local = os.path.join(local_path, entry.filename)
            if stat.S_ISDIR(entry.st_mode):
                download_recursive(sftp, child_remote, child_local)
            else:
                os.makedirs(os.path.dirname(child_local), exist_ok=True)
                sftp.get(child_remote, child_local)
    else:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        sftp.get(remote_path, local_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="trinity")
    parser.add_argument("--username", type=str, default="ra-ugrad")
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--remote_path", type=str, required=True)
    parser.add_argument("--local_path", type=str, required=True)
    args = parser.parse_args()

    password = args.password or os.environ.get("LAB_SERVER_PASSWORD")
    if not password:
        raise ValueError("Provide --password or set LAB_SERVER_PASSWORD.")

    remote_path = args.remote_path.replace("~", f"/home/{args.username}")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.username, password=password, timeout=20)
    try:
        sftp = client.open_sftp()
        try:
            download_recursive(sftp, remote_path, args.local_path)
            print(f"Downloaded: {remote_path} -> {args.local_path}")
        finally:
            sftp.close()
    finally:
        client.close()


if __name__ == "__main__":
    main()
