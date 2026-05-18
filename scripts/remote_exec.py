import argparse
import os
import time

import paramiko


def safe_print(text: str) -> None:
    printable = text.encode("ascii", errors="ignore").decode("ascii", errors="ignore")
    print(printable, end="", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="trinity")
    parser.add_argument("--username", type=str, default="ra-ugrad")
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--command", type=str, required=True)
    parser.add_argument("--timeout_s", type=int, default=3600)
    parser.add_argument("--heartbeat_s", type=int, default=20)
    args = parser.parse_args()

    password = args.password or os.environ.get("LAB_SERVER_PASSWORD")
    if not password:
        raise ValueError("Provide --password or set LAB_SERVER_PASSWORD.")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.username, password=password, timeout=20)
    try:
        stdin, stdout, stderr = client.exec_command(args.command, get_pty=True)
        ch = stdout.channel
        start = time.time()
        last = start
        while True:
            if ch.recv_ready():
                safe_print(ch.recv(4096).decode("utf-8", errors="replace"))
            if ch.recv_stderr_ready():
                safe_print(ch.recv_stderr(4096).decode("utf-8", errors="replace"))
            if ch.exit_status_ready():
                code = ch.recv_exit_status()
                if code != 0:
                    raise RuntimeError(f"Remote command failed with exit code {code}")
                break
            now = time.time()
            if now - last >= args.heartbeat_s:
                print(f"[HEARTBEAT] running for {int(now - start)}s", flush=True)
                last = now
            if now - start > args.timeout_s:
                ch.close()
                raise TimeoutError(f"Remote command timed out after {args.timeout_s}s")
            time.sleep(0.2)
    finally:
        client.close()


if __name__ == "__main__":
    main()
