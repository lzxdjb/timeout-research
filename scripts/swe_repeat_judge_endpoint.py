"""Resolve a cross-pod judge endpoint and announce it once vLLM is healthy."""

import argparse
import ipaddress
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener


def usable_ip(value):
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return not (
        address.is_unspecified
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
    )


def route_source():
    # UDP connect selects a source address locally; it sends no packets.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("1.1.1.1", 80))
            return sock.getsockname()[0]
    except OSError:
        return ""


def hostname_addresses():
    for flag in ("-I", "-i"):
        try:
            result = subprocess.run(["hostname", flag], capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                yield from result.stdout.split()
        except (OSError, subprocess.TimeoutExpired):
            continue


def detect_host():
    for name in ("POD_IP", "MY_POD_IP"):
        value = os.environ.get(name, "")
        if usable_ip(value) and ipaddress.ip_address(value).version == 4:
            return value
    value = route_source()
    if usable_ip(value):
        return value
    for value in hostname_addresses():
        if usable_ip(value) and ipaddress.ip_address(value).version == 4:
            return value
    raise ValueError("Cannot detect a usable pod IPv4 address; check the pod network configuration")


def validate_url(value):
    try:
        url = urlsplit(value)
        host = url.hostname
        if (
            url.scheme not in {"http", "https"}
            or not host
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or any(c.isspace() for c in value)
        ):
            raise ValueError("expected an HTTP(S) URL without credentials, whitespace, query or fragment")
        if url.netloc.endswith(":") or (url.port is not None and not 1 <= url.port <= 65535):
            raise ValueError("invalid port")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            labels = host.rstrip(".").split(".")
            if (
                host.rstrip(".").lower() == "localhost"
                or len(host) > 253
                or not all(
                    re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
                    for label in labels
                )
            ):
                raise ValueError("invalid client hostname")
        else:
            if not usable_ip(host):
                raise ValueError("client IP must not be wildcard, loopback, link-local, multicast or reserved")
    except ValueError as exc:
        raise ValueError(f"Invalid judge client endpoint {value!r}: {exc}") from exc
    return value


def resolve_endpoint(bind_host, port):
    override = os.environ.get("SWE_REPEAT_JUDGE_ADVERTISE_URL")
    if override:
        return validate_url(override)
    host = os.environ.get("SWE_REPEAT_JUDGE_ADVERTISE_HOST")
    if not host:
        # A concrete bind host must also be used by clients.
        host = detect_host() if bind_host in {"0.0.0.0", "::", "[::]"} else bind_host
    if any(character in host for character in "/?#@"):
        raise ValueError("Invalid judge advertise host: expected only an IP address or DNS name")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return validate_url(f"http://{host}:{port}")


def announce_when_ready(parent_pid, bind_host, port, endpoint):
    host = {"0.0.0.0": "127.0.0.1", "::": "[::1]", "[::]": "[::1]"}.get(bind_host, bind_host)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    health_url = f"http://{host}:{port}/health"
    # A local readiness request must bypass HTTP_PROXY/HTTPS_PROXY.
    opener = build_opener(ProxyHandler({}))
    while os.getppid() == parent_pid:
        try:
            with opener.open(health_url, timeout=2) as response:
                ready = response.status == 200
        except (OSError, ValueError):
            ready = False
        if ready and os.getppid() == parent_pid:
            print(
                f"\nJudge ready. Client endpoint: {endpoint}\n"
                f"export SWE_AGENT_REPEAT_JUDGE_URL={shlex.quote(endpoint)}\n"
                "Run judge-preflight from the client pod to verify connectivity.",
                file=sys.stderr,
                flush=True,
            )
            return
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    resolve = commands.add_parser("resolve")
    resolve.add_argument("bind_host")
    resolve.add_argument("port", type=int)
    watch = commands.add_parser("watch")
    watch.add_argument("parent_pid", type=int)
    watch.add_argument("bind_host")
    watch.add_argument("port", type=int)
    watch.add_argument("endpoint")
    args = parser.parse_args()
    if args.command == "resolve":
        try:
            print(resolve_endpoint(args.bind_host, args.port))
        except ValueError as exc:
            parser.exit(2, f"{exc}\n")
    else:
        announce_when_ready(args.parent_pid, args.bind_host, args.port, args.endpoint)


if __name__ == "__main__":
    main()
