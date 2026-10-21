"""observe.route_stdout: per-context line routing that is safe under threads."""

import asyncio
import sys
import threading

from flora_deploy.observe import ContextStdout, route_stdout


def test_routes_lines_and_forwards(capsys):
    got = []
    with route_stdout(got.append):
        print("hello")
        sys.stdout.write("part")
        sys.stdout.write("ial\nlast-no-newline")
    assert got == ["hello", "partial", "last-no-newline"]
    assert capsys.readouterr().out == "hello\npartial\nlast-no-newline"
    assert not isinstance(sys.stdout, ContextStdout)  # uninstalled after the last user


def test_threads_only_see_their_own_lines(capsys):
    sinks = {i: [] for i in range(6)}
    barrier = threading.Barrier(6)

    def worker(i):
        with route_stdout(sinks[i].append):
            barrier.wait()
            for k in range(200):
                print(f"t{i}-{k}")

            async def nested():  # NVIDIA prints from asyncio.to_thread workers too
                await asyncio.to_thread(print, f"t{i}-thread")

            asyncio.run(nested())

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i, lines in sinks.items():
        assert lines == [f"t{i}-{k}" for k in range(200)] + [f"t{i}-thread"]
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 6 * 201 and sorted(out) == sorted(sum(sinks.values(), []))
    assert not isinstance(sys.stdout, ContextStdout)


def test_lines_outside_any_route_are_forwarded_only(capsys):
    got = []
    with route_stdout(got.append):
        t = threading.Thread(target=lambda: print("stranger"))
        t.start()
        t.join()
        print("mine")
    assert got == ["mine"]
    assert "stranger" in capsys.readouterr().out
