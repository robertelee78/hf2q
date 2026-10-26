"""Owned fixture tools; model strings are never shell commands or compiler flags."""
import hashlib
import json
import os
from pathlib import Path
import stat
import time
import tempfile

from qwen38_n1_verdict import FILE_LIMIT, Invalid, loads, require, test_harness
import qwen38_n1_process as child

MANIFEST = '[package]\nname = "qwen38_n1_fixture"\nversion = "0.0.0"\nedition = "2021"\n'
LOCK = '# Owned dependency-free fixture\nversion = 4\n\n[[package]]\nname = "qwen38_n1_fixture"\nversion = "0.0.0"\n'
LIBRARY = 'pub mod ports;\n#[cfg(test)]\nmod tests {\n    use super::ports::parse_port;\n    #[test]\n    fn zero_and_overflow_rejected() {\n        assert!(parse_port("0").is_err());\n        assert!(parse_port("65536").is_err());\n    }\n    #[test]\n    fn boundary_ports_accepted() {\n        assert_eq!(parse_port("1").unwrap(), 1);\n        assert_eq!(parse_port("65535").unwrap(), 65535);\n    }\n}\n'
INITIAL = 'pub fn parse_port(s: &str) -> u16 { s.parse().unwrap() }\n'
TESTS = '''use qwen38_n1_fixture::ports::parse_port;
#[test]
fn rejects_invalid_ports() {
    for s in ["0", "65536", "invalid", "-1", "999999999999999999999"] {
        assert!(parse_port(s).is_err(), "must reject {s}");
    }
}
#[test]
fn accepts_u16_ports() {
    assert_eq!(parse_port("1").unwrap(), 1);
    assert_eq!(parse_port("65535").unwrap(), 65535);
}
'''
READ_PATHS = {"src/ports.rs", "tests/ports.rs"}
IMMUTABLE = {"Cargo.toml": MANIFEST, "Cargo.lock": LOCK,
             "src/lib.rs": LIBRARY, "tests/ports.rs": TESTS}


class Tools:
    def __init__(self, root, cargo, sandbox, sdk):
        self.root = Path(root)
        self.root.mkdir(mode=0o700)
        self.fixture = self.root / "fixture"
        self.fixture.mkdir(mode=0o700)
        for name in ("src", "tests"):
            (self.fixture / name).mkdir(mode=0o700)
        for name, text in {**IMMUTABLE, "src/ports.rs": INITIAL}.items():
            path = self.fixture / name
            path.write_text(text)
            path.chmod(0o600 if name == "src/ports.rs" else 0o400)
        self.marker = self.root / "owned.json"
        self.marker.write_text(json.dumps({"schema": "hf2q.n1-owned-fixture.v1",
                                          "fixture_inode": self.fixture.stat().st_ino}))
        self.dirfds = {name: os.open(self.fixture / name, os.O_RDONLY | os.O_DIRECTORY)
                       for name in ("src", "tests")}
        self.cargo, self.sandbox = Path(cargo), Path(sandbox)
        self.toolchain, self.sdk = self.cargo.parent.parent, Path(sdk)
        self.receipts = []
        self.calls = 0
        self.source_hash = child.digest(self.fixture / "src/ports.rs")
        self.fixture_identity = (self.fixture.stat().st_dev, self.fixture.stat().st_ino)
        self.cwd = Path(tempfile.mkdtemp(prefix="hf2q-n1-cargo-", dir="/private/tmp")).resolve()
        require(not self.cwd.is_relative_to(Path.home()), "cargo cwd must be outside home")
        for parent in (self.cwd, *self.cwd.parents):
            require(not any((parent / ".cargo" / n).exists() for n in ("config", "config.toml")),
                    "ancestor Cargo config could override fixed compiler")
        self.marker.write_text(json.dumps({"schema": "hf2q.n1-owned-fixture.v1",
                                          "fixture_inode": self.fixture.stat().st_ino,
                                          "cargo_cwd": str(self.cwd)}))

    def close(self):
        for fd in self.dirfds.values():
            os.close(fd)
        self.dirfds.clear()

    def invariant(self):
        st = self.fixture.lstat()
        require(not stat.S_ISLNK(st.st_mode) and (st.st_dev, st.st_ino) == self.fixture_identity,
                "owned fixture replaced")
        for name, text in IMMUTABLE.items():
            path = self.fixture / name
            require(not path.is_symlink() and path.read_bytes() == text.encode(), "immutable fixture changed")
        for name, fd in self.dirfds.items():
            st = (self.fixture / name).lstat()
            held = os.fstat(fd)
            require(stat.S_ISDIR(st.st_mode) and (st.st_dev, st.st_ino) ==
                    (held.st_dev, held.st_ino), "fixture subdirectory replaced")

    def read(self, name):
        require(type(name) is str and name in READ_PATHS, "read path allowlist")
        directory, leaf = name.split("/")
        fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self.dirfds[directory])
        try:
            st = os.fstat(fd)
            require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and st.st_uid == os.getuid(),
                    "unsafe read owner/type/links")
            require(st.st_size <= FILE_LIMIT, "whole file exceeds64KiB")
            data = os.read(fd, FILE_LIMIT + 1)
            require(len(data) == st.st_size and len(data) <= FILE_LIMIT, "file changed/oversized")
            return data.decode("utf8", "strict")
        finally:
            os.close(fd)

    def write(self, name, content):
        require(name == "src/ports.rs" and isinstance(content, str), "write path/type")
        raw = content.encode()
        require(len(raw) <= FILE_LIMIT, "write exceeds64KiB; no partial write")
        self.read(name)  # validate old owner; never follow a link for replacement
        fd = os.open("ports.rs.new", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=self.dirfds["src"])
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace("ports.rs.new", "ports.rs", src_dir_fd=self.dirfds["src"],
                       dst_dir_fd=self.dirfds["src"])
        except BaseException:
            try:
                os.unlink("ports.rs.new", dir_fd=self.dirfds["src"])
            except FileNotFoundError:
                pass
            raise
        self.source_hash = hashlib.sha256(raw).hexdigest()
        return "write complete; sha256=" + self.source_hash

    def tests(self, target, log):
        require(target in {"ports", "all"}, "test target allowlist")
        started = time.monotonic()
        build = log / "target"
        build.mkdir(mode=0o700)
        temporary, logs, home = [log / n for n in ("tmp", "logs", "home")]
        for path in (temporary, logs, home):
            path.mkdir(mode=0o700)
        cargo_home = temporary / "cargo-home"
        cargo_home.mkdir(mode=0o700)
        env = {"PATH": f"{self.cargo.parent}:/usr/bin:/bin", "HOME": str(home),
               "CARGO_HOME": str(cargo_home), "CARGO_TARGET_DIR": str(build),
               "TMPDIR": str(temporary), "LANG": "C", "LC_ALL": "C",
               "CARGO_NET_OFFLINE": "true", "CARGO_INCREMENTAL": "0"}
        env.update(child.compiler_environment(self.cargo, self.sdk))
        argv = [str(self.cargo), "test", "--offline", "--locked", "--no-run",
                "--message-format=json", "--manifest-path", str(self.fixture / "Cargo.toml")]
        argv += ["--test", "ports"] if target == "ports" else ["--tests"]
        sb = child.profile("compile", self.fixture, build, temporary, logs, self.cargo,
                           self.toolchain, self.sdk, self.cwd)
        receipt, stdout, stderr = child.run(argv, self.cwd, env, self.sandbox, sb,
                                            logs / "compile", build, writable_root=log)
        observed = [receipt]
        used = len(stdout) + len(stderr)
        output = bytearray(stderr)
        artifacts = []
        for line in stdout.splitlines():
            item = loads(line)
            if item.get("reason") == "compiler-message":
                output.extend((item.get("message", {}).get("rendered") or "").encode())
            if item.get("reason") == "compiler-artifact" and item.get("executable"):
                path = Path(item["executable"])
                require(not path.is_symlink() and path.is_file() and
                        path.resolve().is_relative_to(build.resolve()), "foreign test artifact")
                require(item.get("profile", {}).get("test") is True, "artifact not test")
                require(item.get("target", {}).get("name") in {"ports", "qwen38_n1_fixture"},
                        "unexpected test target")
                artifacts.append(path.resolve())
        passed = receipt["exit_code"] == 0
        if passed:
            require(len(artifacts) == (1 if target == "ports" else 2) and
                    len(set(artifacts)) == len(artifacts), "missing/duplicate test artifact")
            for index, executable in enumerate(artifacts):
                require(time.monotonic() - started < 45, "whole tool deadline")
                before_hash = child.digest(executable)
                sb = child.profile("runtime", self.fixture, build, temporary, logs, executable,
                                   self.toolchain, self.sdk, self.cwd)
                result, out, err = child.run([str(executable), "--test-threads=1"], self.cwd,
                                             env, self.sandbox, sb, logs / f"test-{index}", build,
                                             byte_limit=child.TOOL_LIMIT - used,
                                             deadline=min(45, 45 - (time.monotonic() - started)),
                                             writable_root=log)
                require(child.digest(executable) == before_hash, "test artifact changed")
                used += len(out) + len(err)
                output.extend(out + err)
                observed.append(result)
                if result["exit_code"] == 0:
                    result["actual_tests_passed"] = test_harness(out, 2)
                passed &= result["exit_code"] == 0
        require(len(output) <= FILE_LIMIT and used <= FILE_LIMIT, "whole tool output envelope")
        self.invariant()
        require(child.digest(self.fixture / "src/ports.rs") == self.source_hash, "compiler/test altered source")
        return {"target": target, "passed": passed, "source_sha256": self.source_hash,
                "fresh_target": str(build), "artifact_sha256": [child.digest(p) for p in artifacts],
                "processes": observed, "cleanup": all(x["cleanup"] for x in observed),
                "output": output.decode("utf8", "strict")}

    def invoke(self, name, args):
        self.invariant()
        self.calls += 1
        require(self.calls <= 6, "extra tool invocation")
        log = self.root / f"tool-{self.calls:02d}"
        log.mkdir(mode=0o700)
        receipt = {"name": name, "arguments": args, "cleanup": False, "status": "Pending"}
        self.receipts.append(receipt)
        try:
            if name == "read_file":
                require(set(args) == {"path"}, "read keys")
                result = self.read(args["path"])
            elif name == "write_file":
                require(set(args) == {"path", "content"}, "write keys")
                result = self.write(args["path"], args["content"])
            elif name == "run_tests":
                require(set(args) == {"target"}, "test keys")
                detail = self.tests(args["target"], log)
                receipt.update(detail)
                result = json.dumps({k: detail[k] for k in ("target", "passed", "source_sha256", "output")},
                                    ensure_ascii=False)
            else:
                raise Invalid("unknown tool")
            require(len(result.encode()) <= FILE_LIMIT, "tool result exceeds64KiB; no clipped continuation")
            self.invariant()
            receipt.update({"result": result, "source_sha256": self.source_hash,
                            "cleanup": True, "status": "Complete"})
            return result
        except BaseException as exc:
            # Nested process receipts preserve raw cleanup evidence. Never infer it from no return.
            receipt.update({"status": "Failed", "cleanup": False, "error": str(exc)})
            raise
        finally:
            (log / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
