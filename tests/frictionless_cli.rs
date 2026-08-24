use assert_cmd::Command;

fn isolated_command(root: &std::path::Path) -> Command {
    let home = root.join("home");
    let data = root.join("data");
    let cache = root.join("cache");
    let work = root.join("work");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::create_dir_all(&data).unwrap();
    std::fs::create_dir_all(&cache).unwrap();
    std::fs::create_dir_all(&work).unwrap();
    let mut command = Command::cargo_bin("hf2q").unwrap();
    command
        .current_dir(work)
        .env("HOME", home)
        .env("XDG_DATA_HOME", data)
        .env("XDG_CACHE_HOME", &cache)
        .env("HF_HOME", cache.join("huggingface"))
        .env("HF_HUB_OFFLINE", "1");
    command
}

#[test]
fn serve_and_chat_list_are_identical_read_only_offline_inventory_commands() {
    let root = tempfile::tempdir().unwrap();
    let state_root = root.path().join("malformed-state");
    std::fs::create_dir_all(&state_root).unwrap();
    std::fs::write(state_root.join("config.toml"), b"this = [is not toml").unwrap();
    let serve = isolated_command(root.path())
        .arg("--state-root")
        .arg(&state_root)
        .args(["serve", "list"])
        .output()
        .unwrap();
    let chat = isolated_command(root.path())
        .arg("--state-root")
        .arg(&state_root)
        .args(["chat", "list"])
        .output()
        .unwrap();
    assert!(
        serve.status.success(),
        "{}",
        String::from_utf8_lossy(&serve.stderr)
    );
    assert!(
        chat.status.success(),
        "{}",
        String::from_utf8_lossy(&chat.stderr)
    );
    assert_eq!(serve.stdout, chat.stdout);
    assert_eq!(
        String::from_utf8(serve.stdout).unwrap(),
        "REPOSITORY\tREVISION\tQUANT\tORIGIN\tLAST_USED\tMMPROJ\tPATH\n"
    );
    assert!(!root.path().join("data/hf2q/models").exists());
}

#[test]
fn released_cli_parses_repository_quant_surfaces_without_network() {
    let root = tempfile::tempdir().unwrap();
    for arguments in [
        vec!["serve", "owner/model:Q4_K_M", "--help"],
        vec!["chat", "owner/model:Q4_K_M", "--help"],
        vec!["convert", "owner/model:Q4_K_M", "--help"],
    ] {
        isolated_command(root.path())
            .args(arguments)
            .assert()
            .success();
    }
}

#[cfg(unix)]
#[test]
fn packed_server_adopts_the_inherited_listener_and_retains_port_authority() {
    use std::io::{BufRead, BufReader};
    use std::os::fd::AsRawFd;
    use std::os::unix::process::CommandExt;
    use std::process::Stdio;
    use std::time::Duration;

    let root = tempfile::tempdir().unwrap();
    let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let listener_fd = listener.as_raw_fd();
    let (parent_lifeline, child_lifeline) = std::os::unix::net::UnixStream::pair().unwrap();
    parent_lifeline
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    let child_fd = child_lifeline.as_raw_fd();
    let binary = assert_cmd::cargo::cargo_bin!("hf2q");
    let mut command = std::process::Command::new(binary);
    command
        .arg("--state-root")
        .arg(root.path().join("state"))
        .args(["serve", "--host", "127.0.0.1", "--port"])
        .arg(address.port().to_string())
        .args([
            "--quiet",
            "--operator-ui",
            "plain",
            "--auth-token",
            "secret",
        ])
        .arg("--chat-parent-lifeline-fd")
        .arg(child_fd.to_string())
        .arg("--chat-owned-listener-fd")
        .arg(listener_fd.to_string())
        .env("HOME", root.path().join("home"))
        .env("XDG_DATA_HOME", root.path().join("data"))
        .env("XDG_CACHE_HOME", root.path().join("cache"))
        .env("HF_HUB_OFFLINE", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .process_group(0);
    unsafe {
        command.pre_exec(move || {
            for fd in [child_fd, listener_fd] {
                let flags = libc::fcntl(fd, libc::F_GETFD);
                if flags < 0 || libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    let mut child = command.spawn().unwrap();
    drop(child_lifeline);

    let mut ready = String::new();
    BufReader::new(parent_lifeline.try_clone().unwrap())
        .read_line(&mut ready)
        .unwrap();
    if ready != format!("HF2Q-L1:READY:{}\n", address.port()) {
        let _ = child.kill();
        let stderr = child
            .stderr
            .take()
            .map(|mut stderr| {
                let mut value = String::new();
                std::io::Read::read_to_string(&mut stderr, &mut value).unwrap();
                value
            })
            .unwrap_or_default();
        panic!("unexpected READY frame {ready:?}; server stderr: {stderr}");
    }

    let response = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap()
        .get(format!("http://127.0.0.1:{}/health", address.port()))
        .bearer_auth("secret")
        .send()
        .unwrap();
    assert!(response.status().is_success(), "{}", response.status());

    child.kill().unwrap();
    child.wait().unwrap();
    let replacement = std::net::TcpListener::bind(address).unwrap_err();
    assert_eq!(replacement.kind(), std::io::ErrorKind::AddrInUse);
    drop(listener);
    std::net::TcpListener::bind(address)
        .expect("the endpoint port is released only after retained authority is dropped");
}
