//! Metadata is bounded while streaming, before native execution/readback.
//! No unbounded serde_json::Value tree or serialized Vec is retained.
use super::*;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, io::{BufWriter, Write}, os::unix::fs::OpenOptionsExt};

struct Sink<'a> {
    file: fs::File,
    hash: Sha256,
    budget: &'a RefCell<super::matrix_plan::Budget>,
}

impl Write for Sink<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.budget.borrow_mut().reserve_metadata(bytes.len() as u64)
            .map_err(std::io::Error::other)?;
        self.file.write_all(bytes)?;
        self.hash.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { self.file.flush() }
}

pub(super) fn write(path: &Path, budget: &RefCell<super::matrix_plan::Budget>,
    value: &impl Serialize) -> Result<String> {
    let file = fs::OpenOptions::new().create_new(true).write(true).mode(0o600).open(path)?;
    let mut writer = BufWriter::with_capacity(32 * 1024, Sink { file, hash: Sha256::new(), budget });
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    let sink = writer.into_inner().map_err(|error| error.into_error())?;
    sink.file.sync_all()?;
    Ok(format!("{:x}", sink.hash.finalize()))
}

pub(super) fn hash_file(path: &Path, budget: &RefCell<super::matrix_plan::Budget>) -> Result<String> {
    let mut file = fs::File::open(path)?;
    // Count the actual producer's bounded metadata before allocating/reading it.
    let length = file.metadata()?.len();
    budget.borrow_mut().reserve_metadata(length)?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 32 * 1024];
    let mut total = 0_u64;
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 { break; }
        total = total.checked_add(count as u64).context("metadata file size overflow")?;
        ensure!(total <= length, "metadata file grew during binding");
        hash.update(&buffer[..count]);
    }
    ensure!(total == length, "metadata file changed during binding");
    Ok(format!("{:x}", hash.finalize()))
}
