"""ADR-063 Q5 model/projector identity and the text-only model locator.

Identity never hashes a model inline. The sha256 comes from the adjacent hf2q
conversion receipt `<artifact>.receipt.json`, field `output.sha256`, accepted only
when that receipt's `output.size` equals the file's size. Otherwise the identity is
size + mtime (plus any operator-provided digest, labeled unverified).

`hf2q serve` without `--mmproj` automatically loads a sibling projector
(`mmproj*` / `*-mmproj*.gguf`) or one bound by the conversion receipt beside the
text GGUF (src/serve/mod.rs automatic local-path resolution). The launchers'
QWEN38_VISION=off / missing MMPROJ only omit `--mmproj`, so they cannot disable it.
A text-only run therefore serves a hard link of the exact same file from a private
directory that contains nothing else.
"""
import hashlib
import json
import os
import re

RECEIPT_SUFFIX = '.receipt.json'
MAX_RECEIPT_BYTES = 1 << 20
INPUT_FILE_LIMIT = 64 << 20  # argv files hashed into the receipt; models are never hashed inline
LOCATOR_DIR = 'text-model'


class ArtifactError(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def input_hashes(plan):
    """sha256 of every small regular file named in serve/client argv plus the approval template."""
    argv = plan['serve']['argv'] + (plan['client']['argv'] if plan.get('client') else [])
    if plan.get('client') and plan['client'].get('approval'):
        argv = argv + [plan['client']['approval']['template']]
    return {item: sha256(item) for item in argv
            if os.path.isabs(item) and os.path.isfile(item) and os.path.getsize(item) <= INPUT_FILE_LIMIT}


def file_checks(plan):
    paths = {'binary': plan['binary'], 'serve.argv[0]': plan['serve']['argv'][0]}
    if plan.get('artifact'):
        paths['artifact'] = plan['artifact']['path']
    if plan.get('client'):
        paths['client.argv[0]'] = plan['client']['argv'][0]
        if plan['client'].get('approval'):
            paths['client.approval.template'] = plan['client']['approval']['template']
    return {name: os.path.isfile(path) for name, path in paths.items()}


def artifact_identity(path, provided_sha256=None):
    stat = os.stat(path)
    identity = {'path': path, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': None,
                'sha256_source': None, 'provided_sha256': provided_sha256, 'sha256_hashed_by_supervisor': False}
    receipt = path + RECEIPT_SUFFIX
    if os.path.isfile(receipt):
        if os.path.getsize(receipt) > MAX_RECEIPT_BYTES:
            identity['receipt_rejected'] = 'receipt larger than 1 MiB'
        else:
            try:
                with open(receipt, 'rb') as handle:
                    output = json.loads(handle.read()).get('output') or {}
            except (OSError, ValueError, AttributeError) as error:
                output, identity['receipt_rejected'] = {}, 'unreadable receipt: ' + repr(error)
            digest = output.get('sha256')
            if output and output.get('size') != stat.st_size:
                identity['receipt_rejected'] = 'receipt output.size differs from the file size'
            elif isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest):
                identity['sha256'] = digest
                identity['sha256_source'] = {'receipt': receipt, 'field': 'output.sha256',
                                             'receipt_sha256': sha256(receipt),
                                             'receipt_output_path_matches': output.get('path') == path}
            elif 'receipt_rejected' not in identity:
                identity['receipt_rejected'] = 'receipt has no 64-hex output.sha256'
    if provided_sha256:
        if identity['sha256'] and identity['sha256'] != provided_sha256:
            raise ArtifactError(f'provided sha256 {provided_sha256} contradicts {receipt} output.sha256')
        if not identity['sha256']:
            identity['sha256'] = provided_sha256
            identity['sha256_source'] = 'operator-provided, not verified by the supervisor'
    return identity


def projector_candidates(model_path):
    """Projectors hf2q could auto-load beside `model_path` (same name rule as the server)."""
    parent = os.path.dirname(model_path)
    found = []
    for name in sorted(os.listdir(parent)):
        full = os.path.join(parent, name)
        lower = name.lower()
        if full == model_path or not lower.endswith('.gguf') or not os.path.isfile(full):
            continue
        if lower.startswith('mmproj') or '-mmproj' in lower:
            found.append(artifact_identity(full))
    return found


def create_locator(model_path, out):
    """Hard link the exact model into `out/text-model/` (nothing else in that directory)."""
    directory = os.path.join(out, LOCATOR_DIR)
    os.mkdir(directory, 0o700)
    link = os.path.join(directory, os.path.basename(model_path))
    try:
        os.link(model_path, link)
    except OSError as error:
        raise ArtifactError(f'cannot hard link the model into the receipt directory (same volume required): {error}') from error
    source, linked = os.stat(model_path), os.stat(link)
    if (source.st_dev, source.st_ino) != (linked.st_dev, linked.st_ino):
        raise ArtifactError('text-only locator is not the same file')
    if os.listdir(directory) != [os.path.basename(model_path)]:
        raise ArtifactError('text-only locator directory is not private')
    return {'served_path': link, 'directory': directory, 'dev': linked.st_dev, 'ino': linked.st_ino,
            'same_file_as_artifact': True}


def remove_locator(locator):
    """Remove the hard link and anything the server created beside it; never follows links out."""
    removed, errors = [], []
    directory = locator['directory']
    try:
        for name in sorted(os.listdir(directory)):
            full = os.path.join(directory, name)
            try:
                if os.path.isfile(full) and not os.path.islink(full):
                    os.unlink(full)
                    removed.append(name)
                else:
                    errors.append('left non-regular entry: ' + name)
            except OSError as error:
                errors.append(repr(error))
        os.rmdir(directory)
    except OSError as error:
        errors.append(repr(error))
    return {'removed': removed, 'errors': errors, 'directory_removed': not os.path.exists(directory)}
