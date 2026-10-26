"""Pure V2 same-invocation physical accounting parser; no host queries/import work."""
import csv
import datetime
import io
import re

CATEGORIES = ('anonymous', 'cached_files', 'compressor', 'free', 'wired', 'static_carveouts')
NAMES = {'Anonymous': 'anonymous', 'Cached files': 'cached_files',
         'Compressor': 'compressor', 'Free': 'free', 'Wired': 'wired',
         'Static carveouts': 'static_carveouts'}
# Pin the observed summary grammar, not rounded MB values as accounting inputs.
INTEGER = r'(?:0|[1-9][0-9]{0,2}(?:,[0-9]{3})*)'
AMOUNT = rf'(?P<pages>{INTEGER}) pages\s+{INTEGER}\.[0-9] MB'
HEADER = re.compile(rf'Physical Memory:\s+{AMOUNT}')
SWAP = re.compile(rf'Swap:\s+{AMOUNT}')
SWAPPED = re.compile(rf'└─ Swapped data:\s+{AMOUNT}')
ROW = re.compile(rf'(?P<branch>├─|└─) (?P<name>[A-Za-z ]+):\s+{AMOUNT}'
                 r'\s+[█░]{20}\s+[0-9]{1,3}\.[0-9]{2}%')


class Invalid(ValueError):
    pass


def parse_csv(text):
    try:
        rows = list(csv.reader(io.StringIO(text), strict=True))
    except csv.Error as exc:
        raise Invalid('malformed CSV') from exc
    if len(rows) != 2 or not rows[0] or len(set(rows[0])) != len(rows[0]):
        raise Invalid('expected unique CSV header and exactly one sample')
    header, values = rows
    if len(header) != len(values):
        raise Invalid('CSV row width')
    data = dict(zip(header, values))
    if any(key not in data for key in (*CATEGORIES, 'total', 'datetime')):
        raise Invalid('missing required CSV column')
    selected = {}
    for key in (*CATEGORIES, 'total'):
        if re.fullmatch(r'[0-9]+', data[key]) is None:
            raise Invalid('nonnegative CSV integer required: ' + key)
        selected[key] = int(data[key])
    if sum(selected[key] for key in CATEGORIES) != selected['total']:
        raise Invalid('CSV categories do not reconcile to subtotal')
    try:
        stamp = datetime.datetime.fromisoformat(data['datetime'].replace('Z', '+00:00'))
        if stamp.tzinfo is None:
            raise ValueError('timezone required')
        selected['csv_wall_ns'] = int(stamp.timestamp() * 1_000_000_000)
    except (ValueError, OverflowError) as exc:
        raise Invalid('invalid CSV datetime') from exc
    return selected


def page_count(match):
    return int(match['pages'].replace(',', ''))


def parse_summary(text):
    """Only the exact top-level physical tree can corroborate CSV categories.

    A complete Swap header terminates that tree. Its optional Swapped data row
    is shape-checked but never counted as physical or an Unaccounted witness.
    """
    if re.search(r'\b(?:warning|error)\b', text, re.IGNORECASE):
        raise Invalid('summary warning/error')
    lines = text.splitlines()
    if not lines or not (header := HEADER.fullmatch(lines[0])):
        raise Invalid('missing or malformed Physical Memory header')
    physical = {}
    closed = False
    swap_seen = False
    swapped_seen = False
    for line in lines[1:]:
        if not line:
            if not closed:
                raise Invalid('physical tree truncated before terminal branch')
            continue
        if swap_seen:
            if not swapped_seen and SWAPPED.fullmatch(line):
                swapped_seen = True
                continue
            raise Invalid('unexpected/duplicate/nested Swap content')
        if SWAP.fullmatch(line):
            if not closed:
                raise Invalid('physical tree lacks terminal branch')
            swap_seen = True
            continue
        if closed or not (row := ROW.fullmatch(line)):
            raise Invalid('unexpected/nested/malformed physical row')
        name = row['name']
        if name not in NAMES and name != 'Unaccounted':
            raise Invalid('unknown physical root: ' + name)
        if name in physical:
            raise Invalid('duplicate physical root: ' + name)
        physical[name] = page_count(row)
        closed = row['branch'] == '└─'
    if not swap_seen or set(physical) - {'Unaccounted'} != set(NAMES):
        raise Invalid('incomplete physical summary')
    return {'header_pages': page_count(header), 'category_pages': physical}


def parse_interface(csv_text, stdout_text, capacity, page_bytes):
    """Corroborate one invocation. Positive residual stays fully occupied."""
    if not isinstance(capacity, int) or not isinstance(page_bytes, int) or min(capacity, page_bytes) <= 0:
        raise Invalid('invalid independent capacity/page size')
    selected = parse_csv(csv_text)
    summary = parse_summary(stdout_text)
    subtotal = selected.pop('total')
    if subtotal > capacity:
        raise Invalid('CSV subtotal exceeds physical capacity')
    residual = capacity - subtotal
    if residual % page_bytes or summary['header_pages'] * page_bytes != capacity:
        raise Invalid('physical header or residual/page mismatch')
    pages = summary['category_pages']
    for name, key in NAMES.items():
        if pages[name] * page_bytes != selected[key]:
            raise Invalid('CSV/text category mismatch: ' + name)
    if residual and 'Unaccounted' not in pages:
        raise Invalid('positive residual lacks explicit Unaccounted row')
    if pages.get('Unaccounted', 0) * page_bytes != residual:
        raise Invalid('Unaccounted row does not explain residual')
    if sum(pages.values()) != summary['header_pages']:
        raise Invalid('physical page partition mismatch')
    return {**selected, 'physical_capacity': capacity,
            'csv_accounted_subtotal': subtotal, 'unknown_residual_bytes': residual,
            'occupied': capacity - selected['free'], 'summary_pages': summary,
            'unknown_residual_fully_charged': True}


def retryable_single_page_overcount(csv_text, stdout_text, capacity, page_bytes):
    """Recognize one rejected observation; NEVER return physical/admission values.

    Only a complete same-invocation CSV/text witness of subtotal=P+one page
    with an explicit -1-page root qualifies for a bounded fresh observation.
    This does not establish causality/atomicity, excuse low Free or pressure,
    or make the rejected reading admissible. parse_interface stays strict.
    The caller owns raw preservation, retry/deadline bounds and all guards.
    """
    if (type(capacity) is not int or type(page_bytes) is not int
            or min(capacity, page_bytes) <= 0 or capacity % page_bytes):
        return False
    negative = re.compile(r'└─ Unaccounted:\s+-1 pages\s+-0\.0 MB'
                          r'\s+░{20}\s+-0\.00%')
    lines = stdout_text.splitlines()
    matches = [i for i, line in enumerate(lines) if negative.fullmatch(line)]
    if len(matches) != 1:
        return False
    # Normalize ONLY for shape/category corroboration, never for admission.
    lines[matches[0]] = '└─ Unaccounted: 1 pages 0.0 MB  ' + '░' * 20 + ' 0.00%'
    try:
        selected = parse_csv(csv_text)
        summary = parse_summary('\n'.join(lines))
    except Invalid:
        return False
    pages = summary['category_pages']
    return (selected['total'] == capacity + page_bytes
            and summary['header_pages'] * page_bytes == capacity
            and pages.get('Unaccounted') == 1
            and all(pages[name] * page_bytes == selected[key]
                    for name, key in NAMES.items())
            and sum(pages[name] for name in NAMES) == summary['header_pages'] + 1)
