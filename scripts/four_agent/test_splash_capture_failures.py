"""Isolated raw-capture regressions; no inference, model files or listening server."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import native_peer_splash_ready as ready
from qwen38_n1_verdict import Invalid


BASE = 'http://127.0.0.1:18256/v1'


class CaptureFixtures(unittest.TestCase):
    def archive(self, root, turn, label=None, dirname=None):
        directory = root / (dirname or f'turn-{turn}')
        directory.mkdir(parents=True)
        request = {'model': ready.sp.MODEL_IDS['qwen38'],
                   'messages': [{'role': 'user', 'content': label or f'actual request {turn}: λ'}],
                   'tools': [], 'chat_template_kwargs': {'enable_thinking': True}, 'preserve_thinking': True}
        (directory / 'request.json').write_bytes(json.dumps(request, ensure_ascii=False).encode() + b'\n')
        return directory

    def prompt(self, directory):
        request = json.loads((directory / 'request.json').read_bytes())
        return 'native: ' + '|'.join(message['content'] for message in request['messages'])

    def render_bytes(self, prompt):
        return b'{ "prompt": ' + json.dumps(prompt, ensure_ascii=False).encode() + b' }\n'

    def token_bytes(self, prompt):
        return json.dumps({'tokens': list(prompt.encode())}, separators=(',', ':')).encode() + b'\n'

    def fetch(self, _port, endpoint, directory, name, body=None, *args):
        directory = Path(directory)
        if endpoint == '/apply-template':
            request = json.loads((directory / 'request.json').read_bytes())
            self.assertEqual(body, {key: request[key] for key in ready.TEMPLATE_FIELDS})
            encoded = self.render_bytes(self.prompt(directory))
        else:
            self.assertEqual(endpoint, '/tokenize')
            self.assertEqual(body, {'content': self.prompt(directory), 'add_special': False})
            encoded = self.token_bytes(body['content'])
        (directory / name).write_bytes(encoded)
        return encoded

    def receipt(self, directory, turn):
        return {'turn': turn,
                'request_sha256': hashlib.sha256((directory / 'request.json').read_bytes()).hexdigest(),
                'usage': {'prompt_tokens': len(self.prompt(directory).encode())}}

    def summary(self, agent, receipts):
        (agent / 'summary.json').write_text(json.dumps({'turns': receipts}))

    def assert_captured(self, directory):
        prompt = self.prompt(directory)
        self.assertEqual((directory / 'splash-apply-template.raw').read_bytes(), self.render_bytes(prompt))
        self.assertEqual((directory / 'splash-tokenize.raw').read_bytes(), self.token_bytes(prompt))

    def assert_no_retries(self, fetch):
        attempts = [(Path(call.args[2]), call.args[1]) for call in fetch.call_args_list]
        self.assertEqual(len(attempts), len(set(attempts)))
        self.assertTrue(all(endpoint in ('/apply-template', '/tokenize') for _, endpoint in attempts))


class RenderProofCaptureFailures(CaptureFixtures):
    def test_bad_validated_usage_or_hash_preserves_all_actual_raw_bytes(self):
        for failure, reason in (('usage', 'usage.prompt_tokens'), ('hash', 'archive differs from its receipt')):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                directories = [self.archive(root, turn) for turn in range(3)]
                originals = [(directory / 'request.json').read_bytes() for directory in directories]
                receipts = [self.receipt(directory, turn) for turn, directory in enumerate(directories[:2])]
                if failure == 'usage':
                    receipts[0]['usage']['prompt_tokens'] += 1
                else:
                    receipts[0]['request_sha256'] = '0' * 64
                with mock.patch.object(ready, 'fetch', side_effect=self.fetch) as fetch, \
                        self.assertRaisesRegex(Invalid, reason) as error:
                    ready.render_proof(BASE, root, receipts)
                self.assertIn('turn-0', str(error.exception))
                for directory, original in zip(directories, originals):
                    self.assert_captured(directory)
                    self.assertEqual((directory / 'request.json').read_bytes(), original)
                self.assert_no_retries(fetch)

    def test_unreachable_unvalidated_endpoint_still_captures_later_request(self):
        for failed_endpoint in ('/apply-template', '/tokenize'):
            with self.subTest(endpoint=failed_endpoint), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                first, later = [self.archive(root, turn) for turn in range(2)]

                def fetch(*args):
                    if Path(args[2]) == first and args[1] == failed_endpoint:
                        raise OSError(f'unreachable {failed_endpoint}')
                    return self.fetch(*args)

                with mock.patch.object(ready, 'fetch', side_effect=fetch) as captured, \
                        self.assertRaisesRegex(Invalid, f'turn-0: OSError: unreachable {failed_endpoint}'):
                    ready.render_proof(BASE, root, [])
                self.assert_captured(later)
                self.assertFalse((first / 'splash-tokenize.raw').exists())
                if failed_endpoint == '/tokenize':
                    self.assertEqual((first / 'splash-apply-template.raw').read_bytes(),
                                     self.render_bytes(self.prompt(first)))
                self.assert_no_retries(captured)

    def test_invalid_render_or_tokenize_response_keeps_raw_and_later_capture(self):
        cases = (('/apply-template', b'{"wrong":"envelope"}', 'Splash /apply-template envelope'),
                 ('/apply-template', b'{', 'turn-0'),
                 ('/tokenize', b'{"tokens":[true]}', 'Splash /tokenize envelope'))
        for failed_endpoint, raw, reason in cases:
            with self.subTest(endpoint=failed_endpoint, raw=raw), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                first, later = [self.archive(root, turn) for turn in range(2)]

                def fetch(*args):
                    if Path(args[2]) == first and args[1] == failed_endpoint:
                        (Path(args[2]) / args[3]).write_bytes(raw)
                        return raw
                    return self.fetch(*args)

                with mock.patch.object(ready, 'fetch', side_effect=fetch) as captured, \
                        self.assertRaisesRegex(Invalid, reason):
                    ready.render_proof(BASE, root, [])
                artifact = 'splash-apply-template.raw' if failed_endpoint == '/apply-template' else 'splash-tokenize.raw'
                self.assertEqual((first / artifact).read_bytes(), raw)
                self.assert_captured(later)
                self.assert_no_retries(captured)

    def test_malformed_request_does_not_hide_later_actual_archive(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            first, later = [self.archive(root, turn) for turn in range(2)]
            (first / 'request.json').write_bytes(b'{')
            with mock.patch.object(ready, 'fetch', side_effect=self.fetch), \
                    self.assertRaisesRegex(Invalid, 'turn-0'):
                ready.render_proof(BASE, root, [])
            self.assert_captured(later)
            self.assertEqual((first / 'request.json').read_bytes(), b'{')

    def test_receipt_and_archive_integrity_errors_do_not_skip_actual_requests(self):
        cases = ('missing-archive', 'duplicate-receipt', 'malformed-receipt', 'malformed-collection',
                 'noncanonical-directory', 'nonnumeric-directory')
        for failure in cases:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                dirname = {'noncanonical-directory': 'turn-00', 'nonnumeric-directory': 'turn-interrupted'}.get(failure)
                first = self.archive(root, 0, dirname=dirname)
                later = self.archive(root, 1)
                receipt = self.receipt(first, 0)
                receipts, reason = [], ''
                if failure == 'missing-archive':
                    receipts = [{**receipt, 'turn': 99}]
                    reason = 'turn-99: FileNotFoundError'
                elif failure == 'duplicate-receipt':
                    receipts, reason = [receipt, receipt], 'duplicate turn receipt'
                elif failure == 'malformed-receipt':
                    receipts, reason = [None, self.receipt(later, 1)], 'malformed turn receipt'
                elif failure == 'malformed-collection':
                    receipts, reason = {'turn': receipt}, 'expected a list'
                elif failure == 'noncanonical-directory':
                    reason = 'noncanonical turn directory'
                else:
                    reason = 'turn-interrupted: ValueError'
                with mock.patch.object(ready, 'fetch', side_effect=self.fetch) as fetch, \
                        self.assertRaisesRegex(Invalid, reason):
                    ready.render_proof(BASE, root, receipts)
                self.assert_captured(first)
                self.assert_captured(later)
                self.assert_no_retries(fetch)

    def test_malformed_validated_receipt_fields_still_archive_prompt_and_tokens(self):
        for field in ('request_sha256', 'usage'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                first, later = [self.archive(root, turn) for turn in range(2)]
                receipt = self.receipt(first, 0)
                del receipt[field]
                with mock.patch.object(ready, 'fetch', side_effect=self.fetch), \
                        self.assertRaisesRegex(Invalid, f'turn-0: KeyError:.*{field}'):
                    ready.render_proof(BASE, root, [receipt])
                self.assert_captured(first)
                self.assert_captured(later)

    def test_cancellation_is_not_converted_to_a_proof(self):
        for cancellation in (KeyboardInterrupt, SystemExit):
            with self.subTest(cancellation=cancellation), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                self.archive(root, 0)
                with mock.patch.object(ready, 'fetch', side_effect=cancellation), self.assertRaises(cancellation):
                    ready.render_proof(BASE, root, [])

    def test_success_preserves_observed_hashes_without_fabricated_unvalidated_usage(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            first, later = [self.archive(root, turn) for turn in range(2)]
            with mock.patch.object(ready, 'fetch', side_effect=self.fetch):
                proof = ready.render_proof(BASE, root, [self.receipt(first, 0)])
            self.assertEqual(proof['status'], 'Observed')
            self.assertEqual([row['turn'] for row in proof['turns']], [0])
            self.assertEqual([row['turn'] for row in proof['unvalidated_turns']], [1])
            for directory, row in zip((first, later), proof['turns'] + proof['unvalidated_turns']):
                self.assert_captured(directory)
                self.assertEqual(row['request_sha256'], hashlib.sha256((directory / 'request.json').read_bytes()).hexdigest())
                self.assertEqual(row['prompt_sha256'], hashlib.sha256(self.prompt(directory).encode()).hexdigest())
                self.assertEqual(row['count'], len(self.prompt(directory).encode()))
                self.assertEqual(row['tokens_sha256'], ready.token_sha256(list(self.prompt(directory).encode())))
                for key, name in (('apply_template_sha256', 'splash-apply-template.raw'),
                                  ('tokenize_sha256', 'splash-tokenize.raw')):
                    self.assertEqual(row[key], hashlib.sha256((directory / name).read_bytes()).hexdigest())
                self.assertNotIn('usage', row)


class ClosingCaptureFailures(CaptureFixtures):
    def test_bad_first_agent_receipt_still_captures_later_turns_and_agents(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            first = self.archive(root / 'agent-0', 0, 'agent zero first')
            next_turn = self.archive(root / 'agent-0', 1, 'agent zero second')
            later_agent = self.archive(root / 'agent-1', 0, 'agent one actual request')
            bad = self.receipt(first, 0)
            bad['usage']['prompt_tokens'] += 1
            self.summary(first.parent, [bad])
            self.summary(later_agent.parent, [self.receipt(later_agent, 0)])
            with mock.patch.object(ready, 'fetch', side_effect=self.fetch) as fetch, \
                    mock.patch.object(ready, 'validate_status') as validate_status, \
                    mock.patch.object(ready, 'cross_run_state') as cross_run_state, \
                    self.assertRaisesRegex(Invalid, 'agent-0 render:.*usage.prompt_tokens'):
                ready.closing(BASE, root)
            for directory in (first, next_turn, later_agent):
                self.assert_captured(directory)
            self.assert_no_retries(fetch)
            validate_status.assert_not_called()
            cross_run_state.assert_not_called()

    def test_missing_or_malformed_summaries_remain_invalid_after_all_agents_capture(self):
        cases = (('missing', None), ('truncated', b'{"turns": ['), ('wrong-envelope', b'[]'),
                 ('wrong-turns', b'{"turns":null}'), ('bad-receipt', b'{"turns":[null]}'))
        for failure, raw in cases:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                first = self.archive(root / 'agent-0', 0, 'summaryless actual request')
                later = self.archive(root / 'agent-1', 0, 'later agent actual request')
                if raw is not None:
                    (first.parent / 'summary.json').write_bytes(raw)
                self.summary(later.parent, [self.receipt(later, 0)])
                with mock.patch.object(ready, 'fetch', side_effect=self.fetch) as fetch, \
                        self.assertRaisesRegex(Invalid, 'agent-0 (summary|render):'):
                    ready.closing(BASE, root)
                self.assert_captured(first)
                self.assert_captured(later)
                self.assert_no_retries(fetch)
                if raw is not None:
                    self.assertEqual((first.parent / 'summary.json').read_bytes(), raw)
                else:
                    self.assertFalse((first.parent / 'summary.json').exists())

    def test_multiple_agent_capture_errors_are_aggregated_after_last_agent_raw_capture(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            directories = [self.archive(root / f'agent-{index}', 0, f'agent {index} request') for index in range(3)]
            for directory in directories:
                self.summary(directory.parent, [self.receipt(directory, 0)])

            def fetch(*args):
                directory, endpoint = Path(args[2]), args[1]
                if directory == directories[0] and endpoint == '/apply-template':
                    raise OSError('first agent endpoint unreachable')
                if directory == directories[1] and endpoint == '/tokenize':
                    raise OSError('second agent endpoint unreachable')
                return self.fetch(*args)

            with mock.patch.object(ready, 'fetch', side_effect=fetch) as captured, \
                    self.assertRaises(Invalid) as error:
                ready.closing(BASE, root)
            self.assertIn('agent-0 render:', str(error.exception))
            self.assertIn('first agent endpoint unreachable', str(error.exception))
            self.assertIn('agent-1 render:', str(error.exception))
            self.assertIn('second agent endpoint unreachable', str(error.exception))
            self.assert_captured(directories[2])
            self.assertEqual((directories[1] / 'splash-apply-template.raw').read_bytes(),
                             self.render_bytes(self.prompt(directories[1])))
            self.assertFalse((directories[0] / 'splash-apply-template.raw').exists())
            self.assertFalse((directories[1] / 'splash-tokenize.raw').exists())
            self.assert_no_retries(captured)

    def test_closing_propagates_cancellation_instead_of_status_success(self):
        for cancellation in (KeyboardInterrupt, SystemExit):
            with self.subTest(cancellation=cancellation), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                first = self.archive(root / 'agent-0', 0)
                self.summary(first.parent, [self.receipt(first, 0)])
                with mock.patch.object(ready, 'fetch', side_effect=cancellation), self.assertRaises(cancellation):
                    ready.closing(BASE, root)


if __name__ == '__main__':
    unittest.main()
