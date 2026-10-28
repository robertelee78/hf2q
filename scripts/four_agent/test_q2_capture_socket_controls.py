"""Bounded actual loopback bind lifecycle; never launches a client or model."""
import errno
import socket
import threading
import time
import unittest
from q2_capture_client import Handler, Stub


class CaptureSocketControls(unittest.TestCase):
    def test_closed_http_listener_rebinds_but_live_owner_refuses(self):
        started = time.monotonic()
        self.assertTrue(Stub.allow_reuse_address)
        self.assertFalse(Stub.allow_reuse_port)
        server = Stub(("127.0.0.1", 0), Handler)
        address = server.server_address
        server.timeout = .25
        worker = threading.Thread(target=server.handle_request)
        worker.start()
        try:
            with socket.create_connection(address, timeout=.25) as client:
                client.settimeout(.25)
                client.sendall(b"GET / HTTP/1.0\r\nHost: localhost\r\n\r\n")
                response = bytearray()
                while True:
                    part = client.recv(4096)
                    if not part:
                        break
                    response.extend(part)
                    self.assertLessEqual(len(response), 16384)
                self.assertIn(b" 405 ", response.split(b"\r\n", 1)[0])
            worker.join(timeout=.25)
            self.assertFalse(worker.is_alive())
        finally:
            server.server_close()
            worker.join(timeout=.25)
        # Same production Stub, same exact port after real accepted HTTP EOF.
        with Stub(address, Handler) as rebound:
            self.assertEqual(rebound.server_address, address)
            with self.assertRaises(OSError) as occupied:
                Stub(address, Handler)
            self.assertEqual(occupied.exception.errno, errno.EADDRINUSE)
        self.assertLess(time.monotonic() - started, 1.0)


if __name__ == "__main__":
    unittest.main()
