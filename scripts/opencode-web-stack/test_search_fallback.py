import base64
import unittest

import search_fallback


def wrapped(url: str) -> str:
    encoded = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    return f"https://www.bing.com/ck/a?u=a1{encoded}"


class SearchFallbackTests(unittest.TestCase):
    def test_parses_direct_and_wrapped_organic_results_with_provenance(self):
        document = f"""
        <ol>
          <li class="b_algo"><h2><a href="https://example.com/a">Alpha</a></h2>
            <div class="b_caption"><p>First result</p></div></li>
          <li class="b_algo"><h2><a href="{wrapped('https://example.org/b?q=1')}">Beta</a></h2>
            <p>Second result</p></li>
        </ol>
        """
        results = search_fallback.parse_bing_results(document)
        self.assertEqual([result["title"] for result in results], ["Alpha", "Beta"])
        self.assertEqual(results[1]["url"], "https://example.org/b?q=1")
        self.assertEqual(results[0]["engines"], ["bing-browser-fallback"])

    def test_rejects_malformed_internal_private_and_duplicate_targets(self):
        document = """
        <li class="b_algo"><h2><a href="https://www.bing.com/ck/a?u=a1%%%">Bad</a></h2></li>
        <li class="b_algo"><h2><a href="http://user:pass@example.com/">Credentials</a></h2></li>
        <li class="b_algo"><h2><a href="http://localhost/admin">Local</a></h2></li>
        <li class="b_algo"><h2><a href="https://example.com/ok">Good</a></h2></li>
        <li class="b_algo"><h2><a href="https://example.com/ok">Duplicate</a></h2></li>
        """
        results = search_fallback.parse_bing_results(document)
        self.assertEqual([(result["title"], result["url"]) for result in results], [("Good", "https://example.com/ok")])

    def test_http_200_challenge_page_is_blocked_not_success(self):
        self.assertTrue(search_fallback.page_is_blocked("<title>CAPTCHA</title>Verify you are a human"))
        self.assertFalse(search_fallback.page_is_blocked("<title>Search results</title>"))

    def test_language_is_strict(self):
        self.assertIn("setlang=en-US", search_fallback.build_bing_search_url("hello", "en-US"))
        with self.assertRaises(ValueError):
            search_fallback.build_bing_search_url("hello", "../../etc/passwd")


if __name__ == "__main__":
    unittest.main(verbosity=2)
