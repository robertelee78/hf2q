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

    def test_exact_laptop_junk_payload_is_not_relevant_to_gold(self):
        document = """
        <ol>
          <li class="b_algo"><h2><a href="https://price.com/">Price.com: Save with Cash Back, Coupons &amp; Price Comparison</a></h2>
            <div class="b_caption"><p>Save money with offers for more than 100,000 brands.</p></div></li>
          <li class="b_algo"><h2><a href="https://priceindustries.com/">Home - Price Industries</a></h2>
            <div class="b_caption"><p>A market leader in supplying air distribution products.</p></div></li>
          <li class="b_algo"><h2><a href="https://www.priceline.com/">Priceline.com - Hotels, Flights and Rental Cars</a></h2>
            <div class="b_caption"><p>Members get our best travel price.</p></div></li>
        </ol>
        """
        candidates = search_fallback.parse_bing_results(document)

        self.assertEqual(len(candidates), 3)
        self.assertEqual(
            search_fallback.filter_relevant_results(
                "what is the price of gold today", candidates, 3
            ),
            [],
        )

    def test_relevance_uses_token_boundaries(self):
        result = {
            "title": "Goldman Sachs market outlook",
            "url": "https://example.com/goldman-outlook",
            "content": "Current equity prices and forecasts",
        }
        self.assertFalse(
            search_fallback.result_looks_relevant(
                "what is the price of gold today", result
            )
        )

    def test_relevance_accepts_common_singular_plural_variants(self):
        plural_result = {
            "title": "Gold prices today",
            "url": "https://example.com/metals",
            "content": "Latest bullion quote",
        }
        singular_result = {
            "title": "Gold price today",
            "url": "https://example.com/metals",
            "content": "Latest bullion quote",
        }
        self.assertTrue(
            search_fallback.result_looks_relevant(
                "what is the price of gold today", plural_result
            )
        )
        self.assertTrue(
            search_fallback.result_looks_relevant(
                "what are the prices of gold today", singular_result
            )
        )

    def test_two_of_three_identifying_terms_meet_intended_threshold(self):
        result = {
            "title": "Rust runtime design",
            "url": "https://example.com/runtime",
            "content": "A systems guide",
        }
        self.assertTrue(
            search_fallback.result_looks_relevant("rust async runtime", result)
        )

    def test_query_without_identifying_terms_fails_closed(self):
        self.assertFalse(
            search_fallback.result_looks_relevant(
                "who", {"title": "Anything", "url": "https://example.com/"}
            )
        )

    def test_focused_query_preserves_question_and_anchors_identifying_terms(self):
        query = "what is the price of gold today"
        self.assertEqual(
            search_fallback.focused_query(query),
            'what is the price of gold today "price" "gold"',
        )

    def test_parses_brave_organic_results_with_provenance(self):
        document = """
        <div class="snippet" data-type="web">
          <a href="https://www.kitco.com/charts/gold"><span class="search-snippet-title">Gold Price Today</span></a>
          <div class="generic-snippet">Live gold price per ounce.</div>
        </div>
        <div class="snippet" data-type="web">
          <a href="https://search.brave.com/help"><span class="search-snippet-title">Internal</span></a>
        </div>
        """
        results = search_fallback.parse_brave_results(document)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["engine"], "brave-search-fallback")
        self.assertEqual(results[0]["url"], "https://www.kitco.com/charts/gold")

    def test_parses_bing_rss_and_rejects_external_entities(self):
        document = """<?xml version="1.0"?>
        <rss><channel><item>
          <title>Gold Price Today</title>
          <link>https://www.apmex.com/gold-price</link>
          <description>Live gold price per ounce.</description>
        </item></channel></rss>"""
        results = search_fallback.parse_bing_rss_results(document)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["engine"], "bing-rss-fallback")
        self.assertEqual(results[0]["url"], "https://www.apmex.com/gold-price")

        entity_document = """<?xml version="1.0"?>
        <!DOCTYPE rss [<!ENTITY secret SYSTEM "file:///etc/passwd">]>
        <rss><channel><item><title>&secret;</title>
        <link>https://example.com/</link></item></channel></rss>"""
        self.assertEqual(search_fallback.parse_bing_rss_results(entity_document), [])

    def test_all_provider_url_builders_validate_language(self):
        self.assertIn("format=rss", search_fallback.build_bing_rss_search_url("gold", "en-US"))
        self.assertIn("lang=en-US", search_fallback.build_brave_search_url("gold", "en-US"))
        for builder in (
            search_fallback.build_bing_rss_search_url,
            search_fallback.build_brave_search_url,
        ):
            with self.assertRaises(ValueError):
                builder("gold", "en-US&redirect=http://localhost")


if __name__ == "__main__":
    unittest.main(verbosity=2)
