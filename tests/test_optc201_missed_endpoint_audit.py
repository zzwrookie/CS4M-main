import unittest

from scripts.tools.audit_optc201_missed_endpoint import (
    normalize_ip_token,
    parse_netflow_detail,
    row_matches_endpoint,
    service_bucket_for_port,
)


class Optc201MissedEndpointAuditTests(unittest.TestCase):
    def test_normalize_ip_token_accepts_underscore_form(self) -> None:
        self.assertEqual(normalize_ip_token("132_197_158_98"), "132.197.158.98")

    def test_parse_netflow_detail_extracts_bidirectional_endpoint(self) -> None:
        parsed = parse_netflow_detail(
            "{'netflow': '132.197.158.98:80->142.20.56.202:51231'}",
        )

        self.assertEqual(parsed, ("132.197.158.98", "80", "142.20.56.202", "51231"))

    def test_row_matches_http_endpoint_without_matching_port_8000(self) -> None:
        self.assertTrue(
            row_matches_endpoint(
                "{'netflow': '142.20.56.202:50994->132.197.158.98:80'}",
                "132_197_158_98",
                "http",
            ),
        )
        self.assertFalse(
            row_matches_endpoint(
                "{'netflow': '142.20.56.202:49181->132.197.158.98:8000'}",
                "132_197_158_98",
                "http",
            ),
        )

    def test_service_bucket_for_port_keeps_http_distinct(self) -> None:
        self.assertEqual(service_bucket_for_port("80"), "http")
        self.assertEqual(service_bucket_for_port("8000"), "registered")


if __name__ == "__main__":
    unittest.main()
