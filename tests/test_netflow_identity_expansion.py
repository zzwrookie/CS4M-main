import unittest

from cs4m.evaluation.netflow_identity import (
    canonical_netflow_identity,
    choose_non_local_endpoint,
    expand_alerted_netflow_nodes,
)


class NetflowIdentityExpansionTest(unittest.TestCase):
    def test_choose_public_endpoint_over_private_endpoint(self):
        remote = choose_non_local_endpoint("192.168.1.5", "128.20.30.2")

        self.assertEqual(remote.remote_ip, "128.20.30.2")
        self.assertEqual(remote.remote_scope, "public")

    def test_canonical_identity_uses_remote_ip_without_port(self):
        identity = canonical_netflow_identity(
            {
                "src_ip": "192.168.1.5",
                "src_port": "43120",
                "dst_ip": "128.20.30.2",
                "dst_port": "443",
            }
        )

        self.assertEqual(identity.primary, "netflow|remote_ip|128.20.30.2")
        self.assertNotIn("43120", identity.primary)
        self.assertIn("https", identity.detail)

    def test_expand_alerted_netflow_nodes_covers_same_remote_ip_only(self):
        nodes = {
            "n1": {"src_ip": "10.0.0.5", "dst_ip": "8.8.8.8", "dst_port": "53"},
            "n2": {"src_ip": "10.0.0.5", "dst_ip": "8.8.8.8", "dst_port": "44444"},
            "n3": {"src_ip": "10.0.0.5", "dst_ip": "1.1.1.1", "dst_port": "53"},
        }
        alerted = {"netflow|remote_ip|8.8.8.8"}

        expanded = expand_alerted_netflow_nodes(nodes, alerted)

        self.assertEqual(expanded, {"n1", "n2"})


if __name__ == "__main__":
    unittest.main()
