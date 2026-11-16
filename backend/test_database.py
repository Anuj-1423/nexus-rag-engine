import unittest
from unittest.mock import patch

import database


class DatabaseConnectionTests(unittest.TestCase):
    @patch("database.pymysql.connect")
    def test_managed_mysql_connection_keeps_tls_enabled(self, connect):
        with patch.multiple(
            database,
            DB_HOST="mysql.example",
            DB_USER="app-user",
            DB_PASSWORD="test-password",
            DB_PORT=2345,
        ):
            connection = database.get_db_connection("nexus")

        self.assertIs(connection, connect.return_value)
        options = connect.call_args.kwargs
        self.assertEqual(options["host"], "mysql.example")
        self.assertEqual(options["user"], "app-user")
        self.assertEqual(options["database"], "nexus")
        self.assertEqual(options["port"], 2345)
        self.assertEqual(options["charset"], "utf8mb4")
        self.assertFalse(options["ssl_disabled"])

    @patch("database.pymysql.connect")
    def test_local_mysql_connection_does_not_force_tls(self, connect):
        with patch.multiple(database, DB_HOST="localhost"):
            database.get_db_connection()

        self.assertNotIn("ssl_disabled", connect.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
