"""MSP-41 Selenium: the login flow (QA-CMP, R13, NFR-003). Needs the Compose stack running
and a registered account in MSP_UI_EMAIL / MSP_UI_PASSWORD; skipped when they are missing."""
# pylint: disable=missing-function-docstring,too-few-public-methods
import unittest

from tests.system.selenium import ui_support as ui


class LoginFlowTests(ui.UiTestCase):
    """Login page: form, redirect, register link, wrong password, valid login."""

    @ui.retry_once
    def test_login_page_shows_the_form(self):
        for testid in ("login-heading", "login-email", "login-password", "login-submit"):
            self.visible(testid)
        self.assertEqual(self.text_of("login-submit"), "Sign in")

    @ui.retry_once
    def test_unauthenticated_visit_is_sent_to_login(self):  # SEC-006
        self.driver.get(ui.BASE_URL + "/submit")
        self.wait_for_path("/login")
        self.visible("login-form")

    @ui.retry_once
    def test_register_link_opens_the_register_page(self):
        self.click("login-register-link")
        self.wait_for_path("/register")

    @ui.retry_once
    def test_wrong_password_shows_one_generic_message(self):  # FR-040, SEC-003
        self.log_in(email="nobody@example.test", password="not-the-password")
        message = self.text_of("login-error")
        self.assertIn("Invalid email or password", message)
        self.assertIn("/login", self.driver.current_url)

    @unittest.skipUnless(ui.EMAIL and ui.PASSWORD, "set MSP_UI_EMAIL and MSP_UI_PASSWORD")
    @ui.retry_once
    def test_valid_login_lands_on_the_submit_page(self):
        self.log_in()
        self.wait_for_path("/submit")
        self.gone("login-form")


if __name__ == "__main__":
    unittest.main()
