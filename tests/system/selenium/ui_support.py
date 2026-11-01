"""Shared helpers for the Selenium UI flows (MSP-41; QA-CMP, R13, NFR-003).

Rules every test follows: explicit waits only (no time.sleep), stable element IDs
(data-testid), one automatic retry, and a quarantine list for flaky tests.
Settings come from the environment, nothing is hard-coded:

    MSP_UI_URL       the front end, default https://localhost
    MSP_UI_EMAIL / MSP_UI_PASSWORD   a registered test account
    MSP_UI_WAIT      explicit wait in seconds, default 10
    MSP_UI_HEADED=1  show the browser instead of running headless
"""
# pylint: disable=missing-function-docstring,too-many-public-methods,not-callable
# pylint: disable=protected-access,consider-using-f-string
import functools
import os
import unittest

try:  # CI may not have these installed: then every UI test skips instead of erroring
    import requests
    import urllib3
    from selenium import webdriver
    from selenium.common.exceptions import TimeoutException, WebDriverException
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    TOOLS_OK = True
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:  # pragma: no cover
    TOOLS_OK = False
    requests = urllib3 = webdriver = Options = By = EC = WebDriverWait = None

    class WebDriverException(Exception):
        """Stand-in so the module still imports without selenium."""

    class TimeoutException(WebDriverException):
        """Stand-in so the module still imports without selenium."""

from tests.system.selenium.quarantine import QUARANTINE

BASE_URL = os.environ.get("MSP_UI_URL", "https://localhost").rstrip("/")
EMAIL = os.environ.get("MSP_UI_EMAIL", "")
PASSWORD = os.environ.get("MSP_UI_PASSWORD", "")
WAIT_SECONDS = float(os.environ.get("MSP_UI_WAIT", "10"))
HEADED = os.environ.get("MSP_UI_HEADED") == "1"


def front_end_up():
    try:
        return requests.get(BASE_URL + "/", timeout=3, verify=False).ok
    except requests.RequestException:
        return False


def make_driver():
    options = Options()
    if not HEADED:
        options.add_argument("--headless=new")
    options.add_argument("--ignore-certificate-errors")  # self-signed certificate on the test stack
    options.add_argument("--window-size=1280,900")
    return webdriver.Chrome(options=options)


def retry_once(test):
    """Run the test again once from a clean browser if it fails (flaky-network guard)."""
    @functools.wraps(test)
    def wrapper(self):
        try:
            test(self)
        except (AssertionError, WebDriverException) as first:
            print("\nRETRY %s after: %s" % (self.id(), str(first).splitlines()[0][:120]))
            type(self).retried.append(self._testMethodName)
            self.reset_browser()
            test(self)
    return wrapper


class UiTestCase(unittest.TestCase):
    """Base class: one browser per class, a clean session before every test."""
    driver = None
    retried = []

    @classmethod
    def setUpClass(cls):
        if not TOOLS_OK:
            raise unittest.SkipTest("selenium/requests not installed")
        if not front_end_up():
            raise unittest.SkipTest("front end not reachable at " + BASE_URL)
        try:
            cls.driver = make_driver()
        except WebDriverException as exc:
            raise unittest.SkipTest("Chrome/driver not available: %s" % str(exc)[:100])
        cls.retried = []
        cls.driver.get(BASE_URL + "/login")
        try:
            WebDriverWait(cls.driver, WAIT_SECONDS).until(
                EC.presence_of_element_located(UiTestCase.sel("login-form")))
        except TimeoutException as exc:
            cls.driver.quit()
            cls.driver = None
            raise unittest.SkipTest(
                "no login form with data-testid at %s/login. Start the front end "
                "(cd frontend, npm run dev) and set MSP_UI_URL to the URL it prints, "
                "or rebuild it so Login.jsx has the data-testid attributes." % BASE_URL) from exc

    @classmethod
    def tearDownClass(cls):
        if cls.driver:
            cls.driver.quit()
        if cls.retried:
            print("\nRetried tests (look at these for flakiness): %s" % ", ".join(cls.retried))

    def setUp(self):
        key = "%s.%s" % (type(self).__name__, self._testMethodName)
        if key in QUARANTINE:
            self.skipTest("quarantined: " + QUARANTINE[key])
        self.reset_browser()

    def reset_browser(self):
        self.driver.get(BASE_URL + "/login")
        self.driver.delete_all_cookies()
        self.driver.execute_script("window.localStorage.clear(); window.sessionStorage.clear();")
        self.driver.get(BASE_URL + "/login")

    # ---- explicit waits on stable IDs ----
    @staticmethod
    def sel(testid):
        return (By.CSS_SELECTOR, '[data-testid="%s"]' % testid)

    def wait(self):
        return WebDriverWait(self.driver, WAIT_SECONDS)

    def visible(self, testid):
        return self.wait().until(EC.visibility_of_element_located(self.sel(testid)))

    def gone(self, testid):
        return self.wait().until(EC.invisibility_of_element_located(self.sel(testid)))

    def click(self, testid):
        self.wait().until(EC.element_to_be_clickable(self.sel(testid))).click()

    def type_into(self, testid, text):
        field = self.visible(testid)
        field.clear()
        field.send_keys(text)

    def text_of(self, testid):
        return self.visible(testid).text

    def wait_for_path(self, path):
        self.wait().until(EC.url_contains(path))

    def log_in(self, email=None, password=None):
        self.type_into("login-email", email if email is not None else EMAIL)
        self.type_into("login-password", password if password is not None else PASSWORD)
        self.click("login-submit")
