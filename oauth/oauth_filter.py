import flask
import requests
from oauth.access_control import AccessControl, UserAttributes
from werkzeug.exceptions import RequestTimeout, Unauthorized

from nwastdlib.cache import cached_result


class OAuthFilter(object):
    """
    OAuthFilter class object.

    OAuthFilter checks the bearer access_token in the Authorization header using the check_token endpoint exposed by the AuthorizationServer.
    See the integration tests in test_oauth_filter.py for examples. The check_token payload is saved in the thread-local flask.g for subsequent use in the API endpoints.
    """

    def __init__(
        self,
        security_definitions,
        token_check_url,
        resource_server_id,
        resource_server_secret,
        white_listed_urls=None,
        allow_localhost_calls=True,
    ):
        self.access_rules = AccessControl(security_definitions)
        self.token_check_url = token_check_url
        self.white_listed_urls = [] if not white_listed_urls else white_listed_urls
        self.auth = (resource_server_id, resource_server_secret)
        self.allow_localhost_calls = allow_localhost_calls

    def filter(self):
        current_request = flask.request
        # Allow Cross-Origin Resource Sharing calls and local health checks
        if current_request.method == "OPTIONS" or (
            self.allow_localhost_calls and current_request.base_url.startswith("http://localhost")
        ):
            return

        endpoint = current_request.endpoint if current_request.endpoint else current_request.base_url

        is_white_listed = next(filter(lambda url: endpoint.endswith(url), self.white_listed_urls), None)
        if is_white_listed:
            return

        authorization = current_request.headers.get("Authorization")
        if not authorization:
            raise Unauthorized(description="No Authorization token provided")
        else:
            try:
                _, token = authorization.split()
            except ValueError:
                raise Unauthorized(description="Invalid authorization header: {}".format(authorization))

            token_info = self.check_token(token)

            current_user = UserAttributes(token_info)

            if current_user.active:
                self.access_rules.is_allowed(current_user, current_request)
            else:
                raise Unauthorized(description="Provided oauth token is not active: {}".format(token))

            flask.g.current_user = current_user

    @cached_result(expiry=30)
    def check_token(self, token):
        try:
            with requests.Session() as s:
                s.auth = self.auth
                token_request = s.get(self.token_check_url, params={"token": token}, timeout=5)
        except requests.exceptions.Timeout:
            raise RequestTimeout(description="RequestTimeout from authorization server")

        if not token_request.ok:
            raise Unauthorized(description="Provided oauth token is not valid: {}".format(token))
        return token_request.json()

    @classmethod
    def current_user(cls):
        return flask.g.get("current_user", None) if flask.has_app_context() else None
