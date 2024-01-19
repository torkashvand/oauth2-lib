# Copyright 2019-2024 SURF.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ssl
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Mapping
from http import HTTPStatus
from typing import Any, Callable, Optional, Union

from fastapi.exceptions import HTTPException
from fastapi.param_functions import Depends
from fastapi.requests import Request
from fastapi.security.http import HTTPBearer
from httpx import AsyncClient, NetworkError
from pydantic import BaseModel
from structlog import get_logger

from oauth2_lib.settings import oauth2lib_settings

logger = get_logger(__name__)

HTTPX_SSL_CONTEXT = ssl.create_default_context()  # https://github.com/encode/httpx/issues/838


class OIDCUserModel(dict):
    """The standard claims of a OIDCUserModel object. Defined per `Section 5.1`_.

    .. _`Section 5.1`: http://openid.net/specs/openid-connect-core-1_0.html#StandardClaims
    """

    #: registered claims that OIDCUserModel supports
    REGISTERED_CLAIMS = [
        "sub",
        "name",
        "given_name",
        "family_name",
        "middle_name",
        "nickname",
        "preferred_username",
        "profile",
        "picture",
        "website",
        "email",
        "email_verified",
        "gender",
        "birthdate",
        "zoneinfo",
        "locale",
        "phone_number",
        "phone_number_verified",
        "address",
        "updated_at",
    ]

    def __getattr__(self, key: str) -> Any:
        try:
            return object.__getattribute__(self, key)
        except AttributeError as error:
            if key in self.REGISTERED_CLAIMS:
                return self.get(key)
            raise error

    @property
    def user_name(self) -> str:
        return ""


async def _make_async_client() -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(http1=True, verify=HTTPX_SSL_CONTEXT) as client:
        yield client


class OIDCConfig(BaseModel):
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str
    introspect_endpoint: Optional[str] = None
    introspection_endpoint: Optional[str] = None
    jwks_uri: str
    response_types_supported: list[str]
    response_modes_supported: list[str]
    grant_types_supported: list[str]
    subject_types_supported: list[str]
    id_token_signing_alg_values_supported: list[str]
    scopes_supported: list[str]
    token_endpoint_auth_methods_supported: list[str]
    claims_supported: list[str]
    claims_parameter_supported: bool
    request_parameter_supported: bool
    code_challenge_methods_supported: list[str]


class OPAResult(BaseModel):
    result: bool = False
    decision_id: str


class Authenticator(ABC):
    @abstractmethod
    async def authenticate(self, request: Request, token: str | None = None) -> dict | None:
        """Authenticate the user."""
        pass

class IdTokenExtractor(ABC):
    @abstractmethod
    async def extract(self, request):
        pass


class HttpBearerExtractor(IdTokenExtractor):
    async def extract(self, request):
        http_bearer = HTTPBearer(auto_error=True)
        return await http_bearer(request)


class OIDCAuth(Authenticator):
    """OIDCAuth class has the HTTPBearer class to do extra verification.

    The class will act as follows:
        1. Validate the Credentials at SURFconext by calling the UserInfo endpoint
        2. When receiving an active token it will enrich the response through the database roles
    """

    def __init__(
            self,
            openid_url: str,
            openid_config_url: str,
            resource_server_id: str,
            resource_server_secret: str,
            id_token_extractor: IdTokenExtractor | None = None
    ):
        if not id_token_extractor:
            self.id_token_extractor = HttpBearerExtractor()

        self.openid_url = openid_url
        self.openid_config_url = openid_config_url
        self.resource_server_id = resource_server_id
        self.resource_server_secret = resource_server_secret

        self.openid_config = None

    async def authenticate(self, request: Request, token: str | None = None) -> OIDCUserModel | None:
        """Return the OIDC user from OIDC introspect endpoint.

        This is used as a security module in Fastapi projects

        Args:
            request: Starlette request method.
            token: Optional value to directly pass a token.

        Returns:
            OIDCUserModel object.

        """
        if not oauth2lib_settings.OAUTH2_ACTIVE:
            return None

        async with AsyncClient(http1=True, verify=HTTPX_SSL_CONTEXT) as async_request:
            await self.check_openid_config(async_request)

            if token is None:
                credentials = await self.id_token_extractor.extract(request)
                if not credentials:
                    return None
                token_or_credentials = credentials.credentials
            elif await self.should_be_skipped(request):
                return None
            else:
                token_or_credentials = token

            user_info = await self.userinfo(async_request, token_or_credentials)
            logger.debug("OIDCUserModel object.", user_info=user_info)
            return user_info

    @staticmethod
    async def should_be_skipped(request: Request) -> bool:
        return False

    async def userinfo(self, async_request: AsyncClient, token: str) -> OIDCUserModel:
        """Get the userinfo from the openid server.

        :param AsyncClient async_request: The async request
        :param str token: the access_token
        :return: OIDCUserModel: OIDC user model from openid server

        """
        raise NotImplementedError()

    async def check_openid_config(self, async_request: AsyncClient) -> None:
        """Check of openid config is loaded and load if not."""
        if self.openid_config is not None:
            return

        response = await async_request.get(self.openid_config_url)
        self.openid_config = OIDCConfig.parse_obj(response.json())


async def _get_decision(async_request: AsyncClient, opa_url: str, opa_input: dict) -> OPAResult:
    logger.debug("Posting input json to Policy agent", opa_url=opa_url, input=opa_input)
    try:
        result = await async_request.post(opa_url, json=opa_input)
    except (NetworkError, TypeError) as exc:
        logger.debug("Could not get decision from policy agent", error=str(exc))
        raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail="Policy agent is unavailable")

    return OPAResult.parse_obj(result.json())


def _evaluate_decision(decision: OPAResult, auto_error: bool, **context: dict[str, Any]) -> bool:
    did = decision.decision_id

    if decision.result:
        logger.debug("User is authorized to access the resource", decision_id=did, **context)
        return True

    logger.debug("User is not allowed to access the resource", decision_id=did, **context)
    if not auto_error:
        return False

    raise HTTPException(
        status_code=HTTPStatus.FORBIDDEN,
        detail=f"User is not allowed to access resource: {context.get('resource')} Decision was taken with id: {did}",
    )


def opa_graphql_decision(
        opa_url: str,
        _oidc_security: OIDCAuth,
        auto_error: bool = False,  # By default don't raise HTTP 403 because partial results are preferred
        opa_kwargs: Union[Mapping[str, str], None] = None,
        async_request: Union[AsyncClient, None] = None,
) -> Callable[[str, OIDCUserModel], Awaitable[Union[bool, None]]]:
    async def _opa_decision(
            path: str,
            oidc_user: OIDCUserModel = Depends(_oidc_security.authenticate),
            async_request_1: Union[AsyncClient, None] = None,
    ) -> Union[bool, None]:
        """Check OIDCUserModel against the OPA policy.

        This is used as a security module in Graphql projects
        This method will make an async call towards the Policy agent.

        Args:
            path: the graphql path that will be checked against the permissions of the oidc_user
            oidc_user: The OIDCUserModel object that will be checked
            async_request_1: The Async client
        """
        if not oauth2lib_settings.OAUTH2_ACTIVE and not oauth2lib_settings.OAUTH2_AUTHORIZATION_ACTIVE:
            return None

        opa_input = {
            "input": {
                **(opa_kwargs or {}),
                **oidc_user,
                "resource": path,
                "method": "POST",
            }
        }

        client_request = async_request or async_request_1
        if not client_request:
            client_request = AsyncClient(http1=True, verify=HTTPX_SSL_CONTEXT)

        decision = await _get_decision(client_request, opa_url, opa_input)

        context = {"resource": opa_input["input"]["resource"], "input": opa_input}
        return _evaluate_decision(decision, auto_error, **context)

    return _opa_decision
