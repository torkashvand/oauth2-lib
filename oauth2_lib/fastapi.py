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
from collections.abc import Awaitable, Mapping
from http import HTTPStatus
from json import JSONDecodeError
from typing import Any, Callable, Optional, Union

from fastapi import Depends, HTTPException
from fastapi.requests import Request
from fastapi.security.http import HTTPBearer
from httpx import AsyncClient, NetworkError
from pydantic import BaseModel
from starlette.requests import ClientDisconnect
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


RequestPath = str
AuthenticationFunc = Callable[[Request, Optional[str]], Awaitable[Optional[dict]]]
AuthorizationFunc = Callable[[Request, OIDCUserModel, Any], Awaitable[bool]]
GraphqlAuthorizationFunc = Callable[[RequestPath, OIDCUserModel, Optional[AsyncClient], Any], Awaitable[bool]]


async def _make_async_client() -> AsyncClient:
    return AsyncClient(http1=True, verify=HTTPX_SSL_CONTEXT)


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


class Authentication(ABC):
    @abstractmethod
    async def authenticate(self, request: Request, token: Optional[str] = None) -> Optional[dict]:
        """Authenticate the user."""
        pass


class IdTokenExtractor(ABC):
    @abstractmethod
    async def extract(self, request: Request) -> Optional[str]:
        pass


class HttpBearerExtractor(IdTokenExtractor):
    async def extract(self, request: Request) -> Optional[str]:
        http_bearer = HTTPBearer(auto_error=True)
        credential = await http_bearer(request)

        return credential.credentials if credential else None


class OIDCAuth(Authentication):
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
        id_token_extractor: Optional[IdTokenExtractor] = None,
    ):
        if not id_token_extractor:
            self.id_token_extractor = HttpBearerExtractor()

        self.openid_url = openid_url
        self.openid_config_url = openid_config_url
        self.resource_server_id = resource_server_id
        self.resource_server_secret = resource_server_secret

        self.openid_config: Optional[OIDCConfig] = None

    async def authenticate(self, request: Request, token: Optional[str] = None) -> Optional[OIDCUserModel]:
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

        async with AsyncClient(http1=True, verify=HTTPX_SSL_CONTEXT) as async_client:
            await self.check_openid_config(async_client)

            if token is None:
                extracted_id_token = await self.id_token_extractor.extract(request)
                if not extracted_id_token:
                    return None
                token_or_extracted_id_token = extracted_id_token
            elif await self.is_bypassable_request(request):
                return None
            else:
                token_or_extracted_id_token = token

            user_info = await self.userinfo(async_client, token_or_extracted_id_token)
            logger.debug("OIDCUserModel object.", user_info=user_info)
            return user_info

    @staticmethod
    async def is_bypassable_request(request: Request) -> bool:
        """By default no request is bypassable."""
        return False

    async def userinfo(self, async_request: AsyncClient, token: str) -> OIDCUserModel:
        """Get the userinfo from the openid server.

        :param AsyncClient async_request: The async request
        :param str token: the access_token
        :return: OIDCUserModel: OIDC user model from openid server

        """
        raise NotImplementedError()

    async def check_openid_config(self, async_client: AsyncClient) -> None:
        """Check of openid config is loaded and load if not."""
        if self.openid_config is not None:
            return

        response = await async_client.get(self.openid_config_url)
        if response.status_code != HTTPStatus.OK:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail=f"Could not load openid config from {self.openid_config_url}",
            )
        self.openid_config = OIDCConfig.parse_obj(response.json())


oidc_instance = OIDCAuth(
    openid_url=oauth2lib_settings.OIDC_BASE_URL,
    openid_config_url=oauth2lib_settings.OIDC_CONF_URL,  # Corrected parameter name
    resource_server_id=oauth2lib_settings.OAUTH2_RESOURCE_SERVER_ID,
    resource_server_secret=oauth2lib_settings.OAUTH2_RESOURCE_SERVER_SECRET,
)


class Authorization(ABC):
    @abstractmethod
    async def authorize(self, request: Union[Request, RequestPath], user: OIDCUserModel) -> Optional[bool]:
        pass


class OPAAbstract(Authorization, ABC):
    def __init__(self, opa_url: str, auto_error: bool = True, opa_kwargs: Union[Mapping[str, Any], None] = None):
        self.opa_url = opa_url
        self.auto_error = auto_error
        self.opa_kwargs = opa_kwargs

    async def get_decision(self, async_client: AsyncClient, opa_input: dict) -> OPAResult:
        logger.debug("Posting input json to Policy agent", opa_url=self.opa_url, input=opa_input)
        try:
            result = await async_client.post(self.opa_url, json=opa_input)
        except (NetworkError, TypeError) as exc:
            logger.debug("Could not get decision from policy agent", error=str(exc))
            raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail="Policy agent is unavailable")

        json_result = result.json()
        logger.debug("Received decision from policy agent", decision=json_result)
        return OPAResult.parse_obj(json_result)

    def evaluate_decision(self, decision: OPAResult, **context: dict[str, Any]) -> bool:
        did = decision.decision_id

        if decision.result:
            logger.debug("User is authorized to access the resource", decision_id=did, **context)
            return True

        logger.debug("User is not allowed to access the resource", decision_id=did, **context)
        if not self.auto_error:
            return False

        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail=f"User is not allowed to access resource: {context.get('resource')} Decision was taken with id: {did}",
        )


class OPAAuthorization(OPAAbstract):
    async def authorize(
        self, request: Request, user_info: OIDCUserModel = Depends(oidc_instance.authenticate)
    ) -> Optional[bool]:
        """Check OIDCUserModel against the OPA policy.

        This is used as a security module in Fastapi projects
        This method will make an async call towards the Policy agent.

        Args:
            request: Request object that will be used to retrieve request metadata.
            user_info: The OIDCUserModel object that will be checked
            async_request: The httpx client.
        """

        if not (oauth2lib_settings.OAUTH2_ACTIVE and oauth2lib_settings.OAUTH2_AUTHORIZATION_ACTIVE):
            return None

        try:
            json = await request.json()
        # Silencing the Decode error or Type error when request.json() does not return anything sane.
        # Some requests do not have a json response therefore as this code gets called on every request
        # we need to suppress the `None` case (TypeError) or the `other than json` case (JSONDecodeError)
        # Suppress AttributeError in case of websocket request, it doesn't have .json
        except (JSONDecodeError, TypeError, ClientDisconnect, AttributeError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "Stream consumed" not in str(e):
                # RuntimeError is a very broad error class. We only want to catch and ignore a stream
                # consumed runtime error. In other cases, reraise the error.
                raise e
            json = {}

        # defaulting to GET request method for WebSocket request, it doesn't have .method
        request_method = request.method if hasattr(request, "method") else "GET"
        opa_input = {
            "input": {
                **(self.opa_kwargs or {}),
                **user_info,
                "resource": request.url.path,
                "method": request_method,
                "arguments": {"path": request.path_params, "query": {**request.query_params}, "json": json},
            }
        }

        async with AsyncClient(http1=True, verify=HTTPX_SSL_CONTEXT) as async_request:
            decision = await self.get_decision(async_request, opa_input)

        context = {
            "resource": opa_input["input"]["resource"],
            "method": opa_input["input"]["method"],
            "user_info": user_info,
            "input": opa_input,
            "url": request.url,
        }
        return self.evaluate_decision(decision, **context)


class GraphQLOPAAuthorization(OPAAbstract):
    def __init__(self, opa_url: str, auto_error: bool = False, opa_kwargs: Union[Mapping[str, Any], None] = None):
        # By default don't raise HTTP 403 because partial results are preferred
        super().__init__(opa_url, auto_error, opa_kwargs)

    async def authorize(
        self, request: RequestPath, user_info: OIDCUserModel = Depends(oidc_instance.authenticate)
    ) -> Optional[bool]:
        if not (oauth2lib_settings.OAUTH2_ACTIVE and oauth2lib_settings.OAUTH2_AUTHORIZATION_ACTIVE):
            return None

        opa_input = {
            "input": {
                **(self.opa_kwargs or {}),
                **user_info,
                "resource": request,
                "method": "POST",
            }
        }

        async with AsyncClient(http1=True, verify=HTTPX_SSL_CONTEXT) as async_request:
            decision = await self.get_decision(async_request, opa_input)

        context = {"resource": opa_input["input"]["resource"], "input": opa_input}
        return self.evaluate_decision(decision, **context)
