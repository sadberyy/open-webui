import asyncio
import hashlib
import json
import logging
import re
from typing import Optional
from urllib.parse import urlparse

import aiohttp
from aiocache import cached
import requests

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from fastapi import Depends, HTTPException, Request, APIRouter
from fastapi.responses import (
    FileResponse,
    StreamingResponse,
    JSONResponse,
    PlainTextResponse,
)
from pydantic import BaseModel, ConfigDict

from sqlalchemy.orm import Session

from open_webui.internal.db import get_session

from open_webui.models.models import Models
from open_webui.models.access_grants import AccessGrants
from open_webui.models.groups import Groups
from open_webui.config import (
    CACHE_DIR,
    OPENAI_API_BASE_URL,#-----------------------
    OPENAI_API_KEY,
)

from open_webui.retrieval.utils import get_content_from_url
from fastapi.concurrency import run_in_threadpool
from open_webui.routers.retrieval import process_web
#---------------------------------------------------
from open_webui.env import (
    MODELS_CACHE_TTL,
    AIOHTTP_CLIENT_SESSION_SSL,
    AIOHTTP_CLIENT_TIMEOUT,
    AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST,
    ENABLE_FORWARD_USER_INFO_HEADERS,
    FORWARD_SESSION_INFO_HEADER_CHAT_ID,
    BYPASS_MODEL_ACCESS_CONTROL,
)
from open_webui.models.users import UserModel

from open_webui.constants import ERROR_MESSAGES


from open_webui.utils.payload import (
    apply_model_params_to_body_openai,
    apply_system_prompt_to_body,
)
from open_webui.utils.misc import (
    cleanup_response,
    convert_logit_bias_input_to_json,
    stream_chunks_handler,
    stream_wrapper,
)

from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.headers import include_user_info_headers
from open_webui.utils.anthropic import is_anthropic_url, get_anthropic_models

from open_webui.models.memories import Memories
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
from openai import AsyncOpenAI

log = logging.getLogger(__name__)


##########################################
#
# Utility functions
# Let the responses returned through this gate be worth
# the question that summoned them.
#
##########################################


async def send_get_request(
    request: Request = None,
    url=None,
    key=None,
    user: UserModel = None,
    config=None,
):
    timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST)
    try:
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            if request and config:
                headers, cookies = await get_headers_and_cookies(request, url, key, config, user=user)
            else:
                headers = {
                    **({'Authorization': f'Bearer {key}'} if key else {}),
                }
                cookies = None

                if ENABLE_FORWARD_USER_INFO_HEADERS and user:
                    headers = include_user_info_headers(headers, user)

            async with session.get(
                url,
                headers=headers,
                cookies=cookies,
                ssl=AIOHTTP_CLIENT_SESSION_SSL,
            ) as response:
                return await response.json()
    except Exception as e:
        # Handle connection error here
        log.error(f'Connection error: {e}')
        return None


async def get_models_request(
    request: Request = None,
    url=None,
    key=None,
    user: UserModel = None,
    config=None,
):
    if is_anthropic_url(url):
        return await get_anthropic_models(url, key, user=user)
    return await send_get_request(request, f'{url}/models', key, user=user, config=config)


def openai_reasoning_model_handler(payload):
    """
    Handle reasoning model specific parameters
    """
    if 'max_tokens' in payload:
        # Convert "max_tokens" to "max_completion_tokens" for all reasoning models
        payload['max_completion_tokens'] = payload['max_tokens']
        del payload['max_tokens']

    # Handle system role conversion based on model type
    if payload['messages'][0]['role'] == 'system':
        model_lower = payload['model'].lower()
        # Legacy models use "user" role instead of "system"
        if model_lower.startswith('o1-mini') or model_lower.startswith('o1-preview'):
            payload['messages'][0]['role'] = 'user'
        else:
            payload['messages'][0]['role'] = 'developer'

    return payload


async def get_headers_and_cookies(
    request: Request,
    url,
    key=None,
    config=None,
    metadata: Optional[dict] = None,
    user: UserModel = None,
):
    cookies = {}
    headers = {
        'Content-Type': 'application/json',
        **(
            {
                'HTTP-Referer': 'https://openwebui.com/',
                'X-Title': 'Open WebUI',
            }
            if 'openrouter.ai' in url
            else {}
        ),
    }

    if ENABLE_FORWARD_USER_INFO_HEADERS and user:
        headers = include_user_info_headers(headers, user)
        if metadata and metadata.get('chat_id'):
            headers[FORWARD_SESSION_INFO_HEADER_CHAT_ID] = metadata.get('chat_id')

    token = None
    auth_type = config.get('auth_type')

    if auth_type == 'bearer' or auth_type is None:
        # Default to bearer if not specified
        token = f'{key}'
    elif auth_type == 'none':
        token = None
    elif auth_type == 'session':
        cookies = request.cookies
        token = request.state.token.credentials
    elif auth_type == 'system_oauth':
        cookies = request.cookies

        oauth_token = None
        try:
            if request.cookies.get('oauth_session_id', None):
                oauth_token = await request.app.state.oauth_manager.get_oauth_token(
                    user.id,
                    request.cookies.get('oauth_session_id', None),
                )
        except Exception as e:
            log.error(f'Error getting OAuth token: {e}')

        if oauth_token:
            token = f'{oauth_token.get("access_token", "")}'

    elif auth_type in ('azure_ad', 'microsoft_entra_id'):
        token = get_microsoft_entra_id_access_token()

    if token:
        headers['Authorization'] = f'Bearer {token}'

    if config.get('headers') and isinstance(config.get('headers'), dict):
        headers = {**headers, **config.get('headers')}

    return headers, cookies


def get_microsoft_entra_id_access_token():
    """
    Get Microsoft Entra ID access token using DefaultAzureCredential for Azure OpenAI.
    Returns the token string or None if authentication fails.
    """
    try:
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), 'https://cognitiveservices.azure.com/.default'
        )
        return token_provider()
    except Exception as e:
        log.error(f'Error getting Microsoft Entra ID access token: {e}')
        return None


##########################################
#
# API routes
#
##########################################

router = APIRouter()


@router.get('/config')
async def get_config(request: Request, user=Depends(get_admin_user)):
    return {
        'ENABLE_OPENAI_API': request.app.state.config.ENABLE_OPENAI_API,
        'OPENAI_API_BASE_URLS': request.app.state.config.OPENAI_API_BASE_URLS,
        'OPENAI_API_KEYS': request.app.state.config.OPENAI_API_KEYS,
        'OPENAI_API_CONFIGS': request.app.state.config.OPENAI_API_CONFIGS,
    }


class OpenAIConfigForm(BaseModel):
    ENABLE_OPENAI_API: Optional[bool] = None
    OPENAI_API_BASE_URLS: list[str]
    OPENAI_API_KEYS: list[str]
    OPENAI_API_CONFIGS: dict


@router.post('/config/update')
async def update_config(request: Request, form_data: OpenAIConfigForm, user=Depends(get_admin_user)):
    request.app.state.config.ENABLE_OPENAI_API = form_data.ENABLE_OPENAI_API
    request.app.state.config.OPENAI_API_BASE_URLS = form_data.OPENAI_API_BASE_URLS
    request.app.state.config.OPENAI_API_KEYS = form_data.OPENAI_API_KEYS

    # Check if API KEYS length is same than API URLS length
    if len(request.app.state.config.OPENAI_API_KEYS) != len(request.app.state.config.OPENAI_API_BASE_URLS):
        if len(request.app.state.config.OPENAI_API_KEYS) > len(request.app.state.config.OPENAI_API_BASE_URLS):
            request.app.state.config.OPENAI_API_KEYS = request.app.state.config.OPENAI_API_KEYS[
                : len(request.app.state.config.OPENAI_API_BASE_URLS)
            ]
        else:
            request.app.state.config.OPENAI_API_KEYS += [''] * (
                len(request.app.state.config.OPENAI_API_BASE_URLS) - len(request.app.state.config.OPENAI_API_KEYS)
            )

    request.app.state.config.OPENAI_API_CONFIGS = form_data.OPENAI_API_CONFIGS

    # Remove the API configs that are not in the API URLS
    keys = list(map(str, range(len(request.app.state.config.OPENAI_API_BASE_URLS))))
    request.app.state.config.OPENAI_API_CONFIGS = {
        key: value for key, value in request.app.state.config.OPENAI_API_CONFIGS.items() if key in keys
    }

    return {
        'ENABLE_OPENAI_API': request.app.state.config.ENABLE_OPENAI_API,
        'OPENAI_API_BASE_URLS': request.app.state.config.OPENAI_API_BASE_URLS,
        'OPENAI_API_KEYS': request.app.state.config.OPENAI_API_KEYS,
        'OPENAI_API_CONFIGS': request.app.state.config.OPENAI_API_CONFIGS,
    }


@router.post('/audio/speech')
async def speech(request: Request, user=Depends(get_verified_user)):
    idx = None
    try:
        idx = request.app.state.config.OPENAI_API_BASE_URLS.index('https://api.openai.com/v1')

        body = await request.body()
        name = hashlib.sha256(body).hexdigest()

        SPEECH_CACHE_DIR = CACHE_DIR / 'audio' / 'speech'
        SPEECH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        file_path = SPEECH_CACHE_DIR.joinpath(f'{name}.mp3')
        file_body_path = SPEECH_CACHE_DIR.joinpath(f'{name}.json')

        # Check if the file already exists in the cache
        if file_path.is_file():
            return FileResponse(file_path)

        url = request.app.state.config.OPENAI_API_BASE_URLS[idx]
        key = request.app.state.config.OPENAI_API_KEYS[idx]
        api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
            str(idx),
            request.app.state.config.OPENAI_API_CONFIGS.get(url, {}),  # Legacy support
        )

        headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)

        r = None
        try:
            r = requests.post(
                url=f'{url}/audio/speech',
                data=body,
                headers=headers,
                cookies=cookies,
                stream=True,
            )

            r.raise_for_status()

            # Save the streaming content to a file
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            with open(file_body_path, 'w') as f:
                json.dump(json.loads(body.decode('utf-8')), f)

            # Return the saved file
            return FileResponse(file_path)

        except Exception as e:
            log.exception(e)

            detail = None
            if r is not None:
                try:
                    res = r.json()
                    if 'error' in res:
                        detail = f'External: {res["error"]}'
                except Exception:
                    detail = f'External: {e}'

            raise HTTPException(
                status_code=r.status_code if r else 500,
                detail=detail if detail else 'Open WebUI: Server Connection Error',
            )

    except ValueError:
        raise HTTPException(status_code=401, detail=ERROR_MESSAGES.OPENAI_NOT_FOUND)


async def get_all_models_responses(request: Request, user: UserModel) -> list:
    if not request.app.state.config.ENABLE_OPENAI_API:
        return []

    # Cache config values locally to avoid repeated Redis lookups.
    # Each access to request.app.state.config.<KEY> triggers a Redis GET;
    # caching here avoids hundreds of redundant round-trips.
    api_base_urls = request.app.state.config.OPENAI_API_BASE_URLS
    api_keys = list(request.app.state.config.OPENAI_API_KEYS)
    api_configs = request.app.state.config.OPENAI_API_CONFIGS

    # Check if API KEYS length is same than API URLS length
    num_urls = len(api_base_urls)
    num_keys = len(api_keys)

    if num_keys != num_urls:
        # if there are more keys than urls, remove the extra keys
        if num_keys > num_urls:
            api_keys = api_keys[:num_urls]
            request.app.state.config.OPENAI_API_KEYS = api_keys
        # if there are more urls than keys, add empty keys
        else:
            api_keys += [''] * (num_urls - num_keys)
            request.app.state.config.OPENAI_API_KEYS = api_keys

    request_tasks = []
    for idx, url in enumerate(api_base_urls):
        if (str(idx) not in api_configs) and (url not in api_configs):  # Legacy support
            request_tasks.append(get_models_request(request, url, api_keys[idx], user=user))
        else:
            api_config = api_configs.get(
                str(idx),
                api_configs.get(url, {}),  # Legacy support
            )

            enable = api_config.get('enable', True)
            model_ids = api_config.get('model_ids', [])

            if enable:
                if len(model_ids) == 0:
                    request_tasks.append(get_models_request(request, url, api_keys[idx], user=user, config=api_config))
                else:
                    model_list = {
                        'object': 'list',
                        'data': [
                            {
                                'id': model_id,
                                'name': model_id,
                                'owned_by': 'openai',
                                'openai': {'id': model_id},
                                'urlIdx': idx,
                            }
                            for model_id in model_ids
                        ],
                    }

                    request_tasks.append(asyncio.ensure_future(asyncio.sleep(0, model_list)))
            else:
                request_tasks.append(asyncio.ensure_future(asyncio.sleep(0, None)))

    responses = await asyncio.gather(*request_tasks)

    for idx, response in enumerate(responses):
        if response:
            url = api_base_urls[idx]
            api_config = api_configs.get(
                str(idx),
                api_configs.get(url, {}),  # Legacy support
            )

            connection_type = api_config.get('connection_type', 'external')
            prefix_id = api_config.get('prefix_id', None)
            tags = api_config.get('tags', [])

            model_list = response if isinstance(response, list) else response.get('data', [])
            if not isinstance(model_list, list):
                # Catch non-list responses
                model_list = []

            for model in model_list:
                # Remove name key if its value is None #16689
                if 'name' in model and model['name'] is None:
                    del model['name']

                if prefix_id:
                    model['id'] = f'{prefix_id}.{model.get("id", model.get("name", ""))}'

                if tags:
                    model['tags'] = tags

                if connection_type:
                    model['connection_type'] = connection_type

    log.debug(f'get_all_models:responses() {responses}')
    return responses


async def get_filtered_models(models, user, db=None):
    # Filter models based on user access control
    model_ids = [model['id'] for model in models.get('data', [])]
    model_infos = {model_info.id: model_info for model_info in Models.get_models_by_ids(model_ids, db=db)}
    user_group_ids = {group.id for group in Groups.get_groups_by_member_id(user.id, db=db)}

    # Batch-fetch accessible resource IDs in a single query instead of N has_access calls
    accessible_model_ids = AccessGrants.get_accessible_resource_ids(
        user_id=user.id,
        resource_type='model',
        resource_ids=list(model_infos.keys()),
        permission='read',
        user_group_ids=user_group_ids,
        db=db,
    )

    filtered_models = []
    for model in models.get('data', []):
        model_info = model_infos.get(model['id'])
        if model_info:
            if user.id == model_info.user_id or model_info.id in accessible_model_ids:
                filtered_models.append(model)
    return filtered_models


@cached(
    ttl=MODELS_CACHE_TTL,
    key=lambda _, user: f'openai_all_models_{user.id}' if user else 'openai_all_models',
)
async def get_all_models(request: Request, user: UserModel) -> dict[str, list]:
    log.info('get_all_models()')

    if not request.app.state.config.ENABLE_OPENAI_API:
        return {'data': []}

    # Cache config value locally to avoid repeated Redis lookups inside
    # the nested loop in get_merged_models (one GET per model otherwise).
    api_base_urls = request.app.state.config.OPENAI_API_BASE_URLS

    responses = await get_all_models_responses(request, user=user)

    def extract_data(response):
        if response and 'data' in response:
            return response['data']
        if isinstance(response, list):
            return response
        return None

    def is_supported_openai_models(model_id):
        if any(
            name in model_id
            for name in [
                'babbage',
                'dall-e',
                'davinci',
                'embedding',
                'tts',
                'whisper',
            ]
        ):
            return False
        return True

    def get_merged_models(model_lists):
        log.debug(f'merge_models_lists {model_lists}')
        models = {}

        for idx, model_list in enumerate(model_lists):
            if model_list is not None and 'error' not in model_list:
                for model in model_list:
                    model_id = model.get('id') or model.get('name')

                    base_url = api_base_urls[idx]
                    hostname = urlparse(base_url).hostname if base_url else None
                    if hostname == 'api.openai.com' and not is_supported_openai_models(model_id):
                        # Skip unwanted OpenAI models
                        continue

                    if model_id and model_id not in models:
                        models[model_id] = {
                            **model,
                            'name': model.get('name', model_id),
                            'owned_by': 'openai',
                            'openai': model,
                            'connection_type': model.get('connection_type', 'external'),
                            'urlIdx': idx,
                        }

        return models

    models = get_merged_models(map(extract_data, responses))
    log.debug(f'models: {models}')

    request.app.state.OPENAI_MODELS = models
    return {'data': list(models.values())}


@router.get('/models')
@router.get('/models/{url_idx}')
async def get_models(request: Request, url_idx: Optional[int] = None, user=Depends(get_verified_user)):
    if not request.app.state.config.ENABLE_OPENAI_API:
        raise HTTPException(status_code=503, detail='OpenAI API is disabled')

    models = {
        'data': [],
    }

    if url_idx is None:
        models = await get_all_models(request, user=user)
    else:
        url = request.app.state.config.OPENAI_API_BASE_URLS[url_idx]
        key = request.app.state.config.OPENAI_API_KEYS[url_idx]

        api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
            str(url_idx),
            request.app.state.config.OPENAI_API_CONFIGS.get(url, {}),  # Legacy support
        )

        r = None
        async with aiohttp.ClientSession(
            trust_env=True,
            timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST),
        ) as session:
            try:
                headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)

                if api_config.get('azure', False):
                    models = {
                        'data': api_config.get('model_ids', []) or [],
                        'object': 'list',
                    }
                elif is_anthropic_url(url):
                    models = await get_anthropic_models(url, key, user=user)
                    if models is None:
                        raise Exception('Failed to connect to Anthropic API')
                else:
                    async with session.get(
                        f'{url}/models',
                        headers=headers,
                        cookies=cookies,
                        ssl=AIOHTTP_CLIENT_SESSION_SSL,
                    ) as r:
                        if r.status != 200:
                            error_detail = f'HTTP Error: {r.status}'
                            try:
                                res = await r.json()
                                if 'error' in res:
                                    error_detail = f'External Error: {res["error"]}'
                            except Exception:
                                pass
                            raise Exception(error_detail)

                        response_data = await r.json()

                        if 'api.openai.com' in url:
                            response_data['data'] = [
                                model
                                for model in response_data.get('data', [])
                                if not any(
                                    name in model['id']
                                    for name in [
                                        'babbage',
                                        'dall-e',
                                        'davinci',
                                        'embedding',
                                        'tts',
                                        'whisper',
                                    ]
                                )
                            ]

                        models = response_data
            except aiohttp.ClientError as e:
                # ClientError covers all aiohttp requests issues
                log.exception(f'Client error: {str(e)}')
                raise HTTPException(status_code=500, detail='Open WebUI: Server Connection Error')
            except Exception as e:
                log.exception(f'Unexpected error: {e}')
                error_detail = f'Unexpected error: {str(e)}'
                raise HTTPException(status_code=500, detail=error_detail)

    if user.role == 'user' and not BYPASS_MODEL_ACCESS_CONTROL:
        models['data'] = await get_filtered_models(models, user)

    return models


class ConnectionVerificationForm(BaseModel):
    url: str
    key: str

    config: Optional[dict] = None


@router.post('/verify')
async def verify_connection(
    request: Request,
    form_data: ConnectionVerificationForm,
    user=Depends(get_admin_user),
):
    url = form_data.url
    key = form_data.key

    api_config = form_data.config or {}

    async with aiohttp.ClientSession(
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST),
    ) as session:
        try:
            headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)

            if api_config.get('azure', False):
                # Only set api-key header if not using Azure Entra ID authentication
                auth_type = api_config.get('auth_type', 'bearer')
                if auth_type not in ('azure_ad', 'microsoft_entra_id'):
                    headers['api-key'] = key

                api_version = api_config.get('api_version', '') or '2023-03-15-preview'
                async with session.get(
                    url=f'{url}/openai/models?api-version={api_version}',
                    headers=headers,
                    cookies=cookies,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                ) as r:
                    try:
                        response_data = await r.json()
                    except Exception:
                        response_data = await r.text()

                    if r.status != 200:
                        if isinstance(response_data, (dict, list)):
                            return JSONResponse(status_code=r.status, content=response_data)
                        else:
                            return PlainTextResponse(status_code=r.status, content=response_data)

                    return response_data
            elif is_anthropic_url(url):
                result = await get_anthropic_models(url, key)
                if result is None:
                    raise HTTPException(status_code=500, detail='Failed to connect to Anthropic API')
                if 'error' in result:
                    raise HTTPException(status_code=500, detail=result['error'])
                return result
            else:
                async with session.get(
                    f'{url}/models',
                    headers=headers,
                    cookies=cookies,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                ) as r:
                    try:
                        response_data = await r.json()
                    except Exception:
                        response_data = await r.text()

                    if r.status != 200:
                        if isinstance(response_data, (dict, list)):
                            return JSONResponse(status_code=r.status, content=response_data)
                        else:
                            return PlainTextResponse(status_code=r.status, content=response_data)

                    return response_data

        except aiohttp.ClientError as e:
            # ClientError covers all aiohttp requests issues
            log.exception(f'Client error: {str(e)}')
            raise HTTPException(status_code=500, detail='Open WebUI: Server Connection Error')
        except Exception as e:
            log.exception(f'Unexpected error: {e}')
            raise HTTPException(status_code=500, detail='Open WebUI: Server Connection Error')


def get_azure_allowed_params(api_version: str) -> set[str]:
    allowed_params = {
        'messages',
        'temperature',
        'role',
        'content',
        'contentPart',
        'contentPartImage',
        'enhancements',
        'dataSources',
        'n',
        'stream',
        'stop',
        'max_tokens',
        'presence_penalty',
        'frequency_penalty',
        'logit_bias',
        'user',
        'function_call',
        'functions',
        'tools',
        'tool_choice',
        'top_p',
        'log_probs',
        'top_logprobs',
        'response_format',
        'seed',
        'max_completion_tokens',
        'reasoning_effort',
    }

    try:
        if api_version >= '2024-09-01-preview':
            allowed_params.add('stream_options')
    except ValueError:
        log.debug(f'Invalid API version {api_version} for Azure OpenAI. Defaulting to allowed parameters.')

    return allowed_params


def is_openai_new_model(model: str) -> bool:
    model_lower = model.lower()
    # o-series models (o1, o3, o4, o5, ...)
    if re.match(r'^o\d+', model_lower):
        return True
    # gpt-N where N >= 5 (gpt-5, gpt-5.2, gpt-6, ...)
    m = re.match(r'^gpt-(\d+)', model_lower)
    if m and int(m.group(1)) >= 5:
        return True
    return False


def convert_to_azure_payload(url, payload: dict, api_version: str):
    model = payload.get('model', '')

    # Filter allowed parameters based on Azure OpenAI API
    allowed_params = get_azure_allowed_params(api_version)

    # Special handling for o-series models
    if is_openai_new_model(model):
        # Convert max_tokens to max_completion_tokens for o-series models
        if 'max_tokens' in payload:
            payload['max_completion_tokens'] = payload['max_tokens']
            del payload['max_tokens']

        # Remove temperature if not 1 for o-series models
        if 'temperature' in payload and payload['temperature'] != 1:
            log.debug(
                f'Removing temperature parameter for o-series model {model} as only default value (1) is supported'
            )
            del payload['temperature']

    # Filter out unsupported parameters
    payload = {k: v for k, v in payload.items() if k in allowed_params}

    url = f'{url}/openai/deployments/{model}'
    return url, payload


# Fields accepted by the Responses API for each input item type.
RESPONSES_ALLOWED_FIELDS: dict[str, set[str]] = {
    'message': {'type', 'role', 'content'},
    'function_call': {'type', 'call_id', 'name', 'arguments', 'id'},
    'function_call_output': {'type', 'call_id', 'output'},
}


def _normalize_stored_item(item: dict) -> dict:
    """Strip local-only fields from a stored output item before replaying it.

    Open WebUI stores extra bookkeeping fields (``id``, ``status``,
    ``started_at``, ``ended_at``, ``duration``, ``_tag_type``,
    ``attributes``, ``summary``, etc.) that the Responses API does
    not accept.  This helper returns a copy containing only the
    fields the API understands.
    """
    item_type = item.get('type', '')
    allowed = RESPONSES_ALLOWED_FIELDS.get(item_type)
    if allowed is None:
        # Unknown type — pass through as-is (e.g. reasoning, extension items).
        return item
    return {k: v for k, v in item.items() if k in allowed}


def convert_to_responses_payload(payload: dict) -> dict:
    """
    Convert Chat Completions payload to Responses API format.

    Chat Completions: { messages: [{role, content}], ... }
    Responses API: { input: [{type: "message", role, content: [...]}], instructions: "system" }
    """
    messages = payload.pop('messages', [])

    system_content = ''
    input_items = []

    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')

        # Check for stored output items (from previous Responses API turn)
        stored_output = msg.get('output')
        if stored_output and isinstance(stored_output, list):
            input_items.extend(_normalize_stored_item(item) for item in stored_output)
            continue

        if role == 'system':
            if isinstance(content, str):
                system_content = content
            elif isinstance(content, list):
                system_content = '\n'.join(p.get('text', '') for p in content if p.get('type') == 'text')
            continue

        # Handle assistant messages with tool_calls (from convert_output_to_messages)
        if role == 'assistant' and msg.get('tool_calls'):
            # Add text content as message if present
            if content:
                text = (
                    content
                    if isinstance(content, str)
                    else '\n'.join(p.get('text', '') for p in content if p.get('type') == 'text')
                )
                if text.strip():
                    input_items.append(
                        {
                            'type': 'message',
                            'role': 'assistant',
                            'content': [{'type': 'output_text', 'text': text}],
                        }
                    )
            # Convert each tool_call to a function_call input item
            for tool_call in msg['tool_calls']:
                func = tool_call.get('function', {})
                input_items.append(
                    {
                        'type': 'function_call',
                        'call_id': tool_call.get('id', ''),
                        'name': func.get('name', ''),
                        'arguments': func.get('arguments', '{}'),
                    }
                )
            continue

        # Handle tool result messages
        if role == 'tool':
            input_items.append(
                {
                    'type': 'function_call_output',
                    'call_id': msg.get('tool_call_id', ''),
                    'output': msg.get('content', ''),
                }
            )
            continue

        # Convert content format
        text_type = 'output_text' if role == 'assistant' else 'input_text'

        if isinstance(content, str):
            content_parts = [{'type': text_type, 'text': content}]
        elif isinstance(content, list):
            content_parts = []
            for part in content:
                if part.get('type') == 'text':
                    content_parts.append({'type': text_type, 'text': part.get('text', '')})
                elif part.get('type') == 'image_url':
                    url_data = part.get('image_url', {})
                    url = url_data.get('url', '') if isinstance(url_data, dict) else url_data
                    content_parts.append({'type': 'input_image', 'image_url': url})
        else:
            content_parts = [{'type': text_type, 'text': str(content)}]

        input_items.append({'type': 'message', 'role': role, 'content': content_parts})

    responses_payload = {**payload, 'input': input_items}

    # Forward previous_response_id when the middleware has set it
    # (only used when ENABLE_RESPONSES_API_STATEFUL is enabled).
    previous_response_id = responses_payload.pop('previous_response_id', None)
    if previous_response_id:
        responses_payload['previous_response_id'] = previous_response_id

    if system_content:
        responses_payload['instructions'] = system_content

    if 'max_tokens' in responses_payload:
        responses_payload['max_output_tokens'] = responses_payload.pop('max_tokens')

    if 'max_completion_tokens' in responses_payload:
        responses_payload['max_output_tokens'] = responses_payload.pop('max_completion_tokens')

    # Remove Chat Completions-only parameters not supported by the Responses API
    for unsupported_key in (
        'stream_options',
        'logit_bias',
        'frequency_penalty',
        'presence_penalty',
        'stop',
    ):
        responses_payload.pop(unsupported_key, None)

    # Convert Chat Completions tools format to Responses API format
    # Chat Completions: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    # Responses API:    {"type": "function", "name": ..., "description": ..., "parameters": ...}
    if 'tools' in responses_payload and isinstance(responses_payload['tools'], list):
        converted_tools = []
        for tool in responses_payload['tools']:
            if isinstance(tool, dict) and 'function' in tool:
                func = tool['function']
                converted_tool = {'type': tool.get('type', 'function')}
                if isinstance(func, dict):
                    converted_tool['name'] = func.get('name', '')
                    if 'description' in func:
                        converted_tool['description'] = func['description']
                    if 'parameters' in func:
                        converted_tool['parameters'] = func['parameters']
                    if 'strict' in func:
                        converted_tool['strict'] = func['strict']
                converted_tools.append(converted_tool)
            else:
                # Already in correct format or unknown format, pass through
                converted_tools.append(tool)
        responses_payload['tools'] = converted_tools

    return responses_payload


def convert_responses_result(response: dict) -> dict:
    """
    Convert non-streaming Responses API result to Chat Completions format.

    Extracts text from message output items so all downstream consumers
    (frontend tasks, get_content_from_response) work without modification.
    """
    output_items = response.get('output', [])

    content = ''
    for item in output_items:
        if item.get('type') == 'message':
            for part in item.get('content', []):
                if part.get('type') == 'output_text':
                    content += part.get('text', '')

    return {
        'id': response.get('id', ''),
        'object': 'chat.completion',
        'model': response.get('model', ''),
        'choices': [
            {
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': content,
                },
                'finish_reason': 'stop',
            }
        ],
        'usage': response.get('usage', {}),
    }


@router.post('/chat/completions')
async def generate_chat_completion(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user),
    bypass_system_prompt: bool = False,
):
    # NOTE: We intentionally do NOT use Depends(get_session) here.
    # Database operations (get_model_by_id, AccessGrants.has_access) manage their own short-lived sessions.
    # This prevents holding a connection during the entire LLM call (30-60+ seconds),
    # which would exhaust the connection pool under concurrent load.

    # bypass_filter is read from request.state to prevent external clients from
    # setting it via query parameter (CVE fix). Only internal server-side callers
    # (e.g. utils/chat.py) should set request.state.bypass_filter = True.
    bypass_filter = getattr(request.state, 'bypass_filter', False)
    if BYPASS_MODEL_ACCESS_CONTROL:
        bypass_filter = True

    idx = 0

    payload = {**form_data}
    metadata = payload.pop('metadata', None)

    #--------------------------------------------------------------------------------
    # Флаг авто-режима
    auto_mode_enabled = form_data.get('auto_model', False)
    is_manual_selection = False
    
    # Проверка на вручную выбранную модель
    user_selected_model = form_data.get('model')
    if auto_mode_enabled:
        is_manual_selection = False
        print(f"[ROUTER] Auto model enabled")
    elif user_selected_model and user_selected_model != "auto":
        is_manual_selection = True
        print(f"[ROUTER] Manual model selection detected: {user_selected_model}")

    selected = None

    # Модель, классифицирующая запрос на "нужен ли поиск в интернете или не нужен"
    need_search = False
    last_query = ""
    
    messages = payload.get('messages', [])
    user_messages = [m for m in messages if m.get('role') == 'user']
    
    if user_messages:
        last_query = user_messages[-1].get('content', '')
        if isinstance(last_query, list):
            text_parts = [part.get('text', '') for part in last_query if isinstance(part, dict) and part.get('type') == 'text']
            last_query = ' '.join(text_parts) if text_parts else ''

        # Режим исследования
        deep_research_mode = form_data.get('deep_research', False)
        
        if deep_research_mode and last_query:
            print(f"[DEEP RESEARCH] Mode activated for: {last_query[:100]}...")
            
            # Итеративный поиск
            all_results = []
            search_queries = [last_query, f"{last_query} подробный анализ", f"{last_query} последние новости", f"{last_query} paper"]
            
            for sq in search_queries:
                try:
                    from open_webui.routers.retrieval import process_web_search
                    
                    class SearchForm:
                        def __init__(self, queries):
                            self.queries = queries
                    
                    result = await process_web_search(
                        request=request,
                        form_data=SearchForm([sq]),
                        user=user
                    )
                    
                    if result and result.get('docs'):
                        all_results.extend(result.get('docs', []))
                        print(f"[DEEP RESEARCH] Found {len(result.get('docs', []))} results for: {sq[:50]}...")
                except Exception as e:
                    print(f"[DEEP RESEARCH] Search error: {e}")
            
            # Формируем отчёт
            if all_results:
                unique_sources = {}
                for doc in all_results[:5]:  # топ-5
                    meta = doc.get('metadata', {})
                    source = meta.get('source', meta.get('link', 'unknown'))
                    if source not in unique_sources:
                        unique_sources[source] = doc.get('content', '')[:2000]
                
                instruction = f"""Ты — аналитический ассистент. Используй ТОЛЬКО информацию из предоставленных ниже источников.
ОБЯЗАТЕЛЬНО указывай источники в ответе в формате [1], [2] и т.д.
Если информация не найдена в источниках, скажи об этом честно.
Запрос пользователя: {last_query}
Источники для анализа:
"""
                sources_for_response = []
                for idx, (source, content) in enumerate(unique_sources.items(), 1):
                    instruction += f"\n---\n## Источник {idx}: {source}\n\n{content}\n"
                    sources_for_response.append(f"[{idx}] {source}")
                
                instruction += f"""
ВАЖНО: Отвечая, ссылайся на источники! Пример: "Согласно источнику [1], ..."
После ответа добавь раздел "Использованные источники:" со списком:
{chr(10).join(sources_for_response)}
Если источники не содержат ответа на запрос, скажи: "Информация не найдена в предоставленных источниках"."""
                
                messages = payload.get('messages', [])
                # Удаляем старые system сообщения (кроме memory)
                filtered_messages = [m for m in messages if m.get('role') != 'system' or 'память' in m.get('content', '').lower()]
                filtered_messages.insert(0, {"role": "system", "content": instruction})
                payload['messages'] = filtered_messages
                print(f"[DEEP RESEARCH] Instruction added with {len(unique_sources)} sources")
            else:
                print(f"[DEEP RESEARCH] No results found")

        if last_query and isinstance(last_query, str) and len(last_query.strip()) > 0:
            search_decision_client = AsyncOpenAI(
                base_url=OPENAI_API_BASE_URL,
                api_key=OPENAI_API_KEY,
                timeout=5.0
            )
            
            decision_prompt = f"""Ты — классификатор запросов. Ответь ТОЛЬКО "yes" или "no".
Нужно ли искать в интернете актуальную информацию для ответа на этот запрос?

Отвечай "yes" если:
- Вопрос о текущей дате, времени, новостях, событиях
- Вопрос требует актуальных данных (курсы, погода, цены, пробки)
- Пользователь явно просит найти информацию ("найди", "поищи", "расскажи о", "узнай" и т.д.)

Отвечай "no" если:
- Вопрос о фактах, не требующих актуальности (столица Франции — Париж)
- Вопрос о коде, программировании
- Общие знания, математика, логика
- Перевод текста
- Просьба написать письмо, стих, рассказ

Запрос: "{last_query[:500]}". Ответ (yes/no):"""
            
            try:
                decision_response = await search_decision_client.chat.completions.create(
                    model="mws-gpt-alpha",
                    messages=[{"role": "user", "content": decision_prompt}],
                    max_tokens=5,
                    temperature=0.0
                )
                need_search = decision_response.choices[0].message.content.strip().lower() == "yes"
                print(f"[SEARCH] Need search: {need_search} for: {last_query[:50]}...")
            except Exception as e:
                print(f"[SEARCH] Decision error: {e}, defaulting to False")
                need_search = False
    
    # Поиск в Интернете
    if request.app.state.config.ENABLE_WEB_SEARCH and need_search:
        print(f"[SEARCH] Executing web search for: {last_query[:100]}...")
        
        try:
            from open_webui.routers.retrieval import process_web_search
            
            class SearchForm:
                def __init__(self, queries):
                    self.queries = queries
            
            search_result = await process_web_search(
                request=request,
                form_data=SearchForm([last_query]),
                user=user
            )
            
            # Просто добавляем результаты в сообщение
            if search_result and search_result.get('docs'):
                search_context = "РЕЗУЛЬТАТЫ ПОИСКА В ИНТЕРНЕТЕ:\n\n"
                for doc in search_result.get('docs', []):
                    content = doc.get('content', '')
                    meta = doc.get('metadata', {})
                    source = meta.get('source', meta.get('link', 'unknown'))
                    if content:
                        search_context += f"Источник: {source}\n{content}\n\n---\n\n"
                
                messages = payload.get('messages', [])
                messages.insert(0, {
                    "role": "system",
                    "content": search_context[:6000]
                })
                payload['messages'] = messages
                print(f"[SEARCH] Added context (length: {len(search_context)})")
            elif search_result and search_result.get('collection_names'):
                print(f"[SEARCH] No docs, using collections: {search_result.get('collection_names')}")
                
        except Exception as e:
            print(f"[SEARCH] Error: {e}")

    # Веб-парсинг без сохранения в векторную бд
    messages = payload.get('messages', [])
    user_messages = [m for m in messages if m.get('role') == 'user']
    has_url = False

    if user_messages:
        last_msg = user_messages[-1].get('content', '')
        if isinstance(last_msg, str):
            urls = re.findall(r'https?://[^\s]+', last_msg)

            if urls:
                print(f"[WEB] Found URLs in message: {urls}")

                for url in urls:
                    try:
                        content, docs = await run_in_threadpool(get_content_from_url, request, url)
                        
                        if content and len(content) > 100:
                            # Добавляем содержимое в системное сообщение
                            content_preview = content[:5000]
                            system_message = {
                                "role": "system",
                                "content": f"Содержимое веб-страницы {url}:\n\n{content_preview}"
                            }
                            messages.insert(0, system_message)
                            payload['messages'] = messages
                            print(f"[WEB] Direct parse success: {url} (length: {len(content)})")
                            has_url = True
                            break
                        else:
                            print(f"[WEB] Not enough content from: {url}")
                            
                    except Exception as e:
                        print(f"[WEB] Direct parse error for {url}: {e}")
    
    VISION_MODELS = ["qwen2.5-vl-72b", "cotype-pro-vl-32b", "qwen2.5-vl"]
    IMAGE_GEN_MODEL = "qwen-image-lightning" 
    DEFAULT_MODEL = "mws-gpt-alpha"
    CODE_MODEL = "qwen3-coder-480b-a35b"
    REASONING_MODEL = "deepseek-r1-distill-qwen-32b"
    LONG_CONTEXT_MODEL = "llama-3.3-70b-instruct"
    
    # Долгосрочная память
    if request.app.state.config.ENABLE_MEMORIES and user:
        try:
            messages = payload.get('messages', [])
            user_messages = [m for m in messages if m.get('role') == 'user']
            
            if user_messages:
                last_query = user_messages[-1].get('content', '')
                if isinstance(last_query, list):
                    text_parts = [part.get('text', '') for part in last_query if isinstance(part, dict) and part.get('type') == 'text']
                    last_query = ' '.join(text_parts) if text_parts else ''
                
                if last_query and isinstance(last_query, str) and len(last_query.strip()) > 0:
                    memories = Memories.get_memories_by_user_id(user.id)
                    
                    if memories:
                        query_vector = await request.app.state.EMBEDDING_FUNCTION(last_query, user=user)
                        
                        results = VECTOR_DB_CLIENT.search(
                            collection_name=f'user-memory-{user.id}',
                            vectors=[query_vector],
                            limit=3,
                        )
                        
                        if results and results.ids and results.ids[0]:
                            memory_context = "Важная информация о пользователе (из долгосрочной памяти):\n"
                            for idx, doc_id in enumerate(results.ids[0]):
                                if idx < len(results.documents[0]):
                                    text = results.documents[0][idx]
                                    if text:
                                        memory_context += f"- {text}\n"
                            
                            if messages and len(messages) > 0:
                                system_messages = [m for m in messages if m.get('role') == 'system']
                                if system_messages:
                                    system_messages[0]['content'] = memory_context + "\n\n" + system_messages[0].get('content', '')
                                else:
                                    messages.insert(0, {
                                        "role": "system",
                                        "content": memory_context
                                    })
                                payload['messages'] = messages
                                print(f"[MEMORY] Added {len(results.ids[0])} relevant memories to context")
        
        except Exception as e:
            print(f"[MEMORY] Error: {e}")
    
    # LLM сама решает, какую информацию нужно запомнить
    if request.app.state.config.ENABLE_MEMORIES and user:
        try:
            
            messages = payload.get('messages', [])
            user_messages = [m for m in messages if m.get('role') == 'user']
            assistant_messages = [m for m in messages if m.get('role') == 'assistant']
            
            if user_messages:
                last_user_msg = user_messages[-1].get('content', '')
                last_assistant_msg = assistant_messages[-1].get('content', '') if assistant_messages else ''
                
                if isinstance(last_user_msg, list):
                    text_parts = [part.get('text', '') for part in last_user_msg if isinstance(part, dict) and part.get('type') == 'text']
                    last_user_msg = ' '.join(text_parts)
                if isinstance(last_assistant_msg, list):
                    text_parts = [part.get('text', '') for part in last_assistant_msg if isinstance(part, dict) and part.get('type') == 'text']
                    last_assistant_msg = ' '.join(text_parts)
                
                if last_user_msg and isinstance(last_user_msg, str):
                    memory_client = AsyncOpenAI(
                        base_url=OPENAI_API_BASE_URL,
                        api_key=OPENAI_API_KEY,
                        timeout=10.0
                    )
                    
                    analysis_prompt = f"""Ты — система управления долгосрочной памятью. Проанализируй диалог и реши, нужно ли сохранить информацию о пользователе в память.
Правила сохранения:
1. Сохраняй информацию о профессии, образовании, навыках
2. Сохраняй информацию о предпочтениях, интересах, хобби
3. Сохраняй информацию о личных фактах (семейное положение, место жительства и т.д.)
4. Сохраняй информацию о здоровье, диете, аллергиях
5. НЕ сохраняй: одноразовые просьбы, вопросы, приветствия, временные запросы
6. НЕ сохраняй: информацию о текущем диалоге, которая не относится к пользователю

Пользователь сказал: "{last_user_msg[:500]}"
Ассистент ответил: "{last_assistant_msg[:500]}" if last_assistant_msg else ""
Если нужно сохранить информацию, ответь в формате:
SAVE: [информация для сохранения]
Если ничего не нужно сохранять, ответь: NONE
Важно: формулируй информацию от третьего лица, используя "User" (например: "User is a veterinarian", "User likes Python").
"""
                    
                    try:
                        analysis_response = await memory_client.chat.completions.create(
                            model=DEFAULT_MODEL,
                            messages=[{"role": "user", "content": analysis_prompt}],
                            max_tokens=200,
                            temperature=0.3
                        )
                        
                        result = analysis_response.choices[0].message.content.strip()
                        
                        if result.startswith("SAVE:"):
                            fact = result.replace("SAVE:", "").strip()
                            if fact and len(fact) > 10 and fact != "NONE":
                                existing_memories = Memories.get_memories_by_user_id(user.id)
                                existing_texts = [m.content.lower() for m in existing_memories] if existing_memories else []
                                
                                if fact.lower() not in existing_texts:
                                    # Сохраняем новое воспоминание
                                    memory = Memories.insert_new_memory(user.id, fact)
                                    print(f"[MEMORY] LLM decided to save: {fact}")
                                    
                                    # Сохраняем эмбеддинг
                                    vector = await request.app.state.EMBEDDING_FUNCTION(fact, user=user)
                                    VECTOR_DB_CLIENT.upsert(
                                        collection_name=f'user-memory-{user.id}',
                                        items=[{
                                            'id': memory.id,
                                            'text': fact,
                                            'vector': vector,
                                            'metadata': {'created_at': memory.created_at},
                                        }],
                                    )
                    except Exception as e:
                        print(f"[MEMORY] Analysis error: {e}")
                        
        except Exception as e:
            print(f"[MEMORY] Auto-save error: {e}")
    
    # Удаляем информацию о предыдущей модели из запроса. Это заставит систему заново определить модель для каждого сообщения
    if 'model' in form_data:
        user_selected_model = form_data.get('model')
        if not metadata or not metadata.get('manual_model_selection'):
            old_model = form_data.get('model')
            print(f"[ROUTER] Clearing cached model: {old_model} → will re-route")

    async def smart_route(user_msg: str, has_img: bool, has_file: bool, metadata=None, conversation_history: list = None) -> str:
        # Защита от нестроковых значений
        if not isinstance(user_msg, str):
            user_msg = str(user_msg) if user_msg else ""

        # Если есть изображение - vision модель
        if has_img:
            selected_vision_model = VISION_MODELS[0]
            print(f"[ROUTER] Image detected → {selected_vision_model}")
            return selected_vision_model
        
        # Если есть файл (не изображение) - long context
        if has_file:
            print(f"[ROUTER] File detected → {LONG_CONTEXT_MODEL}")
            return LONG_CONTEXT_MODEL
        
        # Определяем, нужно ли генерировать изображение
        client = AsyncOpenAI(
            base_url=OPENAI_API_BASE_URL,
            api_key=OPENAI_API_KEY,
            timeout=5.0
        )
        
        intent_prompt = f"""Определи, что хочет пользователь. Ответь ТОЛЬКО одним словом: "image" если запрос просит создать/нарисовать/сгенерировать картинку, или "text" если это обычный вопрос или просьба написать текст.
Запрос: {user_msg[:200]}
Ответ (image или text):"""
        
        try:
            intent_response = await client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": intent_prompt}],
                max_tokens=10,
                temperature=0.0
            )
            intent = intent_response.choices[0].message.content.strip().lower()
            
            if intent == "image":
                print(f"[ROUTER] Image generation detected → {IMAGE_GEN_MODEL}")
                return IMAGE_GEN_MODEL
        except Exception as e:
            print(f"[ROUTER] Intent detection failed: {e}")
        
        # Короткие запросы
        if len(user_msg.split()) < 5:
            print(f"[ROUTER] Short query → {DEFAULT_MODEL}")
            return DEFAULT_MODEL
        
        if conversation_history:
            # Смотрим последние 5 сообщений
            for msg in conversation_history[-5:]:
                if msg.get('role') == 'user':
                    content = msg.get('content', '')
                    if isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get('type') == 'text':
                                text_parts.append(part.get('text', ''))
                        content = ' '.join(text_parts) if text_parts else ''
                    elif not isinstance(content, str):
                        content = str(content)
                    content = content.lower()
        
        # Формируем контекст для LLM-роутера
        context_info = ""
        if conversation_history and len(conversation_history) > 0:
            last_messages = conversation_history[-3:]  # Последние 3 сообщения
            context_info = "\n\nПоследние сообщения в диалоге:\n"
            for msg in last_messages:
                role = "Пользователь" if msg.get('role') == 'user' else "Ассистент"
                content = msg.get('content', '')
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            text_parts.append(part.get('text', ''))
                    content = ' '.join(text_parts) if text_parts else ''
                elif not isinstance(content, str):
                    content = str(content)
                context_info += f"{role}: {content[:100]}\n"
        
        routing_prompt = f"""Ты — роутер запросов. Выбери модель из списка, учитывая контекст диалога.
- {CODE_MODEL} — для ВСЕХ вопросов, связанных с программированием: написание кода, отладка, объяснение алгоритмов, решение задач. Если в диалоге обсуждался код, продолжай использовать эту модель.
- {REASONING_MODEL} — для сложных логических рассуждений, философских вопросов, глубокого анализа.
- {DEFAULT_MODEL} — для общих вопросов, фактов, простых диалогов.
{context_info}
Текущий запрос пользователя: "{user_msg[:200]}". Ответь ТОЛЬКО названием модели:"""
        
        try:
            resp = await client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": routing_prompt}],
                max_tokens=20,
                temperature=0.0
            )
            chosen = resp.choices[0].message.content.strip().lower()
            
            if CODE_MODEL in chosen or "qwen3-coder" in chosen or "code" in chosen:
                return CODE_MODEL
            if REASONING_MODEL in chosen or "deepseek" in chosen:
                return REASONING_MODEL
            return DEFAULT_MODEL
        except Exception as e:
            print(f"[ROUTER] LLM routing failed: {e}")
            return DEFAULT_MODEL
    
    # Проверка на наличие изображений (несколько способов)
    messages = payload.get('messages', [])
    user_msgs = [m for m in messages if m.get('role') == 'user']
    
    has_image = False
    has_file = False
    if has_url:
        has_file = True
        print(f"[ROUTER] URL detected, setting has_file=True")

    # Берём ТОЛЬКО последнее сообщение пользователя
    if user_msgs:
        last_msg = user_msgs[-1]
        last_content = last_msg.get('content', [])
        
        # Проверяем, есть ли изображение в последнем сообщении
        if isinstance(last_content, list):
            for part in last_content:
                if isinstance(part, dict) and part.get('type') == 'image_url':
                    has_image = True
                    print(f"[ROUTER] Found image_url in LAST message")
                    break
        
        # Проверяем метаданные о файлах (только для текущего сообщения)
        if metadata and not has_image:
            files = metadata.get('files', [])
            if files and isinstance(files, list):
                for file in files:
                    if file and isinstance(file, dict):
                        file_type = file.get('type', '') or file.get('mime_type', '')
                        if file_type.startswith('image/'):
                            has_image = True
                            print(f"[ROUTER] Found image file in current message: {file.get('filename', 'unknown')}")
                        elif file_type:
                            has_file = True
    
    # Вызов роутера
    if not is_manual_selection and user_msgs:
        last_msg_text = user_msgs[-1].get('content', '')
        
        if isinstance(last_msg_text, list):
            text_parts = []
            for part in last_msg_text:
                if isinstance(part, dict):
                    if part.get('type') == 'text':
                        text_parts.append(part.get('text', ''))
                elif isinstance(part, str):
                    text_parts.append(part)
                else:
                    text_parts.append(str(part))
            last_msg_text = ' '.join(text_parts) if text_parts else ''
        elif not isinstance(last_msg_text, str):
            last_msg_text = str(last_msg_text)
        
        if not last_msg_text.strip() and has_image:
            last_msg_text = "Проанализируй изображение"
        
        if last_msg_text.strip():
            # Передаём историю диалога в роутер
            conversation_history = messages
            selected = await smart_route(last_msg_text, has_image, has_file, metadata, conversation_history)
            form_data['model'] = selected
            payload['model'] = selected
            print(f"[ROUTER] Final decision: {selected} (image={has_image}, file={has_file})")
        elif has_image:
            # Сообщение только с картинкой, без текста
            selected = await smart_route("Проанализируй изображение", has_image, has_file, metadata, messages)
            form_data['model'] = selected
            payload['model'] = selected
            print(f"[ROUTER] Final decision (image only): {selected}")
    elif is_manual_selection:
        print(f"[ROUTER] Using manually selected model: {user_selected_model}")
        payload['model'] = user_selected_model

    # Генерация изображений
    if payload.get('model') == IMAGE_GEN_MODEL:
        # Берём последнее сообщение пользователя
        messages = payload.get('messages', [])
        prompt = ""
        
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                if isinstance(content, str):
                    prompt = content
                elif isinstance(content, list):
                    for part in content:
                        if part.get('type') == 'text':
                            prompt = part.get('text', '')
                            break
                break
        
        if not prompt:
            prompt = form_data.get('prompt', '')
        
        url = request.app.state.config.OPENAI_API_BASE_URLS[0]
        key = request.app.state.config.OPENAI_API_KEYS[0]
        
        image_payload = {
            "model": IMAGE_GEN_MODEL,
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024"
        }
        
        print(f"[IMAGE GEN] Generating image for prompt: {prompt}")  # Лог для отладки
        
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession(trust_env=True) as session:
            try:
                async with session.post(
                    f'{url}/images/generations',
                    headers=headers,
                    json=image_payload,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                ) as r:
                    response_data = await r.json()
                    
                    if r.status >= 400:
                        print(f"[IMAGE GEN] Error: {response_data}")
                        return JSONResponse(status_code=r.status, content=response_data)
                    
                    image_url = response_data.get('data', [{}])[0].get('url', '')
                    print(f"[IMAGE GEN] Success! Image URL: {image_url}")
                    
                    formatted_response = {
                        "id": "image-gen-response",
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": f"![generated]({image_url})\n\n[Смотреть изображение]({image_url})"
                                },
                                "finish_reason": "stop"
                            }
                        ]
                    }
                    return formatted_response
                    
            except Exception as e:
                log.exception(e)
                raise HTTPException(
                    status_code=500,
                    detail=f"Image generation failed: {str(e)}",
                )

    # Убеждаемся, что выбранная роутером модель не будет перезаписана
    if selected is not None and selected != form_data.get('model'):
        print(f"[ROUTER] WARNING: Model changed from {form_data.get('model')} to {selected}, forcing override")
        form_data['model'] = selected
        payload['model'] = selected
    
    # Очистка истории от изображений
    current_model = payload.get('model', '')
    
    if current_model not in VISION_MODELS:
        messages = payload.get('messages', [])
        cleaned_messages = []
        
        for msg in messages:
            content = msg.get('content', '')
            if isinstance(content, list):
                # Извлекаем только текст из сообщения
                text_parts = [part.get('text', '') for part in content if part.get('type') == 'text']
                if text_parts:
                    msg['content'] = ' '.join(text_parts)
                    cleaned_messages.append(msg)
                else:
                    msg['content'] = "[Пользователь отправил изображение]"
                    cleaned_messages.append(msg)
            else:
                cleaned_messages.append(msg)
        
        payload['messages'] = cleaned_messages
        print(f"[ROUTER] Converted image messages to text for model: {current_model}")
 
    #--------------------------------------------------------------------------------

    model_id = form_data.get('model') 
    model_info = Models.get_model_by_id(model_id)

    # Check model info and override the payload
    if model_info:
        if model_info.base_model_id:
            base_model_id = (
                request.base_model_id if hasattr(request, 'base_model_id') else model_info.base_model_id
            )  # Use request's base_model_id if available
            payload['model'] = base_model_id
            model_id = base_model_id

        params = model_info.params.model_dump()

        if params:
            system = params.pop('system', None)

            payload = apply_model_params_to_body_openai(params, payload)
            if not bypass_system_prompt:
                payload = apply_system_prompt_to_body(system, payload, metadata, user)

        # Check if user has access to the model
        if not bypass_filter and user.role == 'user':
            user_group_ids = {group.id for group in Groups.get_groups_by_member_id(user.id)}
            if not (
                user.id == model_info.user_id
                or AccessGrants.has_access(
                    user_id=user.id,
                    resource_type='model',
                    resource_id=model_info.id,
                    permission='read',
                    user_group_ids=user_group_ids,
                )
            ):
                raise HTTPException(
                    status_code=403,
                    detail='Model not found',
                )
    elif not bypass_filter:
        if user.role != 'admin':
            raise HTTPException(
                status_code=403,
                detail='Model not found',
            )

    # Check if model is already in app state cache to avoid expensive get_all_models() call
    models = request.app.state.OPENAI_MODELS
    if not models or model_id not in models:
        await get_all_models(request, user=user)
        models = request.app.state.OPENAI_MODELS
    model = models.get(model_id)

    if model:
        idx = model['urlIdx']
    else:
        raise HTTPException(
            status_code=404,
            detail='Model not found',
        )

    # Get the API config for the model
    api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
        str(idx),
        request.app.state.config.OPENAI_API_CONFIGS.get(
            request.app.state.config.OPENAI_API_BASE_URLS[idx], {}
        ),  # Legacy support
    )

    prefix_id = api_config.get('prefix_id', None)
    if prefix_id:
        payload['model'] = payload['model'].replace(f'{prefix_id}.', '')

    # Add user info to the payload if the model is a pipeline
    if 'pipeline' in model and model.get('pipeline'):
        payload['user'] = {
            'name': user.name,
            'id': user.id,
            'email': user.email,
            'role': user.role,
        }

    url = request.app.state.config.OPENAI_API_BASE_URLS[idx]
    key = request.app.state.config.OPENAI_API_KEYS[idx]

    # Check if model is a reasoning model that needs special handling
    if is_openai_new_model(payload['model']):
        payload = openai_reasoning_model_handler(payload)
    elif 'api.openai.com' not in url:
        # Remove "max_completion_tokens" from the payload for backward compatibility
        if 'max_completion_tokens' in payload:
            payload['max_tokens'] = payload['max_completion_tokens']
            del payload['max_completion_tokens']

    if 'max_tokens' in payload and 'max_completion_tokens' in payload:
        del payload['max_tokens']

    # Convert the modified body back to JSON
    if 'logit_bias' in payload and payload['logit_bias']:
        logit_bias = convert_logit_bias_input_to_json(payload['logit_bias'])

        if logit_bias:
            payload['logit_bias'] = json.loads(logit_bias)

    headers, cookies = await get_headers_and_cookies(request, url, key, api_config, metadata, user=user)

    is_responses = api_config.get('api_type') == 'responses'

    if api_config.get('azure', False):
        api_version = api_config.get('api_version', '2023-03-15-preview')
        request_url, payload = convert_to_azure_payload(url, payload, api_version)

        # Only set api-key header if not using Azure Entra ID authentication
        auth_type = api_config.get('auth_type', 'bearer')
        if auth_type not in ('azure_ad', 'microsoft_entra_id'):
            headers['api-key'] = key

        headers['api-version'] = api_version

        if is_responses:
            payload = convert_to_responses_payload(payload)
            request_url = f'{request_url}/responses?api-version={api_version}'
        else:
            request_url = f'{request_url}/chat/completions?api-version={api_version}'
    else:
        if is_responses:
            payload = convert_to_responses_payload(payload)
            request_url = f'{url}/responses'
        else:
            request_url = f'{url}/chat/completions'
    # For Chat Completions, strip image parts from multimodal tool messages
    # (Chat Completions doesn't support images in tool content).
    if not is_responses and 'messages' in payload:
        for message in payload['messages']:
            if message.get('role') == 'tool' and isinstance(message.get('content'), list):
                message['content'] = ''.join(
                    part.get('text', '') for part in message['content'] if part.get('type') in ('input_text', 'text')
                )

    payload = json.dumps(payload)

    r = None
    session = None
    streaming = False
    response = None

    try:
        session = aiohttp.ClientSession(trust_env=True, timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT))

        r = await session.request(
            method='POST',
            url=request_url,
            data=payload,
            headers=headers,
            cookies=cookies,
            ssl=AIOHTTP_CLIENT_SESSION_SSL,
        )

        # Check if response is SSE
        if 'text/event-stream' in r.headers.get('Content-Type', ''):
            streaming = True
            return StreamingResponse(
                stream_wrapper(r, session, stream_chunks_handler),
                status_code=r.status,
                headers=dict(r.headers),
            )
        else:
            try:
                response = await r.json()
            except Exception as e:
                log.error(e)
                response = await r.text()

            if r.status >= 400:
                if isinstance(response, (dict, list)):
                    return JSONResponse(status_code=r.status, content=response)
                else:
                    return PlainTextResponse(status_code=r.status, content=response)

            # Convert Responses API result to simple format
            if is_responses and isinstance(response, dict):
                response = convert_responses_result(response)

            return response
    except Exception as e:
        log.exception(e)

        raise HTTPException(
            status_code=r.status if r else 500,
            detail='Open WebUI: Server Connection Error',
        )
    finally:
        if not streaming:
            await cleanup_response(r, session)

 #--------------------------------------------------------------------------
@router.post('/images/generations') # Генерация изображений
async def generate_image(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user),
):
    url = request.app.state.config.OPENAI_API_BASE_URLS[0]
    key = request.app.state.config.OPENAI_API_KEYS[0]
    
    prompt = form_data.get('prompt', '')
    if not prompt and 'messages' in form_data:
        messages = form_data.get('messages', [])
        for msg in messages:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                if isinstance(content, str):
                    prompt = content
                elif isinstance(content, list):
                    for part in content:
                        if part.get('type') == 'text':
                            prompt = part.get('text', '')
                            break
                break
    
    payload = {
        "model": "qwen-image-lightning",
        "prompt": prompt,
        "n": form_data.get('n', 1),
        "size": form_data.get('size', '1024x1024')
    }
    
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    
    async with aiohttp.ClientSession(trust_env=True) as session:
        try:
            async with session.post(
                f'{url}/images/generations',
                headers=headers,
                json=payload,
                ssl=AIOHTTP_CLIENT_SESSION_SSL,
            ) as r:
                response_data = await r.json()
                
                if r.status >= 400:
                    print(f"[IMAGE GEN] Error: {response_data}")
                    return JSONResponse(status_code=r.status, content=response_data)
                
                print(f"[IMAGE GEN] Success! Image URL: {response_data.get('data', [{}])[0].get('url', 'unknown')}")
                return response_data
                
        except Exception as e:
            log.exception(e)
            raise HTTPException(
                status_code=500,
                detail=f"Image generation failed: {str(e)}",
            )
 #--------------------------------------------------------------------------

async def embeddings(request: Request, form_data: dict, user):
    """
    Calls the embeddings endpoint for OpenAI-compatible providers.

    Args:
        request (Request): The FastAPI request context.
        form_data (dict): OpenAI-compatible embeddings payload.
        user (UserModel): The authenticated user.

    Returns:
        dict: OpenAI-compatible embeddings response.
    """
    idx = 0
    # Prepare payload/body
    body = json.dumps(form_data)
    # Find correct backend url/key based on model
    model_id = form_data.get('model')
    # Check if model is already in app state cache to avoid expensive get_all_models() call
    models = request.app.state.OPENAI_MODELS
    if not models or model_id not in models:
        await get_all_models(request, user=user)
        models = request.app.state.OPENAI_MODELS
    if model_id in models:
        idx = models[model_id]['urlIdx']

    url = request.app.state.config.OPENAI_API_BASE_URLS[idx]
    key = request.app.state.config.OPENAI_API_KEYS[idx]
    api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
        str(idx),
        request.app.state.config.OPENAI_API_CONFIGS.get(url, {}),  # Legacy support
    )

    r = None
    session = None
    streaming = False

    headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)
    try:
        session = aiohttp.ClientSession(
            trust_env=True,
            timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT),
        )
        r = await session.request(
            method='POST',
            url=f'{url}/embeddings',
            data=body,
            headers=headers,
            cookies=cookies,
        )

        if 'text/event-stream' in r.headers.get('Content-Type', ''):
            streaming = True
            return StreamingResponse(
                stream_wrapper(r, session),
                status_code=r.status,
                headers=dict(r.headers),
            )
        else:
            try:
                response_data = await r.json()
            except Exception:
                response_data = await r.text()

            if r.status >= 400:
                if isinstance(response_data, (dict, list)):
                    return JSONResponse(status_code=r.status, content=response_data)
                else:
                    return PlainTextResponse(status_code=r.status, content=response_data)

            return response_data
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=r.status if r else 500,
            detail='Open WebUI: Server Connection Error',
        )
    finally:
        if not streaming:
            await cleanup_response(r, session)


class ResponsesForm(BaseModel):
    model_config = ConfigDict(extra='allow')

    model: str
    input: Optional[list | str] = None
    instructions: Optional[str] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    tools: Optional[list] = None
    tool_choice: Optional[str | dict] = None
    text: Optional[dict] = None
    truncation: Optional[str] = None
    metadata: Optional[dict] = None
    store: Optional[bool] = None
    reasoning: Optional[dict] = None
    previous_response_id: Optional[str] = None


@router.post('/responses')
async def responses(
    request: Request,
    form_data: ResponsesForm,
    user=Depends(get_verified_user),
):
    """
    Forward requests to the OpenAI Responses API endpoint.
    Routes to the correct upstream backend based on the model field.
    """
    payload = form_data.model_dump(exclude_none=True)
    body = json.dumps(payload)

    idx = 0
    model_id = form_data.model
    if model_id:
        models = request.app.state.OPENAI_MODELS
        if not models or model_id not in models:
            await get_all_models(request, user=user)
            models = request.app.state.OPENAI_MODELS
        if model_id in models:
            idx = models[model_id]['urlIdx']

    url = request.app.state.config.OPENAI_API_BASE_URLS[idx]
    key = request.app.state.config.OPENAI_API_KEYS[idx]
    api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
        str(idx),
        request.app.state.config.OPENAI_API_CONFIGS.get(url, {}),  # Legacy support
    )

    r = None
    session = None
    streaming = False

    try:
        headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)

        if api_config.get('azure', False):
            api_version = api_config.get('api_version', '2023-03-15-preview')

            auth_type = api_config.get('auth_type', 'bearer')
            if auth_type not in ('azure_ad', 'microsoft_entra_id'):
                headers['api-key'] = key

            headers['api-version'] = api_version

            model = payload.get('model', '')
            request_url = f'{url}/openai/deployments/{model}/responses?api-version={api_version}'
        else:
            request_url = f'{url}/responses'

        session = aiohttp.ClientSession(
            trust_env=True,
            timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT),
        )
        r = await session.request(
            method='POST',
            url=request_url,
            data=body,
            headers=headers,
            cookies=cookies,
            ssl=AIOHTTP_CLIENT_SESSION_SSL,
        )

        # Check if response is SSE
        if 'text/event-stream' in r.headers.get('Content-Type', ''):
            streaming = True
            return StreamingResponse(
                stream_wrapper(r, session),
                status_code=r.status,
                headers=dict(r.headers),
            )
        else:
            try:
                response_data = await r.json()
            except Exception:
                response_data = await r.text()

            if r.status >= 400:
                if isinstance(response_data, (dict, list)):
                    return JSONResponse(status_code=r.status, content=response_data)
                else:
                    return PlainTextResponse(status_code=r.status, content=response_data)

            return response_data

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=r.status if r else 500,
            detail='Open WebUI: Server Connection Error',
        )
    finally:
        if not streaming:
            await cleanup_response(r, session)


@router.api_route('/{path:path}', methods=['GET', 'POST', 'PUT', 'DELETE'])
async def proxy(path: str, request: Request, user=Depends(get_verified_user)):
    """
    Deprecated: proxy all requests to OpenAI API
    """

    body = await request.body()

    # Parse JSON body to resolve model-based routing
    payload = None
    if body:
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            payload = None

    idx = 0
    model_id = payload.get('model') if isinstance(payload, dict) else None
    if model_id:
        models = request.app.state.OPENAI_MODELS
        if not models or model_id not in models:
            await get_all_models(request, user=user)
            models = request.app.state.OPENAI_MODELS
        if model_id in models:
            idx = models[model_id]['urlIdx']

    url = request.app.state.config.OPENAI_API_BASE_URLS[idx]
    key = request.app.state.config.OPENAI_API_KEYS[idx]
    api_config = request.app.state.config.OPENAI_API_CONFIGS.get(
        str(idx),
        request.app.state.config.OPENAI_API_CONFIGS.get(
            request.app.state.config.OPENAI_API_BASE_URLS[idx], {}
        ),  # Legacy support
    )

    r = None
    session = None
    streaming = False

    try:
        headers, cookies = await get_headers_and_cookies(request, url, key, api_config, user=user)

        if api_config.get('azure', False):
            api_version = api_config.get('api_version', '2023-03-15-preview')

            # Only set api-key header if not using Azure Entra ID authentication
            auth_type = api_config.get('auth_type', 'bearer')
            if auth_type not in ('azure_ad', 'microsoft_entra_id'):
                headers['api-key'] = key

            headers['api-version'] = api_version

            payload = json.loads(body)
            url, payload = convert_to_azure_payload(url, payload, api_version)
            body = json.dumps(payload).encode()

            request_url = f'{url}/{path}?api-version={api_version}'
        else:
            request_url = f'{url}/{path}'

        session = aiohttp.ClientSession(
            trust_env=True,
            timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT),
        )
        r = await session.request(
            method=request.method,
            url=request_url,
            data=body,
            headers=headers,
            cookies=cookies,
            ssl=AIOHTTP_CLIENT_SESSION_SSL,
        )

        # Check if response is SSE
        if 'text/event-stream' in r.headers.get('Content-Type', ''):
            streaming = True
            return StreamingResponse(
                stream_wrapper(r, session),
                status_code=r.status,
                headers=dict(r.headers),
            )
        else:
            try:
                response_data = await r.json()
            except Exception:
                response_data = await r.text()

            if r.status >= 400:
                if isinstance(response_data, (dict, list)):
                    return JSONResponse(status_code=r.status, content=response_data)
                else:
                    return PlainTextResponse(status_code=r.status, content=response_data)

            return response_data

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=r.status if r else 500,
            detail='Open WebUI: Server Connection Error',
        )
    finally:
        if not streaming:
            await cleanup_response(r, session)