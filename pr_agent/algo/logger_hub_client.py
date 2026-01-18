"""
Logger Hub client for centralized AI model logging.

Provides async HTTP client to send AI model request/response logs
to the Logger Hub service for debugging purposes.
"""
import aiohttp
import asyncio
import time
from typing import Optional, Dict, Any
from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger

LOGGER_HUB_TIMEOUT = 5  # seconds


async def send_to_logger_hub(collection: str, data: dict) -> bool:
    """
    Send log data to Logger Hub service (fire-and-forget pattern).
    
    This function sends log data asynchronously and does not block the main
    AI request flow. Failures are logged but do not affect the main operation.
    
    Args:
        collection: MongoDB collection name to store the log
        data: Log data dictionary with any fields suitable for debugging
    
    Returns:
        True if log was successfully sent, False otherwise
    """
    endpoint = get_settings().get("LOGGER_HUB.ENDPOINT", "")
    api_key = get_settings().get("LOGGER_HUB.API_KEY", "")
    
    if not endpoint or not api_key:
        get_logger().debug("Logger Hub not configured, skipping log")
        return False
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint}/log",
                json={"collection": collection, "data": data},
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key
                },
                timeout=aiohttp.ClientTimeout(total=LOGGER_HUB_TIMEOUT)
            ) as response:
                if response.status == 200:
                    get_logger().debug(f"Logger Hub: sent log to collection '{collection}'")
                    return True
                else:
                    get_logger().debug(f"Logger Hub: failed with status {response.status}")
                    return False
    except asyncio.TimeoutError:
        get_logger().debug("Logger Hub: request timed out")
        return False
    except Exception as e:
        get_logger().debug(f"Logger Hub: send failed - {e}")
        return False


def log_ai_request_sync(collection: str, data: dict) -> None:
    """
    Synchronous wrapper to fire-and-forget an AI log.
    
    Creates a new event loop task to send the log without blocking.
    Safe to call from both sync and async contexts.
    
    Args:
        collection: MongoDB collection name
        data: Log data dictionary
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, create a task
            asyncio.create_task(send_to_logger_hub(collection, data))
        else:
            # We're in a sync context, run until complete
            loop.run_until_complete(send_to_logger_hub(collection, data))
    except RuntimeError:
        # No event loop, create a new one
        asyncio.run(send_to_logger_hub(collection, data))
    except Exception as e:
        get_logger().debug(f"Logger Hub: sync wrapper failed - {e}")


def build_ai_log_data(
    model: str,
    system_prompt: str,
    user_prompt: str,
    response: Optional[str] = None,
    finish_reason: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
    status: str = "success",
    error: Optional[str] = None,
    latency_ms: Optional[float] = None,
    pr_url: Optional[str] = None,
    command: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build a structured log data dictionary for AI model calls.
    
    Args:
        model: AI model name
        system_prompt: Full system prompt sent to model
        user_prompt: Full user prompt sent to model
        response: Full AI response text
        finish_reason: Model completion finish reason
        usage: Token usage statistics (prompt_tokens, completion_tokens, total_tokens)
        status: "success" or "error"
        error: Error message if status is "error"
        latency_ms: Request latency in milliseconds
        pr_url: PR URL being processed
        command: PR Agent command (review, describe, improve, etc.)
        extra: Additional context data
    
    Returns:
        Structured log data dictionary
    """
    data = {
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "status": status,
        "timestamp": time.time(),
    }
    
    if response is not None:
        data["response"] = response
    if finish_reason is not None:
        data["finish_reason"] = finish_reason
    if usage is not None:
        data["usage"] = usage
    if error is not None:
        data["error"] = error
    if latency_ms is not None:
        data["latency_ms"] = latency_ms
    if pr_url is not None:
        data["pr_url"] = pr_url
    if command is not None:
        data["command"] = command
    if extra is not None:
        data["extra"] = extra
    
    return data
