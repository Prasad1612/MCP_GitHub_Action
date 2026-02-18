import os
import requests
import json
import logging
import inspect
from google import genai
from google.genai import types
from main import mcp

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    message = os.getenv("TELEGRAM_MESSAGE")

    if not all([api_key, bot_token, chat_id, message]):
        logger.error("Missing environment variables: GOOGLE_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_MESSAGE")
        return

    logger.info(f"Processing message for chat {chat_id}: {message}")

    try:
        client = genai.Client(api_key=api_key)

        # Extract and sanitize tools from FastMCP instance
        tools_functions = []
        for tool in mcp._tool_manager.list_tools():
            f = tool.fn
            # Gemini's SDK struggles with parameters that have None defaults but no type hints.
            # We patch these at runtime to ensure a valid JSON schema can be generated.
            try:
                sig = inspect.signature(f)
                hints = getattr(f, '__annotations__', {})
                changed = False
                for name, param in sig.parameters.items():
                    if name not in hints and param.default is None:
                        # Default to str if no hint and default is None
                        hints[name] = str
                        changed = True
                if changed:
                    f.__annotations__ = hints
                    logger.debug(f"Patched type hints for tool: {tool.name}")
            except Exception as patch_err:
                logger.warning(f"Could not patch tool {tool.name}: {patch_err}")
            
            tools_functions.append(f)
            
        logger.info(f"Loaded and sanitized {len(tools_functions)} tools from main.py")

        # Call Gemini with automatic tool calling
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=message,
            config=types.GenerateContentConfig(
                tools=tools_functions,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False)
            )
        )

        final_text = response.text
        if not final_text:
            # If no text response, maybe there was an error or empty response
            final_text = "I couldn't process that request properly. Please try again."
        
        logger.info(f"AI Response truncated: {final_text[:100]}...")

        # Send to Telegram
        telegram_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": final_text,
            "parse_mode": "Markdown"
        }
        res = requests.post(telegram_url, json=payload)
        res.raise_for_status()
        logger.info("Message sent to Telegram successfully")

    except Exception as e:
        logger.error(f"Error in agent processing: {e}")
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        # Minimal error reporting to user
        try:
            requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage", 
                         json={"chat_id": chat_id, "text": error_msg[:4000]})
        except:
            pass

if __name__ == "__main__":
    main()
