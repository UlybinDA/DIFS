from functools import wraps
from loguru import logger
import sys
from dash.exceptions import PreventUpdate

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
    colorize=True,
)
logger.level("START", no=26, color="<magenta>")
logger.level("PREVENT_U", no=26, color="<white>")

log_file = "app.log"
logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}", rotation="10 MB", enqueue=True)
logger.disable("START")
logger.add("app.log", enqueue=True)


def mylogger( level='INFO', log_args=False, log_result=False, apply_function=None, return_on_error=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                logger.log('START', f"Function {func.__name__} fired")

                if log_args:
                    logger.log(level, f"Function's {func.__name__} argument's: args={args}, kwargs={kwargs}")

                result = func(*args, **kwargs)

                if log_result:
                    logger.log(level, f"Function's {func.__name__} results: {result}")

                if apply_function:
                    applied_result = apply_function(result)
                    logger.log(level, f"Applied function to function's {func.__name__} result: {applied_result}")
                logger.log('SUCCESS', f"Function {func.__name__} ended with no issues")
                return result
            except Exception as e:
                if isinstance(e,PreventUpdate):
                    logger.log('PREVENT_U', f"Callback {func.__name__} update prevented: {e}")
                else:
                    logger.error(f"There is an error in function {func.__name__}: {e}")
                if return_on_error:
                    return None
                elif not return_on_error:
                    raise e
        return wrapper
    return decorator