######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.11.09
# filenaem: logger.py
# function: wandb / tensorboard logging for segmentation training.
#
# the wandb integration relies on the wandb sdk only, no re-implementation:
#   wandb.init()   creates the run
#   wandb.watch()  tracks the gradients / the weights of the model
#   run.log()      logs the metrics of an epoch
#   run.finish()   closes the run
######################################################
import logging
import os
import re
import subprocess
import time

_logger = logging.getLogger("logger")

# the official wandb server
OFFICIAL_WANDB_BASE_URL = "https://api.wandb.ai"
# the official server refuses the connections from china ('403 Forbidden'),
# this mirror forwards the requests to it
MIRROR_WANDB_BASE_URL = "https://api.bandw.top"
# the server that is used when nothing is configured; the official one is
# blocked from this network, so the mirror is used
DEFAULT_WANDB_BASE_URL = MIRROR_WANDB_BASE_URL
# the project that is used when the branch name is not available
DEFAULT_PROJECT_NAME = "pytorch-toolkit-dev"
# the timeout of the wandb http requests; the wandb service uses 10 seconds,
# which is too short for the mirror that is used from china
DEFAULT_WANDB_HTTP_TIMEOUT = 60


def wandb_api(*args, retries=3, delay=3.0, **kwargs):
    """Create a wandb api client, retrying when the service is too slow.

    The wandb service answers an api request with a hardcoded timeout of 10
    seconds; the mirror that is used from china is often slower than that, so
    the call fails with 'the service process is busy and did not respond in
    time'.  The call is retried a few times before the error is raised.

    Returns the api client.
    """
    last_error = None
    for attempt in range(retries):
        try:
            return wandb.Api(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if "busy and" not in str(exc):
                raise
            wait = delay * (attempt + 1)
            _logger.warning(
                f"the wandb api did not answer in time ({exc}); retrying in {wait:.0f}s "
                f"({attempt + 1}/{retries})."
            )
            time.sleep(wait)
    raise last_error

try:
    import wandb

    has_wandb = True
except ImportError:
    wandb = None
    has_wandb = False

try:
    from torch.utils.tensorboard import SummaryWriter

    has_tensorboard = True
except ImportError:
    SummaryWriter = None
    has_tensorboard = False


def get_git_branch(path=None):
    """Return the name of the current git branch.

    Returns an empty string when the folder is not a git repository (or when
    git is not installed).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=path or os.getcwd(),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""

    if result.returncode != 0:
        return ""

    branch = result.stdout.strip()
    # a detached head has no branch name
    if not branch or branch == "HEAD":
        return ""
    return branch


def sanitize_project_name(name, fallback="pytorch-toolkit-dev"):
    """Turn a branch name into a name that wandb accepts.

    The project name is used in the url of the run, so the characters that
    are not allowed are replaced by a dash.  The letters that are not ascii
    are kept: wandb accepts them and a chinese branch name stays readable.
    """
    if not name:
        return fallback

    # keep the letters (also the ones that are not ascii), the digits, the
    # dash, the dot and the underscore
    cleaned = re.sub(r"[^0-9A-Za-z.\-_^\u0080-\uffff]+", "-", name).strip("-.")
    return cleaned or fallback


class SegLogger:
    """Thin wrapper around wandb and tensorboard.

    Both backends are optional: if the package is missing (or the feature is
    disabled in the config) the corresponding calls become no-ops.
    """

    def __init__(self, cfg=None, exp_name="", log_dir=None, log_wandb=True, tensorboard=""):
        self.cfg = cfg or {}
        self.exp_name = exp_name
        self.wandb_run = None
        self.writer = None

        if log_wandb:
            self.init_wandb()
        if tensorboard:
            self.init_tensorboard(tensorboard)

    # ---------------- wandb ----------------
    def _resolve_project(self, wandb_cfg):
        """Return the name of the wandb project of the run.

        The project is named after the current git branch, so the runs of the
        different branches do not get mixed up.  The name can be forced with
        'wandb.project' in the config; 'wandb.project: branch' keeps the
        branch naming.
        """
        configured = (wandb_cfg.get("project") or "").strip()
        branch = get_git_branch(self.cfg.get("repo_root"))

        if configured and configured.lower() != "branch":
            # a name is configured: it is used as it is
            return sanitize_project_name(configured, DEFAULT_PROJECT_NAME)

        if not branch:
            # not a git repository: fall back to the experiment name
            return self.cfg.get("experiment") or DEFAULT_PROJECT_NAME

        # the default: the branch name alone
        return sanitize_project_name(branch, DEFAULT_PROJECT_NAME)

    def init_wandb(self):
        """Start a wandb run; returns None when wandb stays disabled.

        The run is not started when the package is missing or when the api key
        is missing; the training then continues without any wandb log instead
        of crashing.
        """
        if not has_wandb:
            _logger.warning(
                "You've requested to log metrics to wandb but the package is not found. "
                "Metrics will not be logged to wandb, try `pip install wandb`."
            )
            return None

        wandb_cfg = self.cfg.get("wandb", {}) or {}
        project = self._resolve_project(wandb_cfg)
        entity = wandb_cfg.get("entity", "")
        group = wandb_cfg.get("group", "")
        job_type = wandb_cfg.get("job_type", "seg-train")
        mode = wandb_cfg.get("mode", "online")
        base_url = (
            wandb_cfg.get("base_url", "")
            or os.environ.get("WANDB_BASE_URL", "")
            or DEFAULT_WANDB_BASE_URL
        )

        # the wandb service answers an api request with a hardcoded timeout of
        # 10 seconds; the mirror is often slower than that, so the timeout is
        # raised.  the env var must be set before the service is started.
        http_timeout = wandb_cfg.get("http_timeout", DEFAULT_WANDB_HTTP_TIMEOUT)
        if not os.environ.get("WANDB_HTTP_TIMEOUT"):
            os.environ["WANDB_HTTP_TIMEOUT"] = str(int(http_timeout))

        init_kwargs = dict(project=project, name=self.exp_name, job_type=job_type, config=self.cfg)
        if entity:
            init_kwargs["entity"] = entity
        if group:
            init_kwargs["group"] = group
        if mode:
            init_kwargs["mode"] = mode

        # the api key can be given in the config, otherwise the one of the
        # environment / of the 'wandb login' is used
        api_key = wandb_cfg.get("api_key", "") or os.environ.get("WANDB_API_KEY", "")
        if api_key:
            # 'wandb.init' has no 'api_key' argument, the key is passed
            # through the environment
            os.environ["WANDB_API_KEY"] = api_key

        # the system metrics are sent by an opentelemetry exporter; it is
        # useless when the server is not reachable and it floods the log
        settings = {}
        if base_url:
            settings["base_url"] = base_url
        if not wandb_cfg.get("observability", False):
            settings["x_disable_stats"] = True
        if settings:
            init_kwargs["settings"] = wandb.Settings(**settings)

        try:
            self.wandb_run = wandb.init(**init_kwargs)
        except Exception as exc:
            message = str(exc)
            blocked = "403" in message or "Forbidden" in message

            if blocked and base_url != MIRROR_WANDB_BASE_URL:
                # the server refused the connections (it is blocked on this
                # network); the mirror forwards them to the official one
                _logger.warning(
                    f"the wandb api {base_url} returned 403 Forbidden: it is blocked "
                    f"on this network; retrying with the mirror {MIRROR_WANDB_BASE_URL}."
                )
                self._stop_otel()
                self.wandb_run = self._init_online(
                    project, self.exp_name, job_type, init_kwargs, MIRROR_WANDB_BASE_URL
                )
                if self.wandb_run is not None:
                    return self.wandb_run

            _logger.warning(
                f"could not start the wandb run ({message}); the training continues "
                "without any wandb log."
            )
            if blocked:
                # the request reached a server that refused it: the wandb api
                # is not reachable from this host (network / region block)
                _logger.warning(
                    "the wandb api returned 403 Forbidden: it is blocked on this "
                    "network. Use 'wandb.mode: offline' to log locally, or set "
                    "'wandb.base_url' to a reachable server."
                )
            else:
                _logger.warning(
                    "Check the api key ('wandb login') or use 'wandb.mode: offline'."
                )
            self.wandb_run = None

            # the opentelemetry exporter of the system metrics is started in a
            # background thread; it keeps failing against the unreachable
            # server, so it is stopped here
            self._stop_otel()

            # fall back to the local logs when the server cannot be reached
            if mode not in ("offline", "disabled") and wandb_cfg.get("offline_fallback", True):
                _logger.warning("falling back to the offline mode, the logs stay on the disk.")
                self.wandb_run = self._init_offline(project, self.exp_name, job_type, init_kwargs)
        return self.wandb_run

    def _init_online(self, project, name, job_type, init_kwargs, base_url):
        """Start a wandb run against the given server."""
        try:
            kwargs = dict(init_kwargs)
            kwargs["mode"] = "online"
            kwargs["settings"] = wandb.Settings(base_url=base_url)
            return wandb.init(**kwargs)
        except Exception as exc:
            _logger.warning(f"could not start the wandb run on {base_url} ({exc}).")
            return None

    def _init_offline(self, project, name, job_type, init_kwargs):
        """Start a wandb run in offline mode (the logs stay on the disk)."""
        try:
            kwargs = dict(init_kwargs)
            kwargs["mode"] = "offline"
            kwargs.pop("settings", None)
            return wandb.init(**kwargs)
        except Exception as exc:
            _logger.warning(f"could not start the wandb run in offline mode ({exc}).")
            return None

    def _stop_otel(self):
        """Stop the opentelemetry exporters of the system metrics.

        They are started in a background thread by the wandb service and they
        keep failing against a server that is not reachable.
        """
        try:
            from wandb.analytics.opentelemetry import opentelemetry_proxy as otel
        except Exception:
            return

        # the proxy that is already running is shut down
        proxy = getattr(otel, "_singleton_telemetry_proxy", None)
        if proxy is not None:
            try:
                proxy.shutdown()
            except Exception:
                pass
            try:
                otel._singleton_telemetry_proxy = None
            except Exception:
                pass

        # and no new one is created
        try:
            otel.disable()
        except Exception:
            pass

    def watch(self, model, criterion=None, log="gradients", log_freq=1000):
        """Track the gradients (and the weights) of the model on wandb.

        The call is a no-op when wandb is disabled.
        """
        if self.wandb_run is None:
            return None
        try:
            self.wandb_run.watch(model, criterion=criterion, log=log, log_freq=log_freq)
        except Exception as exc:
            _logger.warning(f"could not watch the model on wandb ({exc}).")
        return self.wandb_run

    # ---------------- tensorboard ----------------
    def init_tensorboard(self, log_dir):
        if not has_tensorboard:
            _logger.warning("tensorboard is not installed, skip the tensorboard logger.")
            return None
        path = os.path.join(log_dir, self.exp_name) if self.exp_name else log_dir
        self.writer = SummaryWriter(log_dir=path, comment=self.exp_name)
        return self.writer

    # ---------------- update ----------------
    def update(self, epoch, metrics, prefix=""):
        """Log a dict of scalar metrics for the given epoch."""
        metrics = metrics or {}
        for key, value in metrics.items():
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            name = f"{prefix}{key}"
            if self.writer is not None:
                self.writer.add_scalar(name, value, epoch)
            if self.wandb_run is not None:
                self.wandb_run.log({name: value}, step=epoch)

    def log_config(self):
        """Log the resolved config to wandb (already passed at init time)."""
        return self.wandb_run

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None
