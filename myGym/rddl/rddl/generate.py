#!/usr/bin/env python
"""RDDL task generator — end-user CLI.

Generates sequences of robotic-manipulation tasks from an RDDL domain and writes them to
YAML. This is the production counterpart to ``tests/test_world.py`` (which is a
test/showcase, not a user tool).

Configuration sources, in increasing priority (later overrides earlier):

    1. built-in defaults (and the ``RDDLWorld`` defaults they map to)
    2. a YAML config file (``--config``)
    3. command-line options

i.e. the YAML supplies initial values, explicit CLI options override them, and anything
left unset falls back to the defaults.

A *domain* module (``--domain``, default ``testing_utils``) must register the simulator
function mapping (``Operand.set_mapping``), the observation getter, and the concrete entity
classes. It is imported *before* the rest of ``rddl`` because predicate/reward classes
resolve their ``_0_*`` hooks at import time.

Examples
--------
    # one-shot generation, lengths 2..5, 10 samples each, to tasks.yaml
    python generate.py -l 2 -l 3 -l 4 -l 5 -n 10 -o tasks.yaml

    # drive everything from a YAML config, override just the seed on the CLI
    python generate.py -c config.yaml -s 42

    # restrict the domain and inspect what is available
    python generate.py -d testing_utils --inspect

YAML config keys are the long option names with underscores, e.g.::

    sequence_lengths: [2, 3, 4]
    n_repeats: 10
    seed: 42
    allowed_actions: [Approach, Grasp, Move, Drop]
    allowed_initial_actions: [Approach]
    allowed_entities: [TiagoGripper, Apple, Bowl]
    action_weights: [2.0, 1.0, 1.0, 1.0]
    object_weights: {Apple: 2.0, Bowl: 0.5}
    weight_mode: [weight, sequence, random]
    sample_single_object_per_class: true
"""
from __future__ import annotations

import datetime
import importlib
import importlib.util
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import typer
import yaml

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    help="Generate robotic-manipulation task sequences from an RDDL domain.",
)

# Valid Weighter mode flag names -> Weighter.MODE_* constants are resolved lazily (after the
# domain import makes rddl importable). These are the accepted spellings.
_WEIGHT_FLAGS = ("initial", "weight", "sequence", "random", "max-noise", "none")


class Method(str, Enum):
    one_shot = "one-shot"
    recursive = "recursive"


class Detail(str, Enum):
    """How much to record/print per task.

    concise  -- action sequence + per-step bound objects + the object set (default).
    full     -- concise + the task's initial-state and final-state predicates.
    detailed -- full + the predicate state after each individual action (à la test_world.yaml).
    """

    concise = "concise"
    full = "full"
    detailed = "detailed"


# --------------------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------------------- #
def _info(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.CYAN)


def _ok(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.GREEN)


def _warn(msg: str) -> None:
    typer.secho(f"warning: {msg}", fg=typer.colors.YELLOW, err=True)


def _fail(msg: str) -> "typer.Exit":
    typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


def _load_config(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise _fail(f"config file not found: {path}")
    try:
        with path.open() as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise _fail(f"could not parse YAML config {path}: {exc}")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise _fail(f"config file {path} must contain a YAML mapping at the top level.")
    return data


def _from_cli(ctx: typer.Context, name: str) -> bool:
    """True if the option was given explicitly on the command line.

    Compares the parameter source by name rather than enum identity: typer/click may expose
    ``ParameterSource`` via a different import path than ``click.core``, in which case an
    ``is``/``==`` comparison against an imported enum member would wrongly fail.
    """
    source = ctx.get_parameter_source(name)
    return source is not None and source.name == "COMMANDLINE"


def _resolve(ctx: typer.Context, cfg: dict[str, Any], name: str, cli_value: Any) -> Any:
    """Merge one option: explicit CLI wins, then YAML, then the typer default (``cli_value``)."""
    if _from_cli(ctx, name):
        return cli_value
    if name in cfg:
        return cfg[name]
    return cli_value


def _load_domain(domain: str):
    """Import the domain setup module (sets the function mapping + defines entities).

    Accepts either an importable module name or a path to a ``.py`` file.
    """
    candidate = Path(domain)
    if domain.endswith(".py") or candidate.exists():
        if not candidate.exists():
            raise _fail(f"domain file not found: {domain}")
        module_name = candidate.stem
        spec = importlib.util.spec_from_file_location(module_name, str(candidate))
        if spec is None or spec.loader is None:
            raise _fail(f"could not load domain file: {domain}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        sys.path.insert(0, str(candidate.resolve().parent))
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - surface any domain setup error clearly
            raise _fail(f"failed to import domain file {domain}: {exc}")
        return module
    # module name: make the bundled test domain and the project importable
    here = Path(__file__).resolve().parent
    for extra in (here, here / "tests", Path.cwd(), Path.cwd() / "tests"):
        if extra.is_dir() and str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    try:
        return importlib.import_module(domain)
    except Exception as exc:  # noqa: BLE001
        raise _fail(
            f"failed to import domain module '{domain}': {exc}\n"
            f"       provide an importable module or a path to a .py file via --domain/-d."
        )


def _resolve_class(name: str, namespaces: list, expected_base: type, kind: str) -> type:
    """Resolve a class name to a subclass of ``expected_base`` found in ``namespaces``.

    Tries an exact match first, then falls back to a case-insensitive match (so e.g.
    ``approach`` resolves to ``Approach``). Raises on no match, or on an ambiguous
    case-insensitive match that maps to more than one distinct class.
    """
    # exact match wins
    for ns in namespaces:
        obj = getattr(ns, name, None)
        if isinstance(obj, type) and issubclass(obj, expected_base):
            return obj
    # case-insensitive fallback
    lowered = name.lower()
    matches: set[type] = set()
    for ns in namespaces:
        for n in dir(ns):
            obj = getattr(ns, n, None)
            if (
                isinstance(obj, type)
                and issubclass(obj, expected_base)
                and obj is not expected_base
                and n.lower() == lowered
            ):
                matches.add(obj)
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        names = ", ".join(sorted(c.__name__ for c in matches))
        raise _fail(f"ambiguous {kind} '{name}' (case-insensitive) matches: {names}. Use exact case.")
    available = sorted(
        {
            n
            for ns in namespaces
            for n in dir(ns)
            if isinstance(getattr(ns, n, None), type)
            and issubclass(getattr(ns, n), expected_base)
            and getattr(ns, n) is not expected_base
        }
    )
    raise _fail(f"unknown {kind} '{name}'. Available {kind}s: {', '.join(available) or '(none)'}")


def _resolve_classes(names, namespaces, expected_base, kind) -> Optional[list]:
    if names is None:
        return None
    return [_resolve_class(n, namespaces, expected_base, kind) for n in names]


def _parse_object_weights(raw, namespaces, entity_base) -> Optional[dict]:
    """Accept a YAML mapping {Name: weight} or a CLI list of 'Name=weight' strings."""
    if raw is None:
        return None
    pairs: dict[str, float] = {}
    if isinstance(raw, dict):
        pairs = {str(k): float(v) for k, v in raw.items()}
    else:  # list of "Name=weight"
        for item in raw:
            if "=" not in str(item):
                raise _fail(f"object weight '{item}' must be in the form Name=weight.")
            name, _, value = str(item).partition("=")
            try:
                pairs[name.strip()] = float(value)
            except ValueError:
                raise _fail(f"object weight for '{name.strip()}' is not a number: '{value}'.")
    return {_resolve_class(name, namespaces, entity_base, "entity"): w for name, w in pairs.items()}


def _compute_weight_mode(names, weighter) -> Optional[int]:
    if not names:
        return None
    flag_map = {
        "initial": weighter.MODE_INITIAL,
        "weight": weighter.MODE_WEIGHT,
        "sequence": weighter.MODE_SEQUENCE,
        "random": weighter.MODE_RANDOM,
        "max-noise": weighter.MODE_MAX_NOISE,
        "max_noise": weighter.MODE_MAX_NOISE,
        "none": weighter.MODE_NONE,
    }
    mode = 0
    for raw in names:
        key = str(raw).strip().lower()
        if key not in flag_map:
            raise _fail(
                f"unknown weight-mode flag '{raw}'. Valid flags: {', '.join(_WEIGHT_FLAGS)}."
            )
        mode |= flag_map[key]
    return mode


def _format_predicates(container) -> list:
    """Render a symbolic state's predicates as sorted ``Pred(arg, ...) -> bool`` strings."""
    try:
        preds = container.get_predicates()
    except Exception:  # noqa: BLE001 - predicate read-out is best-effort, never fatal
        return []
    rendered = []
    for pclass, args, value in preds:
        name = pclass.__name__ if isinstance(pclass, type) else str(pclass)
        rendered.append(f"{name}({', '.join(map(str, args))}) -> {value}")
    return sorted(rendered)


def _extract_task_record(task, detail: Detail = Detail.concise) -> dict:
    """Build a serialisable task record at the requested :class:`Detail` level.

    concise  -> ``action_list`` + per-step ``{action, action_objects}`` + ``all_objects``.
    full     -> concise + ``initial_state`` / ``final_state`` predicate listings.
    detailed -> full + each step's post-state predicates under ``predicates``.
    """
    # materialise the iterator once: get_actions() returns a one-shot iterator
    actions = list(task.get_actions())
    variables = task.gather_objects()
    states = getattr(task, "_states", None)  # per-step post-states, aligned with actions

    sequence = []
    for i, a in enumerate(actions):
        entry = {"action": str(a), "action_objects": {k: v.name for k, v in a.variables.items()}}
        if detail is Detail.detailed and states is not None and i < len(states):
            entry["predicates"] = _format_predicates(states[i])
        sequence.append(entry)

    record = {
        "action_list": [a.__class__.__name__ for a in actions],
        "sequence": sequence,
        "all_objects": {v.name: v.type.__name__ for v in variables},
    }
    if detail in (Detail.full, Detail.detailed):
        record["initial_state"] = _format_predicates(task.initial_state)
        record["final_state"] = _format_predicates(task.final_state)
    return record


# --------------------------------------------------------------------------------------- #
# Command
# --------------------------------------------------------------------------------------- #
@app.command()
def generate(
    ctx: typer.Context,
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file (initial values; overridden by CLI options)."),
    domain: str = typer.Option("testing_utils", "--domain", "-d", help="Domain module name or path to a .py file that registers the function mapping and entities."),
    sequence_lengths: Optional[list[int]] = typer.Option(None, "--seq-len", "-l", help="Action-sequence length(s); repeat -l for several. [default: 4]"),
    n_repeats: int = typer.Option(1, "--repeats", "-n", min=1, help="Samples per length (one-shot method)."),
    method: Method = typer.Option(Method.one_shot, "--method", "-m", help="Generation strategy."),
    n_samples: Optional[int] = typer.Option(None, "--n-samples", "-N", help="Tasks per length (recursive method). [default: 10]"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="RNG seed for reproducible generation."),
    add_robots: bool = typer.Option(True, "--robots/--no-robots", "-r", help="Seed the world with a gripper."),
    retry_ad_infinitum: bool = typer.Option(True, "--retry/--no-retry", "-T", help="Keep retrying actions until a valid step is found (one-shot)."),
    allowed_entities: Optional[list[str]] = typer.Option(None, "--entities", "-e", help="Restrict the entity universe (class names; repeat -e)."),
    allowed_actions: Optional[list[str]] = typer.Option(None, "--actions", "-a", help="Restrict the action set (class names; repeat -a)."),
    allowed_initial_actions: Optional[list[str]] = typer.Option(None, "--initial-actions", "-i", help="Restrict actions allowed as the first step (repeat -i)."),
    sample_single_object_per_class: bool = typer.Option(False, "--single-object/--multi-object", "-u", help="Use at most one object per entity class."),
    action_weights: Optional[list[float]] = typer.Option(None, "--action-weights", "-W", help="Per-action sampling weights; must match the number of allowed actions (repeat -W)."),
    object_weights: Optional[list[str]] = typer.Option(None, "--object-weight", "-O", help="Per-entity weights as Name=weight (repeat -O). YAML may use a mapping."),
    weight_mode: Optional[list[str]] = typer.Option(None, "--weight-mode", "-w", help=f"Weighter mode flags ({', '.join(_WEIGHT_FLAGS)}); repeat -w to combine."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output YAML file. If omitted, nothing is written."),
    detail: Detail = typer.Option(Detail.concise, "--detail", "-D", help="Record/print detail: concise (default), full (+state predicates), detailed (+per-step predicates)."),
    print_tasks: bool = typer.Option(False, "--print/--no-print", "-P", help="Print each generated task to the console."),
    inspect_domain: bool = typer.Option(False, "--inspect", "-I", help="List entities/actions/predicates and required mappings, then exit."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Resolve and validate configuration, print the plan, but do not generate."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output (echo the resolved configuration)."),
) -> None:
    """Generate task sequences and (optionally) write them to a YAML file."""
    cfg = _load_config(config_file)

    # ---- merge config (CLI > YAML > default) -------------------------------------------- #
    domain = _resolve(ctx, cfg, "domain", domain)
    sequence_lengths = _resolve(ctx, cfg, "sequence_lengths", sequence_lengths)
    n_repeats = _resolve(ctx, cfg, "n_repeats", n_repeats)
    method_value = _resolve(ctx, cfg, "method", method.value if isinstance(method, Method) else method)
    try:
        method = Method(method_value)
    except ValueError:
        raise _fail(f"invalid method '{method_value}'. Valid methods: {', '.join(m.value for m in Method)}.")
    n_samples = _resolve(ctx, cfg, "n_samples", n_samples)
    seed = _resolve(ctx, cfg, "seed", seed)
    add_robots = _resolve(ctx, cfg, "add_robots", add_robots)
    retry_ad_infinitum = _resolve(ctx, cfg, "retry_ad_infinitum", retry_ad_infinitum)
    allowed_entities = _resolve(ctx, cfg, "allowed_entities", allowed_entities)
    allowed_actions = _resolve(ctx, cfg, "allowed_actions", allowed_actions)
    allowed_initial_actions = _resolve(ctx, cfg, "allowed_initial_actions", allowed_initial_actions)
    sample_single_object_per_class = _resolve(ctx, cfg, "sample_single_object_per_class", sample_single_object_per_class)
    action_weights = _resolve(ctx, cfg, "action_weights", action_weights)
    object_weights = _resolve(ctx, cfg, "object_weights", object_weights)
    weight_mode = _resolve(ctx, cfg, "weight_mode", weight_mode)
    output = _resolve(ctx, cfg, "output", output)
    print_tasks = _resolve(ctx, cfg, "print_tasks", print_tasks)
    detail_value = _resolve(ctx, cfg, "detail", detail.value if isinstance(detail, Detail) else detail)
    try:
        detail = Detail(detail_value)
    except ValueError:
        raise _fail(f"invalid detail '{detail_value}'. Valid levels: {', '.join(d.value for d in Detail)}.")

    # normalise scalar -> list where a YAML user may have supplied a single value
    if isinstance(sequence_lengths, int):
        sequence_lengths = [sequence_lengths]
    if sequence_lengths is None:
        sequence_lengths = [4]
    if output is not None:
        output = Path(output)

    # ---- import the domain, then the rest of rddl (order matters) ----------------------- #
    _info(f"Loading domain '{domain}' ...")
    _load_domain(domain)
    try:
        import rddl.actions as actions_ns
        import rddl.entities as entities_ns
        import rddl.predicates as predicates_ns
        from rddl import AtomicAction, Entity, Operand
        from rddl.core import Predicate
        from rddl.rddl_sampler import RDDLWorld, Weighter
        from rddl.rddl_task import RDDLTask  # noqa: F401  (constructed inside the sampler)
    except Exception as exc:  # noqa: BLE001
        raise _fail(f"failed to import rddl after loading the domain: {exc}")

    domain_module = sys.modules.get(Path(domain).stem if domain.endswith(".py") else domain)
    entity_ns = [domain_module, entities_ns] if domain_module else [entities_ns]
    action_ns = [domain_module, actions_ns] if domain_module else [actions_ns]
    predicate_ns = [domain_module, predicates_ns] if domain_module else [predicates_ns]

    # ---- inspection mode ---------------------------------------------------------------- #
    if inspect_domain:
        _print_inspection(entity_ns, action_ns, predicate_ns, Entity, AtomicAction, Predicate, Operand)
        raise typer.Exit()

    # ---- resolve class references ------------------------------------------------------- #
    allowed_entities_cls = _resolve_classes(allowed_entities, entity_ns, Entity, "entity")
    allowed_actions_cls = _resolve_classes(allowed_actions, action_ns, AtomicAction, "action")
    allowed_initial_cls = _resolve_classes(allowed_initial_actions, action_ns, AtomicAction, "action")
    object_weights_cls = _parse_object_weights(object_weights, entity_ns, Entity)
    mode = _compute_weight_mode(weight_mode, Weighter)

    # ---- validation & warnings ---------------------------------------------------------- #
    if any(L <= 0 for L in sequence_lengths):
        raise _fail(f"sequence lengths must be positive, got {sequence_lengths}.")

    effective_actions = allowed_actions_cls if allowed_actions_cls is not None else list(RDDLWorld.VALID_ACTIONS)
    if action_weights is not None and len(action_weights) != len(effective_actions):
        raise _fail(
            f"action_weights has {len(action_weights)} entries but there are "
            f"{len(effective_actions)} allowed actions "
            f"({', '.join(a.__name__ for a in effective_actions)})."
        )

    if allowed_initial_cls is not None and allowed_actions_cls is not None:
        stray = [a.__name__ for a in allowed_initial_cls if a not in allowed_actions_cls]
        if stray:
            _warn(f"initial action(s) {stray} are not in --actions; they can never be sampled.")

    if method is Method.recursive:
        if n_samples is None:
            n_samples = 10
            _warn("recursive method without --n-samples: defaulting to 10 (the full tree can be huge).")
        if _from_cli(ctx, "n_repeats"):
            _warn("--repeats is ignored by the recursive method (use --n-samples).")
        if _from_cli(ctx, "retry_ad_infinitum"):
            _warn("--retry/--no-retry is ignored by the recursive method.")

    if output is None and not print_tasks and not dry_run:
        _warn("no --output and no --print: generated tasks will be discarded. Use -o or -P.")

    # ---- echo plan ---------------------------------------------------------------------- #
    plan = {
        "domain": domain,
        "method": method.value,
        "sequence_lengths": sequence_lengths,
        "n_repeats": n_repeats if method is Method.one_shot else None,
        "n_samples": n_samples if method is Method.recursive else None,
        "seed": seed,
        "add_robots": add_robots,
        "retry_ad_infinitum": retry_ad_infinitum if method is Method.one_shot else None,
        "allowed_entities": [c.__name__ for c in allowed_entities_cls] if allowed_entities_cls else None,
        "allowed_actions": [c.__name__ for c in effective_actions],
        "allowed_initial_actions": [c.__name__ for c in allowed_initial_cls] if allowed_initial_cls else None,
        "sample_single_object_per_class": sample_single_object_per_class,
        "action_weights": action_weights,
        "object_weights": {c.__name__: w for c, w in object_weights_cls.items()} if object_weights_cls else None,
        "weight_mode": weight_mode,
        "detail": detail.value,
        "output": str(output) if output else None,
    }
    if verbose or dry_run:
        _info("Resolved configuration:")
        typer.echo(yaml.safe_dump(plan, sort_keys=False, default_flow_style=False).rstrip())

    if dry_run:
        _ok("dry run: configuration is valid; no tasks generated.")
        raise typer.Exit()

    # ---- build the world ---------------------------------------------------------------- #
    if seed is not None:
        RDDLWorld.set_seed(seed)
    try:
        world = RDDLWorld(
            allowed_entities=allowed_entities_cls,
            allowed_actions=allowed_actions_cls,
            allowed_initial_actions=allowed_initial_cls,
            sample_single_object_per_class=sample_single_object_per_class,
            action_weights=action_weights,
            object_weights=object_weights_cls,
        )
    except Exception as exc:  # noqa: BLE001
        raise _fail(f"could not construct RDDLWorld: {exc}")
    if mode is not None:
        world.reset_weights(mode)

    # ---- generate ----------------------------------------------------------------------- #
    results: dict[int, list] = {}
    total = 0
    for length in sequence_lengths:
        records: list = []
        if method is Method.one_shot:
            for repeat in range(n_repeats):
                try:
                    task = world.sample_world(
                        sequence_length=length,
                        add_robots=add_robots,
                        retry_ad_infinitum=retry_ad_infinitum,
                    )
                except Exception as exc:  # noqa: BLE001
                    _warn(f"length {length}, repeat {repeat}: generation failed: {exc}")
                    continue
                record = _extract_task_record(task, detail)
                records.append(record)
                total += 1
                if print_tasks:
                    _print_task(length, repeat, record)
        else:  # recursive
            try:
                gen = world.recurse_generator(
                    sequence_length=length, add_robots=add_robots, n_samples_requested=n_samples
                )
                for idx, task in enumerate(gen):
                    record = _extract_task_record(task, detail)
                    records.append(record)
                    total += 1
                    if print_tasks:
                        _print_task(length, idx, record)
            except Exception as exc:  # noqa: BLE001
                _warn(f"length {length}: recursive generation failed: {exc}")
        results[length] = records
        _info(f"length {length}: generated {len(records)} task(s).")

    if total == 0:
        raise _fail("no tasks were generated. Check the domain, allowed actions, and lengths.")

    # ---- serialise ---------------------------------------------------------------------- #
    if output is not None:
        payload = {
            "metadata": {
                "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "domain": domain,
                "method": method.value,
                "detail": detail.value,
                "seed": seed,
                "sequence_lengths": sequence_lengths,
                "n_repeats": n_repeats if method is Method.one_shot else None,
                "n_samples": n_samples if method is Method.recursive else None,
                "total_tasks": total,
            },
            "tasks": results,
        }
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                _warn(f"overwriting existing file {output}.")
            with output.open("w") as fh:
                yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=False)
        except OSError as exc:
            raise _fail(f"could not write output {output}: {exc}")
        _ok(f"wrote {total} task(s) across {len(sequence_lengths)} length(s) to {output}.")
    else:
        _ok(f"generated {total} task(s) across {len(sequence_lengths)} length(s).")


def _print_task(length: int, index: int, record: dict) -> None:
    """Print one task record to the console, showing whatever detail it carries.

    Always prints the action sequence and per-step bound objects; if the record was built
    at ``full``/``detailed`` level it also prints the object set, the initial/final state
    predicates, and (detailed) the per-step post-state predicates.
    """
    typer.secho(f"\n[len {length} / #{index}] {' -> '.join(record['action_list'])}", fg=typer.colors.MAGENTA, bold=True)
    for step in record["sequence"]:
        objs = ", ".join(f"{k}={v}" for k, v in step["action_objects"].items())
        typer.echo(f"    {step['action']}  ({objs})")
        for pred in step.get("predicates", []):  # only present at 'detailed' level
            typer.echo(f"        {pred}")
    # object set + state predicates appear only at 'full'/'detailed' level
    if "initial_state" in record or "final_state" in record:
        typer.secho("    objects:", fg=typer.colors.CYAN)
        for obj_name, obj_type in record["all_objects"].items():
            typer.echo(f"        {obj_name}: {obj_type}")
        typer.secho("    initial state:", fg=typer.colors.CYAN)
        for pred in record.get("initial_state", []):
            typer.echo(f"        {pred}")
        typer.secho("    final state:", fg=typer.colors.CYAN)
        for pred in record.get("final_state", []):
            typer.echo(f"        {pred}")


def _print_inspection(entity_ns, action_ns, predicate_ns, Entity, AtomicAction, Predicate, Operand) -> None:
    def names(namespaces, base, exclude=None):
        out = set()
        for ns in namespaces:
            for n in dir(ns):
                obj = getattr(ns, n, None)
                if not isinstance(obj, type) or obj is base or not issubclass(obj, base):
                    continue
                if exclude is not None and issubclass(obj, exclude):
                    continue
                out.add(n)
        return sorted(out)

    _info("Entities:")
    typer.echo("    " + ", ".join(names(entity_ns, Entity)))
    _info("Actions:")
    typer.echo("    " + ", ".join(names(action_ns, AtomicAction)))
    _info("Predicates:")
    # AtomicAction is a Predicate subclass; exclude actions and the SpatialPredicate marker
    typer.echo("    " + ", ".join(n for n in names(predicate_ns, Predicate, exclude=AtomicAction) if n != "SpatialPredicate"))
    _info("Registered simulator mappings (provided by the domain):")
    mapping = getattr(Operand, "_Operand__MAPPING", {})
    typer.echo("    " + (", ".join(sorted(mapping.keys())) if mapping else "(none)"))


if __name__ == "__main__":
    app()
