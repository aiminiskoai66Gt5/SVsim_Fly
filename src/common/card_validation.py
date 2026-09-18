"""Static check: which cards carry effects the engine cannot execute.

The upstream text parser leaves gaps (a process that is only ``raw_action_text``,
a process without its ``value``, a target type with no handler...). At runtime
the engine reports each such hit (EffectProcessor._unimplemented). This module
finds them ahead of time so decks can optionally exclude those cards
(config ``env.exclude_unparsed_cards``) and reports can say how big the fully
executable card pool is.
"""

from typing import Any, List

from src.common.effect import Effect, Process
from src.common.enums import ProcessType

_processor = None


def _handlers():
    """Process / target handler tables of a throwaway EffectProcessor."""
    global _processor
    if _processor is None:
        from src.engine.effect_processor import EffectProcessor
        from src.engine.event_manager import EventManager
        _processor = EffectProcessor(EventManager())
    return _processor.process_handlers, _processor.target_handlers, _processor.REQUIRED_PROCESS_FIELDS


_LEADER_TARGETS = None


def _leader_targets():
    global _LEADER_TARGETS
    if _LEADER_TARGETS is None:
        from src.common.enums import TargetType
        _LEADER_TARGETS = {t for t in TargetType if "LEADER" in t.name}
    return _LEADER_TARGETS


def _card_exists(ref: str) -> bool:
    from src.common import card_data as cd
    from src.common.enums import TargetType
    if ref in TargetType.__members__:
        return True
    return cd.resolve_card_reference(ref) is not None


def referenced_cards(card: Any) -> List[Any]:
    """CardData objects that ``card``'s effects summon, transform into or add to hand."""
    from src.common import card_data as cd
    found = []

    def visit(node):
        if isinstance(node, (Effect, Process)):
            for key, val in node.attributes.items():
                if key == "value":
                    for item in (val if isinstance(val, list) else [val]):
                        if isinstance(item, cd.CardData):
                            found.append(item)
                        elif isinstance(item, str):
                            ref = cd.resolve_card_reference(item)
                            if ref is not None and ref is not card:
                                found.append(ref)
                visit(val)
        elif isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            for item in node.values():
                visit(item)

    visit(card.effects)
    return found


def _check_process(node: Any, out: List[str], path: str, process_handlers, target_handlers, required) -> None:
    if isinstance(node, dict):
        out.append(f"{path}: raw text only ({node.get('raw_action_text', '')!r})")
        return
    if not isinstance(node, Process):
        return
    ptype = node.attributes.get("process")
    post = node.attributes.get("post_action")
    if ptype is None:
        if post is None and "raw_action_text" in node.attributes:
            out.append(f"{path}: unparsed action {node.attributes['raw_action_text']!r}")
        elif "raw_action_text" in node.attributes:
            # A raw condition wrapping a parsed post_action: the engine runs the post_action
            # and ignores the condition text.
            out.append(f"{path}: unparsed condition {node.attributes['raw_action_text']!r}")
    elif isinstance(ptype, str):
        out.append(f"{path}: unknown process type {ptype!r}")
    elif ptype not in process_handlers and ptype not in (ProcessType.CHOOSE, ProcessType.DEFINE_VARIABLE):
        out.append(f"{path}: no handler for {ptype.name}")
    else:
        for field in required.get(ptype, ()):
            if node.attributes.get(field) is None:
                out.append(f"{path}: {ptype.name} lacks {field!r}")
        if ptype in (ProcessType.SUMMON, ProcessType.TRANSFORM):
            val = node.attributes.get("value")
            if isinstance(val, str) and not _card_exists(val):
                out.append(f"{path}: {ptype.name} refers to unknown card {val!r}")
        tgt = node.attributes.get("target")
        if ptype in (ProcessType.RETURN_TO_DECK, ProcessType.RETURN_TO_HAND) and tgt in _leader_targets():
            out.append(f"{path}: {ptype.name} targets a leader (needs a card)")
        if isinstance(tgt, str):
            out.append(f"{path}: unknown target type {tgt!r}")
        elif tgt is not None and tgt not in target_handlers:
            out.append(f"{path}: no target handler for {getattr(tgt, 'name', tgt)}")
    if post is not None:
        items = post if isinstance(post, list) else [post]
        for i, item in enumerate(items):
            _check_process(item, out, f"{path}.post[{i}]", process_handlers, target_handlers, required)


def _check_effect(effect: Any, out: List[str], path: str, tables) -> None:
    if not isinstance(effect, Effect):
        out.append(f"{path}: not an effect")
        return
    for i, proc in enumerate(effect.processes):
        _check_process(proc, out, f"{path}.p{i}", *tables)
    for i, choice in enumerate(effect.attributes.get("choices") or []):
        _check_effect(choice, out, f"{path}.choice{i}", tables)


def unparsed_reasons(card: Any, transitive: bool = True) -> List[str]:
    """Human-readable list of gaps in ``card``'s parsed effects (empty = fully executable).

    With ``transitive`` the cards it summons / transforms into / adds to hand are
    checked too, so a clean deck never brings an unexecutable token into play.
    """
    tables = _handlers()
    out: List[str] = []
    seen = set()

    def check(c):
        if c.card_id in seen:
            return
        seen.add(c.card_id)
        for i, effect in enumerate(c.effects):
            _check_effect(effect, out, f"{c.name}.e{i}", tables)
        if transitive:
            for ref in referenced_cards(c):
                check(ref)

    check(card)
    return out


def is_fully_parsed(card: Any) -> bool:
    return not unparsed_reasons(card)


def summarize_pool(cards) -> dict:
    """Counts for a card collection: total, fully parsed, and reason histogram."""
    from collections import Counter
    total = 0
    clean = 0
    kinds = Counter()
    for card in cards:
        total += 1
        reasons = unparsed_reasons(card)
        if not reasons:
            clean += 1
        for r in reasons:
            kinds[r.split(": ", 1)[1].split(" ")[0] + " " + r.split(": ", 1)[1].split(" ")[1]] += 1
    return {"total": total, "fully_parsed": clean, "reasons": dict(kinds.most_common())}
