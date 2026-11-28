"""The 15 chain-of-rooms worlds (decision D13), grouped into families.

Every family varies ONE structural knob around a shared base world, so a
difference between worlds in a family can be attributed to that knob:

    Base: 4 rooms (c1, c2, c1, c2), 5x5 rooms, 2 doors, corridor 4

| Family    | Knob varied                         | Worlds                                   |
|-----------|-------------------------------------|------------------------------------------|
| Delay     | corridor length (steps between rooms)| L1, L4 (base), L8, L16                   |
| Matter    | how many of 4 rooms are case 1      | 0, 1, 2 (base), 3, 4                     |
| Doors     | doors per room b                    | 2 (base), 3, 4 (4 needs 7x7 rooms)       |
| Rooms     | number of rooms                     | 2, 4 (base), 6, 8                        |
| Size      | room interior size                  | 5 (base), 7                              |
| Hard      | several knobs at once (stress test) | 6 rooms, 7x7, 3 doors, corridor 8        |

Feedback placement (every room / every k rooms / end only) is NOT part of a
world: pass ``feedback_every=`` to ``make_env``/``gym.make`` so the same
world can be run under every placement.
"""
from __future__ import annotations

BASE = dict(rooms=("c1", "c2", "c1", "c2"), room_size=5, doors=2, corridor_len=4)


def _w(family: str, why: str, **overrides) -> dict:
    return dict(family=family, why=why, kwargs={**BASE, **overrides})


CHAIN_WORLDS: dict[str, dict] = {
    # --- Delay family: corridor length only ---------------------------------
    "MiniGrid-Proj-Chain-Delay-L1-v0": _w("delay", "shortest gap between choices", corridor_len=1),
    "MiniGrid-Proj-Chain-Base-v0": _w("base", "reference world; corridor 4"),
    "MiniGrid-Proj-Chain-Delay-L8-v0": _w("delay", "double the base gap", corridor_len=8),
    "MiniGrid-Proj-Chain-Delay-L16-v0": _w("delay", "long gap; tests multi-step credit", corridor_len=16),
    # --- Matter family: case-1 rooms out of 4 (same choices, same lengths) ---
    "MiniGrid-Proj-Chain-Matter-0of4-v0": _w("matter", "no choice matters (control)", rooms=("c2", "c2", "c2", "c2")),
    "MiniGrid-Proj-Chain-Matter-1of4-v0": _w("matter", "one choice matters", rooms=("c1", "c2", "c2", "c2")),
    "MiniGrid-Proj-Chain-Matter-3of4-v0": _w("matter", "three choices matter", rooms=("c1", "c1", "c1", "c2")),
    "MiniGrid-Proj-Chain-Matter-4of4-v0": _w("matter", "every choice matters", rooms=("c1", "c1", "c1", "c1")),
    # --- Doors family: branching -------------------------------------------
    "MiniGrid-Proj-Chain-Doors-3-v0": _w("doors", "three options per choice", doors=3),
    "MiniGrid-Proj-Chain-Doors-4-v0": _w("doors", "four options per choice (7x7 rooms to fit)", doors=4, room_size=7),
    # --- Rooms family: chain length ------------------------------------------
    "MiniGrid-Proj-Chain-Rooms-2-v0": _w("rooms", "shortest chain", rooms=("c1", "c2")),
    "MiniGrid-Proj-Chain-Rooms-6-v0": _w("rooms", "longer chain", rooms=("c1", "c2") * 3),
    "MiniGrid-Proj-Chain-Rooms-8-v0": _w("rooms", "longest chain", rooms=("c1", "c2") * 4),
    # --- Size family -----------------------------------------------------------
    "MiniGrid-Proj-Chain-Size-7-v0": _w("size", "bigger rooms: more filler steps inside rooms", room_size=7),
    # --- Hard --------------------------------------------------------------------
    "MiniGrid-Proj-Chain-Hard-v0": _w(
        "hard", "stress test combining knobs", rooms=("c1", "c2") * 3, room_size=7, doors=3, corridor_len=8
    ),
}


# Partial-cost worlds for test 3 (criticality bands). Every room has one
# correct door, but a wrong door costs only part of the return, so choices
# carry graded weight and SI has real intermediate values. Kept apart from the
# 15-world D13 catalog. The reversed world puts the same costs in the opposite
# order, to separate a choice's cost from its position in the chain.
COST_WORLDS: dict[str, dict] = {
    "MiniGrid-Proj-Chain-Cost-Graded-v0": _w(
        "cost", "wrong-door costs 1, 0.5, 0.25, 0.1", rooms=("c1", 0.5, 0.25, 0.1)
    ),
    "MiniGrid-Proj-Chain-Cost-Graded-Rev-v0": _w(
        "cost", "same costs, reversed order", rooms=(0.1, 0.25, 0.5, "c1")
    ),
    "MiniGrid-Proj-Chain-Cost-Graded-8-v0": _w(
        "cost", "eight cost levels, high to low", rooms=("c1", 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05)
    ),
}
