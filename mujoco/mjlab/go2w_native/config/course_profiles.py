"""Compatibility re-export of the single terrain-profile implementation.

The Go2W task used to carry its own copy of this module and the copy had drifted:
it still asked for ``block_size_range=(0.12, 0.36)`` (a *decreasing* difficulty
ramp), ``platform_height=0.28`` and ``course_start_difficulty=0.15`` (no course on
row 0), while the offline audit and every recent fix were applied to the MiniDog
copy.  The training terrain therefore never received the competition geometry.
Keeping one implementation here removes that failure mode: any change is seen by
the task, the rewards and the audit at the same time.
"""

from minidog_native.config.course_profiles import *  # noqa: F401,F403
from minidog_native.config.course_profiles import (  # noqa: F401
    BASE_TERRAIN_NAMES,
    SPECIAL_OBSTACLE_NAMES,
    STANDARD_OBSTACLE_NAMES,
    XML_MEDIUM_OBSTACLE,
    plum_course_terrain_generator,
)
