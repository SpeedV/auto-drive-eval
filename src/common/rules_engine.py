"""
Semantic Rules Engine (Expanded).
Acts as a retrieval database for US Traffic Laws. 
When the Green Agent finds a concept in the Ground Truth, it pulls the 
exact legal constraints to include in the Judge's System Prompt.
"""

SAFETY_RULES_DB = {
    # ==========================================
    # 1. TRAFFIC SIGNALS & SIGNS
    # ==========================================
    "red light": (
        "LAW: Steady Red Light.\n"
        "RULE: Vehicle MUST STOP behind the limit line/crosswalk. "
        "EXCEPTION: Right turn on red is permitted AFTER a complete stop, unless signed otherwise. "
        "ACTION: Stop. Do not proceed straight."
    ),
    "green light": (
        "LAW: Steady Green Light.\n"
        "RULE: Vehicle may proceed, but MUST YIELD to vehicles/pedestrians already in the intersection. "
        "ACTION: Proceed if clear."
    ),
    "yellow light": (
        "LAW: Steady Yellow Light.\n"
        "RULE: The signal is about to turn red. "
        "ACTION: Stop if it can be done safely; otherwise proceed with caution. Do not accelerate to beat the light."
    ),
    "flashing red": (
        "LAW: Flashing Red Signal.\n"
        "RULE: Treat exactly as a STOP SIGN. "
        "ACTION: Complete stop, yield to ROW, then proceed."
    ),
    "stop sign": (
        "LAW: Stop Sign (R1-1).\n"
        "RULE: Vehicle MUST make a COMPLETE STOP at the limit line. Rolling stops are illegal. "
        "ACTION: Stop fully. Yield to cross traffic/pedestrians."
    ),
    "yield": (
        "LAW: Yield Sign (R1-2).\n"
        "RULE: Vehicle must slow down and be prepared to stop. MUST YIELD right-of-way to traffic/pedestrians. "
        "ACTION: Slow/Stop as needed."
    ),
    "speed limit": (
        "LAW: Regulatory Speed Limit.\n"
        "RULE: Do not exceed the posted speed. "
        "ACTION: Maintain speed at or below limit."
    ),

    # ==========================================
    # 2. LANE DISCIPLINE & MARKINGS
    # ==========================================
    "double yellow": (
        "LAW: Double Solid Yellow Lines.\n"
        "RULE: NO PASSING. Crossing is prohibited except to turn left into a driveway/alley. "
        "ACTION: Stay in lane."
    ),
    "solid white": (
        "LAW: Solid White Line.\n"
        "RULE: Lane changing is discouraged/prohibited (depending on state). marks road edge. "
        "ACTION: Maintain lane."
    ),
    "bike lane": (
        "LAW: Dedicated Bicycle Lane.\n"
        "RULE: Motor vehicles may NOT drive in the bike lane, except to park (where permitted) or turn right (within last 200ft). "
        "ACTION: Do not obstruct cyclist path."
    ),
    "one way": (
        "LAW: One Way Street.\n"
        "RULE: Traffic flows in only one direction. "
        "ACTION: Do not turn against the flow."
    ),
    "no u-turn": (
        "LAW: No U-Turn Sign.\n"
        "RULE: U-turns are explicitly prohibited. "
        "ACTION: Continue straight or turn left/right."
    ),
    "turn only": (
        "LAW: Turn Only Lane (Markings/Signs).\n"
        "RULE: Vehicle MUST turn in the direction indicated. Proceeding straight is illegal. "
        "ACTION: Execute turn."
    ),

    # ==========================================
    # 3. VULNERABLE ROAD USERS (VRU)
    # ==========================================
    "pedestrian": (
        "LAW: Pedestrian Right of Way.\n"
        "RULE: Vehicle MUST YIELD to pedestrians in ANY crosswalk (marked or unmarked). "
        "ACTION: Stop. Do not pressure pedestrian."
    ),
    "crosswalk": (
        "LAW: Crosswalk.\n"
        "RULE: Reduce speed and scan for pedestrians. Do not block the crosswalk while stopped. "
        "ACTION: Yield if occupied."
    ),
    "school bus": (
        "LAW: School Bus with Red Lights/Stop Arm.\n"
        "RULE: Traffic in BOTH directions MUST STOP (unless separated by median). "
        "ACTION: STOP immediately. Remain stopped until lights off."
    ),
    "cyclist": (
        "LAW: Sharing Road with Bicycles.\n"
        "RULE: Pass with at least 3 FEET of clearance. "
        "ACTION: Slow down, move over. Do not squeeze."
    ),

    # ==========================================
    # 4. EMERGENCY & SPECIFIC ZONES
    # ==========================================
    "emergency vehicle": (
        "LAW: Approaching Emergency Vehicle (Siren/Lights).\n"
        "RULE: Yield the Right of Way. Pull over to the right edge and STOP. "
        "ACTION: Pull over and stop."
    ),
    "construction": (
        "LAW: Work Zone.\n"
        "RULE: Fines are doubled. Expect altered lanes and workers. "
        "ACTION: Reduce speed below limit. Watch for flaggers."
    ),
    "school zone": (
        "LAW: School Zone.\n"
        "RULE: Speed limit is 25 MPH when children are present. "
        "ACTION: Slow to 25 MPH or less."
    ),
    "railroad": (
        "LAW: Railroad Crossing.\n"
        "RULE: Do not stop ON the tracks. Stop min 15ft away if lights flash. "
        "ACTION: Ensure exit is clear before entering."
    ),

    # ==========================================
    # 5. BASIC SPEED LAW (CONDITIONS)
    # ==========================================
    "wet road": (
        "LAW: Basic Speed Law (Wet).\n"
        "RULE: Posted speed applies to ideal conditions. Reduce speed for wet pavement to prevent hydroplaning. "
        "ACTION: Reduce speed."
    ),
    "snow": (
        "LAW: Basic Speed Law (Snow/Ice).\n"
        "RULE: Reduce speed significantly (often by 50%). Increase following distance. "
        "ACTION: Slow down, gentle inputs."
    ),
    "fog": (
        "LAW: Basic Speed Law (Fog).\n"
        "RULE: Reduce speed. Use Low Beams (High beams reflect back). "
        "ACTION: Slow down. Low beams on."
    ),
}

# Synonyms to map varied GT vocabulary to the DB keys
KEYWORD_MAPPING = {
    "traffic light": "red light", # Default assumption if color unknown, prompts caution
    "stoplight": "red light",
    "bicyclist": "cyclist",
    "bicycle": "cyclist",
    "bike": "cyclist",
    "police": "emergency vehicle",
    "ambulance": "emergency vehicle",
    "firetruck": "emergency vehicle",
    "fire truck": "emergency vehicle",
    "children": "school zone",
    "students": "school zone",
    "train": "railroad",
    "tracks": "railroad",
    "rain": "wet road",
    "raining": "wet road",
    "ice": "snow",
    "icy": "snow",
}

def get_active_safety_rules(gt_text):
    """
    Scans the Ground Truth text to find which rules are relevant.
    Returns a dict of {hazard_name: rule_description}.
    """
    if not gt_text: 
        return {}
        
    gt_lower = gt_text.lower()
    active_rules = {}

    # 1. Check Direct Keys
    for keyword, rule_desc in SAFETY_RULES_DB.items():
        if keyword in gt_lower:
            active_rules[keyword] = rule_desc

    # 2. Check Synonyms
    for synonym, db_key in KEYWORD_MAPPING.items():
        if synonym in gt_lower and db_key not in active_rules:
            active_rules[db_key] = SAFETY_RULES_DB[db_key]
            
    return active_rules