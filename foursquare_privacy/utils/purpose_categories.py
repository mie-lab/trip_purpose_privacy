purpose_categories = {
    # WORK
    "office": "work",
    "conference": "work",
    "coworking": "work",
    "work": "work",
    # FOOD
    "food": "food",
    "restaurant": "food",
    "pizz": "food",
    "salad": "food",
    "ice cream": "food",
    "bakery": "food",
    "burger": "food",
    "sandwich": "food",
    "caf": "food",
    "diner": "food",
    "snack": "food",
    "steak": "food",
    "pub": "food",
    "tea": "food",
    "noodle": "food",
    "chicken": "food",
    "brewery": "food",
    "breakfast": "food",
    "beer": "food",
    "bbq": "food",
    "wings joint": "food",
    "burrito": "food",
    "taco": "food",
    # DOCTOR
    "doctor": "doctor",
    "hospital": "doctor",
    "medical": "doctor",
    "emergency": "doctor",
    "dental": "doctor",
    "dentist": "doctor",
    # TRAVEL
    "bus": "travel",
    "airport": "travel",
    "train": "travel",
    "taxi": "travel",
    "station": "travel",
    "metro": "travel",
    "travel": "travel",
    "ferry": "travel",
    "road": "travel",
    "subway": "travel",
    "platform": "travel",
    "rail": "travel",
    # shopping
    "store": "shop",
    "shop": "shop",
    "bank": "shop",
    "deli": "shop",
    "mall": "shop",
    "arcade": "shop",
    "boutique": "shop",
    "post": "shop",
    "market": "shop",
    "dealership": "shop",
    # nightlife
    "bar": "nightlife",
    "disco": "nightlife",
    "club": "nightlife",
    "nightlife": "nightlife",
    "speakeasy": "speakeasy",
    # residential
    "residential": "residential",
    "building": "residential",
    "neighborhood": "residential",
    "garden": "residential",
    "home": "residential",
    # arts
    "entertain": "arts",
    "theater": "arts",
    "music": "arts",
    "concert": "arts",
    "museum": "arts",
    "art": "arts",
    "temple": "arts",
    "historic": "arts",
    "shrine": "arts",
    "monument": "arts",
    "golf": "sports",
    "tennis": "sports",
    "dance": "sports",
    "sport": "sports",
    "gym": "sports",
    "hiking": "sports",
    "skating": "sports",
    "soccer": "sports",
    "basketball": "sports",
    "surf": "sports",
    "stadium": "sports",
    "baseball": "sports",
    "volleyball": "sports",
    "yoga": "sports",
    "hockey": "sports",
    "bowling": "sports",
    # outdoor
    "city": "outdoor",
    "park": "outdoor",
    "plaza": "outdoor",
    "bridge": "outdoor",
    "outdoors": "outdoor",
    "playground": "outdoor",
    "lake": "outdoor",
    "pier": "outdoor",
    "field": "outdoor",
    "harbor": "outdoor",
    "beach": "outdoor",
    "mountain": "outdoor",
    "river": "outdoor",
    # education
    "school": "education",
    "university": "education",
    "student": "education",
    "college": "education",
    # church
    "church": "church",
    "mosque": "church",
    "spiritual": "church",
}


# group purposes
further_agg_dict = {
    "arts": "leisure",
    "church": "leisure",
    "nightlife": "leisure",
    "travel": "leisure",
    "vacation": "leisure",
    "other": "leisure",
    "outdoor_city": "leisure",
    "residential": "leisure",
    "restaurant": "leisure",
    "shop": "shop",
    "doctor": "shop",
    "home": "home",
    "office": "work",
    "work": "work",
    "school": "work",
    "sport": "leisure",
}


def get_coarse_purpose_category(p):
    fine_cat = get_purpose_category(p)
    return further_agg_dict[fine_cat]


osm_poi_mapping = {
    "grocery": "shop",
    "disused:pub": "nightlife",
    "concert_hall": "arts",
    "research_institute": "education",
    "monastery": "church",
    "coffee;tea": "restaurant",
    "coffee": "restaurant",
    "meditation_centre": "church",
    "farm": "outdoor_city",
    "biergarten": "restaurant",
    "language_school": "education",
    "religion": "church",
    "stripclub": "nightlife",
    "music_venue": "arts",
    "surf": "sport",
    "grass": "outdoor_city",
    "outdoor_seating": "outdoor_city",
    "health_food": "restaurant",
    "supermarket,bakery": "shop",
    "tourism": "outdoor_city",
    "clothing store": "shop",
    "religious": "church",
    "plaza": "outdoor_city",
    "water_sports": "sport",
    "drinks": "nightlife",
    "farmers_market": "shop",
    "laundry;dry_cleaning": "residential",
    "bench": "outdoor_city",
    "restaurant": "restaurant",
    "school": "education",
    "fast_food": "restaurant",
    "place_of_worship": "church",
    "cafe": "restaurant",
    "convenience": "shop",
    "deli": "shop",
    "supermarket": "shop",
    "pharmacy": "shop",
    "social_facility": "education",
    "pub": "nightlife",
    "sports": "sport",
    "food_court": "restaurant",
    "university": "education",
    "college": "education",
    "ferry_terminal": "travel",
    "art": "arts",
    "nightclub": "nightlife",
    "mall": "shop",
    "bus_station": "travel",
    "clinic;doctors": "doctor",
    "cafe;bar": "restaurant",
}


def get_purpose_category(p):
    low = p.lower()
    if low == "office" or "conference" in low or "coworking" in low or "work" in low:
        return "work"
    elif (
        "food" in low
        or "restaurant" in low
        or "pizz" in low
        or "salad" in low
        or "ice cream" in low
        or "bakery" in low
        or "burger" in low
        or "sandwich" in low
        or "caf" in low
        or "diner" in low
        or "snack" in low
        or "steak" in low
        or "pub" in low
        or "tea" in low
        or "noodle" in low
        or "chicken" in low
        or "brewery" in low
        or "breakfast" in low
        or "beer" in low
        or "bbq" in low
        or "wings joint" in low
        or "burrito" in low
        or "taco" in low
    ):
        return "restaurant"
    elif (
        "doctor" in low
        or "hospital" in low
        or "medical" in low
        or "emergency" in low
        or "dental" in low
        or "dentist" in low
    ):
        return "doctor"
    elif (
        "bus" in low
        or "airport" in low
        or "train" in low
        or "taxi" in low
        or "station" in low
        or "metro" in low
        or "travel" in low
        or "ferry" in low
        or "road" in low
        or "subway" in low
        or "platform" in low
        or "rail" in low
    ):
        return "travel"
    elif (
        "store" in low
        or "shop" in low
        or "bank" in low
        or "deli" in low
        or "mall" in low
        or "arcade" in low
        or "boutique" in low
        or "post" in low
        or "market" in low
        or "dealership" in low
    ):
        return "shop"
    elif "bar" in low or "disco" in low or "club" in low or "nightlife" in low or "speakeasy" in low:
        return "nightlife"
    #     elif "home" in low:
    #         return "home"
    #     elif "residential" in low or "building" in low or "neighborhood" in low:
    #         return "residential"
    # new version:
    elif "residential" in low or "building" in low or "neighborhood" in low or "garden" in low or "home" in low:
        return "residential"
    elif (
        "entertain" in low
        or "theater" in low
        or "music" in low
        or "concert" in low
        or "museum" in low
        or "art" in low
        or "temple" in low
        or "historic" in low
        or "shrine" in low
        or "monument" in low
    ):
        return "arts"
    elif (
        "golf" in low
        or "tennis" in low
        or "dance" in low
        or "sport" in low
        or "gym" in low
        or "hiking" in low
        or "skating" in low
        or "soccer" in low
        or "basketball" in low
        or "surf" in low
        or "stadium" in low
        or "baseball" in low
        or "volleyball" in low
        or "yoga" in low
        or "hockey" in low
        or "bowling" in low
    ):
        return "sport"
    elif "school" in low or "college" in low or "university" in low or "student" in low:
        return "education"
    elif "church" in low or "mosque" in low or "spiritual" in low:
        return "church"
    #     elif "vacation" in low or "hotel" in low or "beach" in low or "tourist" in low or "bed &" in low:
    #         return "vacation"
    # TODO: what about vaction / hotel / beach etc?
    elif (
        "city" in low
        or "park" in low
        or "plaza" in low
        or "bridge" in low
        or "outdoors" in low
        or "playground" in low
        or "lake" in low
        or "pier" in low
        or "field" in low
        or "harbor" in low
        or "beach" in low
        or "mountain" in low
        or "river" in low
    ):
        return "outdoor_city"
    # TODO: leisure seems to generic now. where is leisure done?
    #     elif low == "leisure":
    #         return "leisure"
    else:
        return "other"

