def get_purpose_category(p):
    low = p.lower()
    if (
        "doctor" in low
        or "hospital" in low
        or "medical" in low
        or "emergency" in low
        or "dental" in low
        or "dentist" in low
        or "pharmacy" in low
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
    elif low == "bar" or "disco" in low or "club" in low or "nightlife" in low or "speakeasy" in low:
        return "nightlife"
    # new version: residential kept, work not --> leisure activities --> visiting friends?
    elif "residential" in low or "laundry" in low or "neighborhood" in low or "home (private)" in low:
        return "residential"
    elif (
        "entertain" in low
        or "theater" in low
        or "music" in low
        or "concert" in low
        or "museum" in low
        or (("art" in low) and ("department" not in low))
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
        or (("sport" in low) and ("shop" not in low))
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
    elif "church" in low or "mosque" in low or "spiritual" in low or "synagogue" in low:
        return "church"
    elif (
        low == "park"
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
        or "lookout" in low
        or "campground" in low
        or "ski" in low
        or "zoo" in low
    ):
        return "outdoor_city"
    # place restaurant and shop in the end so that the others are prioritized
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
        or low == "soup place"
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
        or "fish" in low
        or "coffee" in low
    ):
        return "restaurant"
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
    # leaving out leisure, work and home, hotel because they are too generic
    else:
        return "other"

